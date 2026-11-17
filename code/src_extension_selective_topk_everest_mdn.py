"""M3: Everest with a mixture density model of the cheap score's error (closer to
the CMDN of Lai et al., SIGMOD 2021) instead of the residual-based model.

Everest's guarantee is conditional on the score distribution F_f it assumes for
every uncleaned tuple. src_extension_selective_topk_everest.py builds F_f from
one shared distribution of normalized residuals times a per-tuple scale. Here a
small mixture density network (MDN) outputs, for each tuple, a mixture of
N_COMP Gaussians for D = P_e - P_c given the cheap information (x, P_c, R, C):
  F_f(t) = P(U_e <= t) = sum_j w_j Phi(((t - U_c) / (R + C) - mu_j) / sigma_j),
using the affine identity U_e = U_c + (R + C) D. The MDN is trained on one half
of the fit split's queries and early-stopped on the other half (the calibration
and test queries are not used). Everything else (selection by E[X_f],
confidence, stopping rule, start from the cheap top-k, answers, metrics) is the
code of the residual-based reimplementation.
Writes paper/src_extension/selective_topk_everest_mdn_{results,summary}.csv.
Usage: py src_extension_selective_topk_everest_mdn.py <n_seeds> <domains> [workers]
"""
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import ndtr

import src_extension_selective_topk_crc as crc
from src_extension_selective_topk_diagnostic import utility
from src_extension_selective_topk_everest import N_INT, THRESHOLDS

N_COMP, HIDDEN, EPOCHS, BATCH, LR, PATIENCE = 3, 64, 60, 1024, 1e-3, 6
OUT = Path("paper/src_extension")


def features(d):
    return np.column_stack([d["xc"], d["pc"], d["R"], d["C"]]).astype(np.float32)


def fit_mdn(fitd, seed):
    import torch
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    X = features(fitd)
    D = (fitd["pe"] - fitd["pc"]).astype(np.float32)
    uq = np.unique(fitd["qid"])
    a = np.isin(fitd["qid"], np.random.default_rng(seed + 11).permutation(uq)[: len(uq) // 2])
    mu_x, sd_x = np.nanmean(X[a], axis=0), np.nanstd(X[a], axis=0) + 1e-6
    sd_d = float(D[a].std()) + 1e-9

    def prep(x):
        return torch.tensor(np.nan_to_num((x - mu_x) / sd_x), dtype=torch.float32)

    net = torch.nn.Sequential(torch.nn.Linear(X.shape[1], HIDDEN), torch.nn.ReLU(),
                              torch.nn.Linear(HIDDEN, HIDDEN), torch.nn.ReLU(),
                              torch.nn.Linear(HIDDEN, 3 * N_COMP))

    def params(out):
        logit_w, mu, log_s = out.split(N_COMP, dim=1)
        return torch.log_softmax(logit_w, dim=1), mu, torch.nn.functional.softplus(log_s) + 1e-3

    def nll(x, d):
        lw, mu, s = params(net(x))
        z = (d[:, None] - mu) / s
        return -torch.logsumexp(lw - 0.5 * z ** 2 - torch.log(s) - 0.9189385, dim=1).mean()

    xa, da = prep(X[a]), torch.tensor(D[a] / sd_d)
    xb, db = prep(X[~a]), torch.tensor(D[~a] / sd_d)
    opt = torch.optim.Adam(net.parameters(), lr=LR)
    best, best_state, bad = np.inf, None, 0
    for _ in range(EPOCHS):
        perm = torch.randperm(len(xa))
        for i in range(0, len(xa), BATCH):
            b = perm[i:i + BATCH]
            opt.zero_grad()
            loss = nll(xa[b], da[b])
            loss.backward()
            opt.step()
        with torch.no_grad():
            v = float(nll(xb, db))
        if v < best - 1e-4:
            best, best_state, bad = v, {k: t.clone() for k, t in net.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    net.load_state_dict(best_state)

    def predict(d):
        with torch.no_grad():
            lw, mu, s = params(net(prep(features(d))))
        return np.exp(lw.numpy()), mu.numpy() * sd_d, s.numpy() * sd_d
    return predict, best


def everest_query_mdn(uc, ue, budget, w, mu, sig, k, thres):
    """ev.everest_query with F_f from the per-tuple Gaussian mixture of D."""
    n = len(uc)
    cleaned = np.zeros(n, dtype=bool)
    cleaned[crc.topk(uc, k)] = True

    def cdf(idx, t):
        d = (np.asarray(t)[None, :] - uc[idx, None]) / budget[idx, None]          # (u, T)
        return (w[idx, None, :] * ndtr((d[:, :, None] - mu[idx, None, :]) / sig[idx, None, :])).sum(axis=2)
    key, u_idx, Fg, logF = None, None, None, None
    while True:
        c_idx = np.flatnonzero(cleaned)
        order = c_idx[np.argsort(-ue[c_idx], kind="stable")]
        answer = order[:k]
        s_k, s_p = ue[order[k - 1]], ue[order[k - 2]] if k >= 2 else ue[order[k - 1]]
        if (s_k, s_p) != key:
            key = (s_k, s_p)
            u_idx = np.flatnonzero(~cleaned)
            if len(u_idx) == 0:
                return answer, int(cleaned.sum()), 1.0
            grid = np.linspace(s_k, s_p, N_INT) if s_p > s_k else np.array([s_k, s_k])
            Fg = np.clip(cdf(u_idx, grid), 1e-300, 1.0)
            logF = np.log(Fg)
        if len(u_idx) == 0:
            return answer, int(cleaned.sum()), 1.0
        conf = float(np.exp(logF[:, 0].sum()))
        if conf >= thres:
            return answer, int(cleaned.sum()), conf
        logG = logF.sum(axis=0)[None, :] - logF
        G = np.exp(logG)
        mass = np.diff(Fg, axis=1)
        mid = 0.5 * (G[:, 1:] + G[:, :-1])
        ex = Fg[:, 0] * G[:, 0] + (mass * mid).sum(axis=1) + (1 - Fg[:, -1]) * G[:, -1]
        pos = int(np.argmax(ex))
        cleaned[u_idx[pos]] = True
        u_idx, Fg, logF = np.delete(u_idx, pos), np.delete(Fg, pos, axis=0), np.delete(logF, pos, axis=0)


def task(args):
    domain, seed = args
    t0 = time.time()
    data = crc.DOMAINS[domain](seed)
    predict, val_nll = fit_mdn(data["fit"], seed)
    d = data["test"]
    U_c, U_e = utility(d["pc"], d["R"], d["C"]), utility(d["pe"], d["R"], d["C"])
    budget = d["R"] + d["C"]
    w, mu, sig = predict(d)
    rows = []
    for k_frac in crc.K_FRACTIONS:
        for thres in THRESHOLDS:
            miss, calls, exact, confs = [], [], [], []
            for q in np.unique(d["qid"]):
                ix = np.flatnonzero(d["qid"] == q)
                k = max(1, int(round(k_frac * len(ix))))
                if k >= len(ix):
                    continue
                ans, n_clean, conf = everest_query_mdn(U_c[ix], U_e[ix], budget[ix], w[ix], mu[ix], sig[ix], k, thres)
                ref = set(crc.topk(U_e[ix], k).tolist())
                hit = len(ref & set(ans.tolist()))
                miss.append(1 - hit / k)
                exact.append(float(hit == k))
                calls.append(n_clean / len(ix))
                confs.append(conf)
            rows.append({"domain": domain, "seed": seed, "k_fraction": k_frac, "thres": thres, "model": "mdn",
                         "val_nll": val_nll, "n_test_queries": len(miss), "test_miss": float(np.mean(miss)),
                         "exact_rate": float(np.mean(exact)), "claimed_conf": float(np.mean(confs)),
                         "test_call_fraction": float(np.mean(calls))})
    return rows, f"[{domain}] seed {seed} done in {time.time() - t0:.0f}s"


def summarize():
    t = pd.read_csv(OUT / "selective_topk_everest_mdn_results.csv")
    g = t.assign(h=t.exact_rate * t.n_test_queries, c=t.claimed_conf * t.n_test_queries) \
        .groupby(["domain", "k_fraction", "thres"]).agg(n=("n_test_queries", "sum"), h=("h", "sum"), c=("c", "sum"),
                                                         seeds=("seed", "nunique"), calls=("test_call_fraction", "mean"),
                                                         miss=("test_miss", "mean")).reset_index()
    g["claimed"], g["exact"] = g.c / g.n, g.h / g.n
    g["z_pooled"] = (g.exact - g.claimed) / np.sqrt(g.claimed * (1 - g.claimed) / g.n)
    dd = t.assign(diff=t.exact_rate - t.claimed_conf).groupby(["domain", "k_fraction", "thres"])["diff"]
    g["t_seed"] = (dd.mean() / (dd.std(ddof=1) / np.sqrt(dd.count()))).to_numpy()
    g = g.drop(columns=["h", "c"])
    g.to_csv(OUT / "selective_topk_everest_mdn_summary.csv", index=False)
    print(g.round(3).to_string(index=False))
    print("violations: z<-3", int((g.z_pooled < -3).sum()), " t<-3", int((g.t_seed < -3).sum()), "of", len(g))


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "summarize":
        return summarize()
    n_seeds, domains = int(sys.argv[1]), sys.argv[2].split(",")
    workers = int(sys.argv[3]) if len(sys.argv) > 3 else 4
    path = OUT / "selective_topk_everest_mdn_results.csv"
    rows = pd.read_csv(path).to_dict("records") if path.exists() else []
    done = {(r["domain"], r["seed"]) for r in rows}
    todo = [(d, s) for d in domains for s in range(n_seeds) if (d, s) not in done]
    with Pool(workers) as pool:
        for r, msg in pool.imap_unordered(task, todo):
            rows += r
            print(msg, flush=True)
            pd.DataFrame(rows).to_csv(path, index=False)
    summarize()


if __name__ == "__main__":
    main()
