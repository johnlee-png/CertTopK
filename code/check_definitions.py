"""Checks that the definitions in the paper match the implementation, including boundary cases.

Each check prints PASS/FAIL. Run: py -X utf8 check_definitions.py
Covered: s_hat positivity (floor), priorities defined for every input, cut-off ties (boundary contact refined first),
Lemma 1 crossing condition, normalized regret with a zero denominator, loss in [0,1], envelope properties,
full-refinement fallback (lambda_G = +inf gives zero loss), nested refined sets, the CRC threshold rule and its
fallback, raw-loss LTT never refining more than envelope LTT, and P3/P3c determinism per split stream.
"""
import numpy as np

import src_extension_selective_topk_crc as crc
from src_extension_selective_topk_baselines import crc_index, curves, ltt_index, topk_mask
from src_extension_selective_topk_p3_extra import iid_sampler, p3_priority

RESULTS = []


def check(name, ok, detail=""):
    RESULTS.append((name, bool(ok)))
    print(f"{'PASS' if ok else 'FAIL'}  {name}  {detail}")


def synthetic_split(n_q=30, n=60, seed=0, ties=False, zero_resid=False):
    rng = np.random.default_rng(seed)
    N = n_q * n
    pc = rng.uniform(0.05, 0.95, N)
    pe = pc if zero_resid else np.clip(pc + rng.normal(0, 0.08, N), 0, 1)
    R, C = rng.uniform(1, 5, N), rng.uniform(1, 5, N)
    if ties:                                    # make many cheap utilities identical
        pc = np.round(pc, 1)
        R, C = np.full(N, 2.0), np.full(N, 2.0)
    return {"pc": pc, "pe": pe, "R": R, "C": C, "xc": rng.normal(size=(N, 3)), "qid": np.repeat(np.arange(n_q), n)}


def main():
    fit = synthetic_split(seed=1)
    scale = crc.fit_scale(fit)
    s = scale(synthetic_split(seed=2))
    check("s_hat > 0 on new data (floor = 5% quantile of |Pe-Pc| + 1e-6)", np.all(s > 0), f"min={s.min():.2e}")
    fit0 = synthetic_split(seed=3, zero_resid=True)
    s0 = crc.fit_scale(fit0)(fit0)
    check("s_hat > 0 even when every residual is 0", np.all(s0 > 0), f"min={s0.min():.2e}")

    d = synthetic_split(seed=4, ties=True)
    qs = crc.query_arrays(d, scale, 0.10, 0, stream=2)
    finite = all(np.all(np.isfinite(q["pri"][n])) for q in qs for n in ("P0", "P1", "P2", "TOPM", "RND"))
    check("all priorities finite with ties and equal budgets", finite)
    tie_ok, n_tie = True, 0
    for q in qs:
        k = q["k"]
        srt = np.sort(q["uc"])[::-1]
        if srt[k - 1] == srt[k]:                  # tie at the cut-off: boundary contact
            n_tie += 1
            tie_ok &= np.all(q["pri"]["P2"][q["uc"] == srt[k - 1]] == 0)
    check("tie at the cut-off gives P2 = 0 (refined first; boundary-contact rule)", tie_ok and n_tie > 0, f"{n_tie} queries")

    rng = np.random.default_rng(5)
    ok = True
    for _ in range(2000):
        uc, theta, budget, dp = rng.normal(), rng.normal(), rng.uniform(0.1, 5), rng.normal(0, 0.3)
        ue = uc + budget * dp
        crosses = (ue - theta) * np.sign(uc - theta) <= 0 if uc != theta else True
        cond = dp * np.sign(theta - uc) >= abs(uc - theta) / budget
        ok &= crosses == cond
    check("Lemma 1: crossing iff (Pe-Pc) sgn(theta-Uc) >= pi_i (2,000 random cases)", ok)

    q = {"uc": np.array([3.0, 2.0, 1.0, 0.0]), "ue": np.array([1.0, 1.0, 1.0, 1.0]), "k": 2, "pri": {"P2": np.arange(4.0)}}
    grid = np.array([-np.inf, 0.5, 1.5, np.inf])
    L, Rg, F = curves(q, "P2", grid)
    check("regret is 0 when all expensive utilities are equal (zero denominator)", np.all(Rg == 0), f"R={Rg}")

    d = synthetic_split(seed=6)
    qs = crc.query_arrays(d, scale, 0.10, 0, stream=1)
    pooled = np.concatenate([q["pri"]["P2"] for q in qs])
    grid = np.unique(np.concatenate([[-np.inf], np.quantile(pooled, np.linspace(0, 1, 51)), [np.inf]]))
    Ls, Fs, Rs, envs, nested = [], [], [], [], True
    for q in qs:
        L, Rg, F = curves(q, "P2", grid)
        Ls.append(L), Fs.append(F), Rs.append(Rg)
        envs.append(crc.envelope(L))
        sets = [q["pri"]["P2"] <= lam for lam in grid]
        nested &= all(np.all(a <= b) for a, b in zip(sets[:-1], sets[1:]))
    Ls, Fs, Rs, envs = map(np.array, (Ls, Fs, Rs, envs))
    check("loss and regret in [0,1]", Ls.min() >= 0 and Ls.max() <= 1 and Rs.min() >= 0 and Rs.max() <= 1)
    check("refined sets nested in lambda", nested)
    check("envelope non-increasing and >= loss", np.all(np.diff(envs, axis=1) <= 1e-12) and np.all(envs >= Ls - 1e-12))
    check("lambda_G = +inf refines everything and has zero loss", np.all(Fs[:, -1] == 1) and np.all(Ls[:, -1] == 0)
          and np.all(envs[:, -1] == 0))
    m = len(envs)
    for eps in (0.05, 0.10):
        j = crc_index(envs, eps)
        crit = m / (m + 1) * envs.mean(axis=0) + 1 / (m + 1)
        ok = (crit[j] <= eps or j == len(grid) - 1) and np.all(crit[:j] > eps)
        check(f"CRC picks the smallest qualifying grid point (eps={eps})", ok, f"j={j}")
    tiny = crc_index(envs[:3], 0.05)
    check("CRC falls back to lambda_G when 1/(m+1) > eps", tiny == len(grid) - 1, f"m=3 -> j={tiny}")
    raw_ok = all(ltt_index(Ls, e) <= ltt_index(envs, e) for e in (0.05, 0.10, 0.2))
    check("raw-loss LTT never refines more than envelope LTT", raw_ok)

    z = rng.normal(size=500)
    q0 = qs[0]
    w = np.ones(len(q0["uc"]))
    a = p3_priority(q0["uc"], w, q0["k"], q0["pri"]["P2"], 200, iid_sampler(z), np.random.default_rng([0, 2, 99]))
    b = p3_priority(q0["uc"], w, q0["k"], q0["pri"]["P2"], 200, iid_sampler(z), np.random.default_rng([0, 2, 99]))
    c = p3_priority(q0["uc"], w, q0["k"], q0["pri"]["P2"], 200, iid_sampler(z), np.random.default_rng([0, 1, 99]))
    check("P3 is deterministic for a split stream and differs across streams", np.array_equal(a, b) and not np.array_equal(a, c))
    check("P3 values in [0,1] plus tie-break below 1/B", a.min() >= 0 and a.max() <= 1 + 1e-6)
    print(f"\n{sum(o for _, o in RESULTS)} of {len(RESULTS)} checks passed")


if __name__ == "__main__":
    main()
