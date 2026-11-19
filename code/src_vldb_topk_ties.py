"""Exact-k top-k membership for simulated score matrices, with the same tie rule as crc.topk.

crc.topk breaks ties at the k-th score by the smaller candidate index. The ICDE-era simulations
(P3, P3c, lossaware, hc_time, order_select_final, p3c_curves, temporal) used `u_sim >= kth`,
which keeps every tied candidate and so can select more than k. topk_mask_batch fixes that.
"""
import inspect
import os

import numpy as np

import src_extension_selective_topk_crc as crc

# Set CERTTOPK_TIE_LOG=<dir> to append one line per call: how often the old `>= kth` rule over-selected.
TIE_LOG = os.environ.get("CERTTOPK_TIE_LOG")


def _log_ties(u, k, mask, old):
    domain = split = ""
    for fr in inspect.stack()[2:8]:
        loc = fr.frame.f_locals
        domain = domain or str(loc.get("domain", ""))
        split = split or str(loc.get("split", ""))
    row_tied = (old.sum(axis=1) > k)
    line = (f"{os.environ.get('CERTTOPK_TIE_TAG', '')},{domain},{split},{u.shape[0]},{u.shape[1]},{k},"
            f"{int(row_tied.sum())},{int((old != mask).sum())}\n")
    with open(os.path.join(TIE_LOG, f"ties_{os.getpid()}.csv"), "a") as fh:
        fh.write(line)


def topk_mask_batch(u, k):
    """Boolean (B, n) mask of the k largest entries of each row of u; ties at the k-th value go to
    the smaller column index, matching crc.topk row by row. Every row has exactly k True entries."""
    u = np.asarray(u, float)
    kth = -np.partition(-u, k - 1, axis=1)[:, k - 1:k]
    above = u > kth
    tied = u == kth
    need = k - above.sum(axis=1, keepdims=True)
    mask = above | (tied & (np.cumsum(tied, axis=1) <= need))
    if TIE_LOG:
        _log_ties(u, k, mask, u >= kth)
    return mask


def flip_prob(u_sim, in_cheap, k):
    """Per-candidate probability that simulated top-k membership differs from the cheap top-k."""
    in_sim = topk_mask_batch(u_sim, k)
    return (in_sim != in_cheap[None, :]).mean(axis=0)


def cutoff_tie_rate(u_sim, k):
    """Fraction of rows whose k-th value is tied with an entry outside the exact top-k
    (the rows where `u_sim >= kth` selects more than k)."""
    kth = -np.partition(-u_sim, k - 1, axis=1)[:, k - 1:k]
    return float(((u_sim >= kth).sum(axis=1) > k).mean())


def _self_test():
    # Professor's counterexample: k=1, cheap (1.8, 1.6, 1.4), every simulated utility 2.
    uc, k = np.array([1.8, 1.6, 1.4]), 1
    in_cheap = np.zeros(3, bool)
    in_cheap[crc.topk(uc, k)] = True
    u_sim = np.full((1, 3), 2.0)
    old = np.where(in_cheap[None, :], u_sim < 2.0, u_sim >= 2.0).mean(axis=0)
    new = flip_prob(u_sim, in_cheap, k)
    assert old.tolist() == [0, 1, 1], old
    assert new.tolist() == [0, 0, 0], new
    # Row-by-row agreement with crc.topk on heavily tied random integers.
    rng = np.random.default_rng(0)
    for _ in range(2000):
        n = int(rng.integers(2, 30))
        k = int(rng.integers(1, n + 1))
        u = rng.integers(0, 4, size=(5, n)).astype(float)
        m = topk_mask_batch(u, k)
        assert (m.sum(axis=1) == k).all()
        for b in range(5):
            ref = np.zeros(n, bool)
            ref[crc.topk(u[b], k)] = True
            assert (m[b] == ref).all()
    print("self-test passed: counterexample", old.tolist(), "->", new.tolist(), "; 10000 random rows match crc.topk")


if __name__ == "__main__":
    _self_test()
