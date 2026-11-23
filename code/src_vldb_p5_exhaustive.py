"""Exhaustive check of the wrong-side bound (Theorem 5) including ties at the top-k boundary.

For every n <= N_MAX, every cheap and expensive score vector over a small value set (so ties, including ties between
the k-th and (k+1)-th expensive scores, occur in all patterns), every k < n, and every refined set S, checks
L(S) <= #{i not in S : wrong-side}/k and the sharper L(S) <= max(#unrefined wrong-side members,
#unrefined wrong-side non-members)/k with the common tie rule (smaller index first). Reports the number of cases,
how many hit the bound with equality, and how many have a degenerate cut-off (k-th = (k+1)-th expensive score).
Usage: py src_vldb_p5_exhaustive.py
"""
import itertools

import numpy as np

import src_extension_selective_topk_crc as crc
from src_vldb_p4 import wrong_side

CASES = [(3, 4), (4, 4), (5, 3)]          # (n, number of distinct values)


def main():
    total = equal = equal_max = degenerate = 0
    for n, v in CASES:
        masks = np.array(list(itertools.product([False, True], repeat=n)))
        vals = list(itertools.product(range(v), repeat=n))
        for ue in vals:
            ue = np.array(ue, float)
            srt = np.sort(ue)[::-1]
            for k in range(1, n):
                ref = np.zeros(n, bool)
                ref[crc.topk(ue, k)] = True
                deg = srt[k - 1] == srt[k]
                for uc in vals:
                    uc = np.array(uc, float)
                    wrong = wrong_side(uc, ue[None, :], k)[0]
                    for S in masks:
                        ans = np.zeros(n, bool)
                        ans[crc.topk(np.where(S, ue, uc), k)] = True
                        loss = k - (ans & ref).sum()
                        bound = (wrong & ~S).sum()
                        bmax = max((wrong & ~S & ref).sum(), (wrong & ~S & ~ref).sum())
                        assert loss <= bmax <= bound, (uc, ue, k, S)
                        total += 1
                        equal += loss == bound
                        equal_max += loss == bmax
                        degenerate += deg
        print(f"n={n}, values 0..{v - 1}: done", flush=True)
    print(f"bound holds in all {total:,} (score vectors, k, S) cases; equality in {equal:,}; "
          f"max form holds too, equality in {equal_max:,}; degenerate cut-off (k-th = (k+1)-th expensive score) in {degenerate:,}")


if __name__ == "__main__":
    main()
