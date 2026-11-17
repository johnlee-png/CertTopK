"""Runs the Everest baseline (src_extension_selective_topk_everest.run, unchanged)
with one process per (domain, seed). Everest recomputes E[X_f] for every
uncleaned tuple after each call, so its cost grows with query size squared and
is single-threaded; on the large domains (IEEE 1,000-row and Lending Club
5,000-row queries) running seeds in parallel is what makes 20 seeds feasible.
Results are merged into selective_topk_everest_results/summary like the serial
script (rows of other domains are kept).
Usage: py src_extension_selective_topk_everest_parallel.py <n_seeds> <domains> [workers]
"""
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import pandas as pd

import src_extension_selective_topk_everest as ev

OUT = Path("paper/src_extension")


def task(args):
    domain, seed = args
    t0 = time.time()
    rows = ev.run(domain, seed)
    return rows, f"[{domain}] seed {seed} done in {time.time() - t0:.0f}s"


def main():
    n_seeds, domains = int(sys.argv[1]), sys.argv[2].split(",")
    workers = int(sys.argv[3]) if len(sys.argv) > 3 else 10
    path = OUT / "selective_topk_everest_results.csv"
    old = pd.read_csv(path)
    # Resume: "--resume" keeps the rows of (domain, seed) pairs already in the file and runs only the rest.
    resume = "--resume" in sys.argv
    done = set(zip(old["domain"], old["seed"])) if resume else set()
    keep = old if resume else old[~old["domain"].isin(domains)]
    todo = [(d, s) for d in domains for s in range(n_seeds) if (d, s) not in done]
    print(f"{len(done)} (domain, seed) pairs kept, {len(todo)} to run", flush=True)
    rows = []
    with Pool(workers) as pool:
        for r, msg in pool.imap_unordered(task, todo):
            rows += r
            print(msg, flush=True)
            pd.concat([keep, pd.DataFrame(rows)]).to_csv(path, index=False)
    t = pd.concat([keep, pd.DataFrame(rows)], ignore_index=True).sort_values(["domain", "seed", "k_fraction", "thres"])
    t.to_csv(path, index=False)
    t.to_json(OUT / "selective_topk_everest_results.json", orient="records", indent=2)
    summ = t.groupby(["domain", "k_fraction", "thres"]).agg(
        seeds=("seed", "nunique"), miss_mean=("test_miss", "mean"), exact_rate=("exact_rate", "mean"),
        claimed_conf=("claimed_conf", "mean"), calls_mean=("test_call_fraction", "mean")).reset_index()
    summ.to_csv(OUT / "selective_topk_everest_summary.csv", index=False)
    pd.set_option("display.width", 250)
    print(summ.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
