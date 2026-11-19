"""Cutoff-tie rate of the simulated top-k (how often the old `>= kth` rule selected more than k) per script and domain,
from the CERTTOPK_TIE_LOG lines written by src_vldb_topk_ties.topk_mask_batch during the reruns.
Reads paper/src_extension/tie_log/ (set CERTTOPK_TIE_LOG to that folder during the reruns); writes paper/src_extension/tie_rates.csv."""
from pathlib import Path

import pandas as pd

D = Path("paper/src_extension")
cols = ["script", "domain", "split", "B", "n", "k", "tied_rows", "changed_entries"]
t = pd.concat([pd.read_csv(p, names=cols) for p in sorted((D / "tie_log").glob("ties_*.csv"))], ignore_index=True)
t["entries"] = t.B * t.n
g = t.groupby(["script", "domain"]).agg(calls=("B", "size"), scenarios=("B", "sum"), tied_scenarios=("tied_rows", "sum"),
                                         queries_with_tie=("tied_rows", lambda x: (x > 0).sum()),
                                         changed=("changed_entries", "sum"), entries=("entries", "sum")).reset_index()
g["scenario_tie_rate"] = g.tied_scenarios / g.scenarios
g["query_tie_rate"] = g.queries_with_tie / g.calls
g["changed_share"] = g.changed / g.entries
g.to_csv(D / "tie_rates.csv", index=False)
with pd.option_context("display.width", 200):
    print(g[["script", "domain", "calls", "scenario_tie_rate", "query_tie_rate", "changed_share"]].to_string(index=False))
