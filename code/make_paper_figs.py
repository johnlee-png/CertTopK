"""English figures for the paper (vector PDF) from the result CSVs.
Usage: py -X utf8 code/make_paper_figs.py   (from the artifact root)
Writes figures/*.pdf
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
RES = HERE.parent / "results"
FIG = HERE.parent / "figures"
FIG.mkdir(exist_ok=True)
plt.rcParams.update({"text.usetex": True, "font.family": "serif", "font.size": 8,
                     "text.latex.preamble": r"\usepackage[tt=false]{libertine}\usepackage[libertine]{newtxmath}",
                     "axes.edgecolor": "#444444", "axes.linewidth": 0.6, "axes.labelcolor": "#222222",
                     "xtick.color": "#333333", "ytick.color": "#333333", "xtick.major.width": 0.6,
                     "ytick.major.width": 0.6, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
                     "axes.grid": True, "grid.color": "#e6e6e6", "grid.linewidth": 0.5,
                     "axes.spines.top": False, "axes.spines.right": False})
NAME = {"soccer": "Soccer", "diabetes": "Diabetes", "home_credit": "Home Credit", "kdd98": "KDD98",
        "telco": "Telco", "ieee_fraud": "IEEE-CIS", "lending_club": "Lending Club", "lc_llm": "LC-LLM"}
SEVEN = ["soccer", "diabetes", "home_credit", "kdd98", "telco", "ieee_fraud", "lending_club"]
STYLE = {"P2": ("#2a78d6", "-", 1.6, "P2 (ours)"), "EVEREST": ("#eb6834", "--", 1.3, "Everest order"),
         "TOPM": ("#1baf7a", "-.", 1.0, "TOPM"), "P1": ("#eda100", ":", 1.1, "P1"),
         "P0": ("#e87ba4", (0, (5, 2, 1, 2)), 0.9, "P0"), "RND": ("#898781", (0, (1, 2)), 0.8, "Random")}

# ---- Fig. 1: guarantee (left) and Everest claims (right)
crc = pd.read_csv(RES / "selective_topk_crc_summary.csv")
ev = pd.read_csv(RES / "selective_topk_everest_results.csv")
ev = ev[ev.domain.isin(SEVEN)]
g = ev.assign(h=ev.exact_rate * ev.n_test_queries, c=ev.claimed_conf * ev.n_test_queries) \
    .groupby(["domain", "k_fraction", "thres"]).agg(n=("n_test_queries", "sum"), h=("h", "sum"), c=("c", "sum")).reset_index()
g["exact"], g["claimed"] = g.h / g.n, g.c / g.n
g["z"] = (g.exact - g.claimed) / np.sqrt(g.claimed * (1 - g.claimed) / g.n)
fig, (a, b) = plt.subplots(1, 2, figsize=(7.0, 2.2), gridspec_kw={"width_ratios": [1.25, 1]})
for i, d in enumerate(SEVEN):
    r = (crc[crc.domain == d].miss_mean / crc[crc.domain == d].eps).to_numpy()
    a.scatter(r, np.full(len(r), i) + np.random.default_rng(i).uniform(-0.2, 0.2, len(r)), s=7, color="#2a78d6", alpha=0.7, linewidths=0)
a.set_yticks(range(len(SEVEN)), [NAME[d] for d in SEVEN])
a.set_xlim(0, 1.04)
a.set_xticks(np.arange(0, 1.01, 0.2))
a.set_xlabel("mean mis-selection rate / $\\varepsilon$ (CRC, mean of 20 splits)")
a.set_title("(a) CRC: 210 settings, all $\\leq 1$", fontsize=7.5)
a.grid(axis="y", visible=False)
ok, bad = g[g.z >= -3], g[g.z < -3]
b.plot([0, 1], [0, 1], color="#898781", lw=0.8)
b.scatter(ok.claimed, ok.exact, s=10, color="#2a78d6", label=f"no significant shortfall ({len(ok)})", linewidths=0)
b.scatter(bad.claimed, bad.exact, s=14, color="#d03b3b", marker="x", label=f"shortfall, $z<-3$ ({len(bad)})", linewidths=1)
b.set_xlim(0.45, 1.01)
b.set_ylim(0.2, 1.01)
b.set_xlabel("claimed confidence")
b.set_ylabel("observed exact-match rate")
b.set_title("(b) Everest-style rule (our re-implementation)", fontsize=7.5)
b.legend(frameon=False, fontsize=6.5, loc="lower right")
fig.tight_layout()
fig.savefig(FIG / "fig_guarantee.pdf")
plt.close(fig)

# ---- Fig. 2: same-call comparison, k = 10%, 7 domains + LC-LLM
par = pd.concat([pd.read_csv(RES / "selective_topk_pareto_curves.csv"),
                 pd.read_csv(RES / "selective_topk_p3extra_curves.csv"),
                 pd.read_csv(RES / "selective_topk_p3c_curves.csv")], ignore_index=True)
PSTYLE = {"P3c": ("#4b1f7a", "-", 1.6, "P3c (ours)"), "P3": ("#a582c9", (0, (4, 1.5)), 1.1, "P3 (ours)"), **STYLE}
doms = SEVEN + ["lc_llm"]
fig, axes = plt.subplots(2, 4, figsize=(7.0, 3.1))
for ax, d in zip(axes.flat, doms):
    c = par[(par.domain == d) & (par.k_fraction == 0.10)].groupby(["method", "call_fraction"]).miss.mean().reset_index()
    for m, (col, ls, lw, _) in PSTYLE.items():
        s = c[c.method == m]
        ax.plot(100 * s.call_fraction, s.miss, color=col, ls=ls, lw=lw)
    ax.set_title(NAME[d], fontsize=7.5)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, None)
handles = [plt.Line2D([], [], color=c, ls=ls, lw=lw) for c, ls, lw, _ in PSTYLE.values()]
fig.legend(handles, [v[3] for v in PSTYLE.values()], loc="upper center", ncol=8, frameon=False, fontsize=7)
fig.supxlabel(r"expensive calls (\% of candidates)", fontsize=8)
fig.supylabel("mis-selection rate", fontsize=7.5)
fig.tight_layout(rect=(0, 0, 1, 0.93))
fig.savefig(FIG / "fig_pareto.pdf")
plt.close(fig)

# ---- Fig. 3: cumulative expensive calls with calibration cost (k = 10%, eps = 0.10)
ms = pd.read_csv(RES / "selective_topk_m_sensitivity_summary.csv")
ms = ms[(ms.priority == "P2") & (ms.k_fraction == 0.10) & (ms.eps == 0.10)]
evs = pd.read_csv(RES / "selective_topk_everest_summary.csv")
m_cal = crc.groupby("domain").m_cal.mean()
fig, axes = plt.subplots(1, 4, figsize=(7.0, 1.75))
for ax, d in zip(axes, ["soccer", "home_credit", "ieee_fraud", "lending_club"]):
    r = ms[ms.domain == d].set_index("m_label").calls_mean
    f_ev = float(evs[(evs.domain == d) & (evs.k_fraction == 0.10) & (evs.thres == 0.9)].calls_mean.iloc[0])
    n = np.arange(0, 301)
    ax.plot(n, n, color="#898781", lw=0.8, ls=(0, (1, 2)))
    ax.plot(n, m_cal[d] + n * r["all"], color="#2a78d6", lw=1.4)
    ax.plot(n, 30 + n * r["30"], color="#2a78d6", lw=1.1, ls="--")
    ax.plot(n, n * f_ev, color="#eb6834", lw=1.1, ls="-.")
    ax.set_title(NAME[d], fontsize=7.5)
    ax.set_xlim(0, 300)
    ax.set_ylim(0, 300)
labels = ["all expensive", r"P2+CRC (all $m$ cal.\ queries)", r"P2+CRC (30 cal.\ queries)",
          "Everest 0.9 (no calibration)"]
handles = [plt.Line2D([], [], color=c, ls=ls, lw=lw) for c, ls, lw in
           [("#898781", (0, (1, 2)), 0.8), ("#2a78d6", "-", 1.4), ("#2a78d6", "--", 1.1), ("#eb6834", "-.", 1.1)]]
fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, fontsize=6.5)
fig.supxlabel(r"queries processed $N$", fontsize=8)
fig.supylabel("cum. calls\n(query units)", fontsize=7)
fig.tight_layout(rect=(0, 0, 1, 0.86))
fig.savefig(FIG / "fig_cost.pdf")
plt.close(fig)
print("wrote", sorted(p.name for p in FIG.glob("*.pdf")))
