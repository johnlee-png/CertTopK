"""Explanatory (concept) figures for the paper; illustrative toy numbers, not results.
Usage: py -X utf8 code/make_concept_figs.py   (from the artifact root)
Writes figures/fig_overview.pdf, figures/fig_budget.pdf, figures/fig_envelope.pdf
Text is typeset by LaTeX in Linux Libertine (libertine + newtxmath), the font of the acmart body.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402
import numpy as np  # noqa: E402

FIG = Path(__file__).resolve().parent.parent / "figures"
FIG.mkdir(exist_ok=True)
plt.rcParams.update({"text.usetex": True, "font.family": "serif", "font.size": 8,
                     "text.latex.preamble": r"\usepackage[tt=false]{libertine}\usepackage[libertine]{newtxmath}",
                     "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
                     "xtick.major.size": 2.5, "ytick.major.size": 2.5,
                     "axes.spines.top": False, "axes.spines.right": False})
INK, MUTED, LIGHT = "#222222", "#6b6b6b", "#d9d9d9"
BLUE, ORANGE, RED, GREEN = "#1f5fa8", "#c8641e", "#b22222", "#2e8b57"


def ml(*lines):
    """Centered multi-line LaTeX label."""
    return r"\begin{tabular}{@{}c@{}}" + r"\\".join(lines) + r"\end{tabular}"


def box(ax, x, y, w, h, text, fc="white", lw=0.6):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.012", fc=fc, ec=INK, lw=lw))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=7.5, color=INK)


def arrow(ax, p, q, color=INK, ls="-", lw=0.7):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=7, lw=lw, color=color, linestyle=ls,
                                 shrinkA=0, shrinkB=0))


# ---------------------------------------------------------------- Fig. overview: offline calibration + online query
fig = plt.figure(figsize=(6.6, 1.9))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(-0.005, 1.005)
ax.set_ylim(0, 1)
ax.axis("off")
H1, H2, Y1, Y2 = 0.27, 0.30, 0.60, 0.08
ax.text(0.0, 0.93, r"\textbf{(a) Offline, once}", fontsize=8, color=INK)
xs1 = [(0.0, 0.165), (0.2, 0.165), (0.4, 0.165), (0.6, 0.225)]
labels1 = [ml(r"$m$ calibration queries", r"(all candidates refined)"),
           ml(r"loss curves", r"$L_{Q_t}(\lambda)$, $t = 1..m$"),
           ml(r"monotone envelopes", r"$\tilde L_{Q_t}(\lambda)$ (Sec.~III-D)"),
           ml(r"CRC: smallest $\hat\lambda$ with corrected", r"mean $\leq \varepsilon$; LTT: test downward",
              r"from $\lambda_G$ at level $\delta$")]
for (x, w), t in zip(xs1, labels1):
    box(ax, x, Y1, w, H1, t)
for (x, w), (x2, _) in zip(xs1[:-1], xs1[1:]):
    arrow(ax, (x + w, Y1 + H1 / 2), (x2, Y1 + H1 / 2))
ax.text(0.84, Y1 + H1 / 2, ml(r"one $\hat\lambda$ per", r"$(k, \varepsilon, \text{order})$, all from", r"the same curves"),
        fontsize=7, va="center", ha="left", color=MUTED)
ax.text(0.0, 0.43, r"\textbf{(b) Online, every query}", fontsize=8, color=INK)
xs2 = [(0.0, 0.12), (0.155, 0.205), (0.395, 0.185), (0.615, 0.18), (0.83, 0.17)]
labels2 = [ml(r"query $Q$", r"($n_Q$ candidates)"),
           ml(r"cheap model for all:", r"$U^c_i = P^c_i R_i - (1-P^c_i) C_i$"),
           ml(r"priority $\pi_i$ of the order", r"chosen on $D_0$ (e.g., P2, P3c)"),
           ml(r"refine $\{i : \pi_i \leq \hat\lambda\}$", r"with the expensive model"),
           ml(r"top-$k$ of mixed scores,", r"$\mathbb{E}[\text{mis-selection}] \leq \varepsilon$", r"(LTT: w.p.\ $1-\delta$)")]
for i, ((x, w), t) in enumerate(zip(xs2, labels2)):
    box(ax, x, Y2, w, H2, t, fc="#eeeeee" if i == len(xs2) - 1 else "white", lw=0.9 if i == len(xs2) - 1 else 0.6)
for (x, w), (x2, _) in zip(xs2[:-1], xs2[1:]):
    arrow(ax, (x + w, Y2 + H2 / 2), (x2, Y2 + H2 / 2))
arrow(ax, (0.73, Y1), (0.705, Y2 + H2), color=BLUE, ls=(0, (3, 2)), lw=0.8)
ax.text(0.73, (Y1 + Y2 + H2) / 2, r"$\hat\lambda$", color=BLUE, fontsize=8.5, va="center")
fig.savefig(FIG / "fig_overview.pdf")
plt.close(fig)

# ---------------------------------------------------------------- Fig. budget: toy query, why R + C matters
# Eight loans with cheap utility; cut-off for k = 3 between the 3rd and 4th. Two candidates at the same distance
# below the cut-off: A with a small budget, B with a large one. Bars show how far each can move for the same
# probability error |Pe - Pc| = 0.05 (U moves by (R + C) * 0.05).
uc = np.array([9.0, 7.5, 6.2, 4.8, 4.8, 3.0, 2.0, 1.0])
budget = np.array([20, 30, 25, 20, 80, 30, 25, 20])
names = ["", "", "", "A", "B", "", "", ""]
k = 3
theta = 0.5 * (np.sort(uc)[::-1][k - 1] + np.sort(uc)[::-1][k])
fig, a = plt.subplots(figsize=(3.4, 1.85))
x = np.arange(len(uc))
need = [abs(uc[3] - theta) / budget[3], abs(uc[4] - theta) / budget[4]]
a.bar(x, uc, color=["#7a9cc6" if i < k else LIGHT for i in range(len(uc))], width=0.6, edgecolor=INK, linewidth=0.4)
for i in (3, 4):
    up = budget[i] * 0.05
    col = ORANGE if i == 4 else BLUE
    a.errorbar(x[i], uc[i], yerr=[[0], [up]], fmt="none", ecolor=col, elinewidth=1.6, capsize=2.5, capthick=1.0)
    a.text(x[i] + (0.35 if i == 4 else 0.0), uc[i] + up + (-0.6 if i == 4 else 0.35),
           ml(f"{names[i]}: $R{{+}}C = {budget[i]}$", f"$\\pi = {need[i - 3]:.3f}$"),
           ha="left" if i == 4 else "center", va="center" if i == 4 else "bottom", fontsize=6.8, color=col)
a.axhline(theta, color=RED, lw=0.8, ls=(0, (4, 2)))
a.text(7.45, theta + 0.25, r"cheap cut-off $\theta_c$ ($k = 3$)", color=RED, fontsize=6.8, ha="right")
a.set_xticks(x, [f"$c_{i + 1}$" for i in x])
a.set_ylabel(r"cheap utility $U^c$")
a.set_ylim(0, 10.8)
fig.tight_layout(pad=0.3)
fig.savefig(FIG / "fig_budget.pdf")
plt.close(fig)

# ---------------------------------------------------------------- Fig. envelope: non-monotone loss of one query
lam = np.linspace(0, 1, 11)
L = np.array([0.40, 0.30, 0.30, 0.20, 0.30, 0.10, 0.20, 0.10, 0.0, 0.10, 0.0])
env = np.maximum.accumulate(L[::-1])[::-1]
fig, ax = plt.subplots(figsize=(3.4, 1.75))
ax.fill_between(lam, L, env, step="post", color="#c9d9ee", lw=0, label=r"gap $\tilde L_Q - L_Q$")
ax.step(lam, L, where="post", color=MUTED, lw=0.9, ls=(0, (3, 1.5)), label=r"loss $L_Q(\lambda)$ of one query")
ax.step(lam, env, where="post", color=BLUE, lw=1.4,
        label=r"envelope $\tilde L_Q(\lambda) = \max_{\lambda' \geq \lambda} L_Q(\lambda')$")
for j in (3, 5, 8):
    ax.plot(lam[j + 1], L[j + 1], "o", ms=3, mfc="white", mec=RED, mew=0.8, zorder=5)
ax.annotate(r"refining more raises the loss", xy=(lam[4] + 0.01, 0.30), xytext=(0.42, 0.43), fontsize=7, color=RED,
            arrowprops=dict(arrowstyle="-", color=RED, lw=0.6))
ax.set_xlabel(r"refinement level $\lambda$ (more candidates refined $\rightarrow$)")
ax.set_ylabel(r"mis-selection rate")
ax.set_xlim(0, 1.02)
ax.set_ylim(0, 0.48)
ax.set_yticks([0, 0.1, 0.2, 0.3, 0.4])
ax.legend(frameon=False, fontsize=6.8, loc="lower left", handlelength=1.8, borderaxespad=0.2)
fig.tight_layout(pad=0.3)
fig.savefig(FIG / "fig_envelope.pdf")
plt.close(fig)
print("wrote concept figures")
