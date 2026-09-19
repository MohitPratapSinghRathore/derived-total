"""Figure 1: the fitted aggregate against the derived total.

The paper's claim is that these two objects are different and that only one of
them can be reported. The figure says it directly rather than through numbers.

Panel A is the total error rate on log-log axes. A power law is a straight line
there, so the fitted aggregate is straight by construction. The derived total, the
sum of the same category curves, is not: it bends, because a sum of power laws with
differing exponents has a slope that moves. Both pass through the observed points,
which is the point. Nothing in range distinguishes them, and the reader can see the
gap opening only where the forecast was actually read.

Panel B is the same fact as a slope. The fitted aggregate contributes a horizontal
line, since a power law has one exponent everywhere. The derived total's slope
drifts across a substantial part of the band spanned by the category exponents,
which are drawn as reference ticks so the drift can be compared against the very
quantities the aggregate was meant to summarise.

Colour is never the only cue: the fitted aggregate is dashed everywhere and the
derived total solid, both panels carry direct labels, and the shaded region marks
extrapolation in both, so the figure survives greyscale printing and the two
palette slots that fall below 3:1 contrast against the surface.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import ds_stats as S
import admissibility as A

OUT = os.path.join(S.ROOT, "paper_dsr", "figures")
BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, MUTED, GRID, SHADE = "#1a1a19", "#6b6a63", "#e6e5e0", "#f2f1ec"


def style(ax, logy=True):
    ax.set_xscale("log")
    if logy:
        ax.set_yscale("log")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(which="both", colors=MUTED, labelsize=8)
    ax.grid(True, which="major", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def main():
    os.makedirs(OUT, exist_ok=True)
    rows = S.ladder_rows()
    fits = A.parts_fits(rows)
    tot = A.powerlaw(rows, "illegal_rate")

    n_lo = min(q["params"] for q in rows)
    n_hi = max(q["params"] for q in rows)
    grid = np.logspace(np.log10(n_lo), 12, 400)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.2, 3.5))

    # ---------------------------------------------------------- panel A
    style(ax1)
    ax1.axvspan(n_hi, grid[-1], color=SHADE, zorder=0)
    ax1.scatter([q["params"] for q in rows],
                [q["illegal_rate"] for q in rows],
                s=20, color=INK, marker="o", edgecolors="white",
                linewidths=0.7, zorder=4, label="observed runs")
    ax1.plot(grid, [np.exp(np.polyval(tot, np.log(n))) for n in grid],
             color=ORANGE, linestyle="--", linewidth=1.7, zorder=3,
             label="aggregate, fitted as a power law")
    ax1.plot(grid, [A.T_sum(fits, n) for n in grid],
             color=BLUE, linestyle="-", linewidth=1.7, zorder=3,
             label="total, derived from the categories")
    ax1.set_xlabel("parameters", fontsize=9, color=INK)
    ax1.set_ylabel("total error rate", fontsize=9, color=INK)
    ax1.set_title("A. One of these is not a power law", fontsize=9.5,
                  color=INK, loc="left")
    ax1.text(n_hi * 1.5, 0.22, "extrapolation", fontsize=7.5, color=MUTED)
    ax1.legend(fontsize=7.2, frameon=False, loc="lower left")

    # ---------------------------------------------------------- panel B
    style(ax2, logy=False)
    x_left = n_lo / 3.0
    ax2.set_xlim(x_left, grid[-1])
    ax2.axvspan(n_hi, grid[-1], color=SHADE, zorder=0)
    exps = sorted(mc[0] for mc in fits.values())
    # The category exponents as a band with rules inside the axes, so nothing is
    # clipped at the spine and the drift can be read against them directly.
    ax2.axhspan(exps[0], exps[-1], xmin=0, xmax=1, color=GRID, alpha=0.5,
                zorder=1)
    for e in exps:
        ax2.plot([x_left * 1.15, n_lo * 0.75], [e, e], color=MUTED,
                 linewidth=1.1, zorder=2)
    ax2.annotate("span of the\ncategory exponents",
                 xy=(x_left * 1.5, (exps[0] + exps[-1]) / 2),
                 xytext=(n_lo * 2.2, (exps[0] + exps[-1]) / 2 - 0.07),
                 fontsize=7.2, color=MUTED, va="center",
                 arrowprops=dict(arrowstyle="-", color=MUTED, linewidth=0.7))
    ax2.axhline(tot[0], color=ORANGE, linestyle="--", linewidth=1.7, zorder=3,
                label="fitted aggregate exponent")
    ax2.plot(grid, [A.effective_exponent(fits, n) for n in grid],
             color=BLUE, linewidth=1.7, zorder=3,
             label="slope of the derived total")
    ax2.set_xlabel("parameters", fontsize=9, color=INK)
    ax2.set_ylabel(r"$\mathrm{d}\log T\,/\,\mathrm{d}\log N$", fontsize=9,
                   color=INK)
    ax2.set_title("B. The aggregate exponent is a local slope", fontsize=9.5,
                  color=INK, loc="left")
    ax2.legend(fontsize=7.2, frameon=False, loc="lower right")

    fig.tight_layout()
    p = os.path.join(OUT, "derived_vs_fitted.pdf")
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print("wrote", p)


if __name__ == "__main__":
    main()
