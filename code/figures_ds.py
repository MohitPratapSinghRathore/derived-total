"""Figure 1: absolute failure rate against parameter count, both ladders.

Log-log, one y-scale per panel, so a straight line is a power law and its slope
is the exponent reported in the table. Points are individual runs; the line is
the same full-data fit ds_stats reports, so the figure cannot disagree with the
text.

Colour follows the concept, not the panel: the local failure class is blue and
the global one orange in both ladders. Identity is never colour alone -- every
series also has its own marker shape and a direct label, because two of the
palette's slots fall below 3:1 contrast against the surface.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import ds_stats as S

OUT = os.path.join(S.ROOT, "paper_scaling", "figures")
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, MUTED, GRID = "#1a1a19", "#6b6a63", "#e6e5e0"


def style(ax):
    ax.set_xscale("log")
    ax.set_yscale("log")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(which="both", colors=MUTED, labelsize=8)
    ax.grid(True, which="major", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def series(ax, xs, ys, fit_x, color, marker, label, dashed=False, dy=0):
    ax.scatter(xs, ys, s=22, color=color, marker=marker, edgecolors="white",
               linewidths=0.8, zorder=3, label=label)
    lx = np.log(fit_x)
    b, a = np.polyfit(np.log(xs), np.log(ys), 1)
    gx = np.linspace(lx.min(), lx.max(), 50)
    ax.plot(np.exp(gx), np.exp(a + b * gx), color=color, linewidth=2,
            linestyle="--" if dashed else "-", zorder=2)
    ax.annotate(f"{label}  ({b:+.2f})", xy=(np.exp(gx[-1]), np.exp(a + b * gx[-1])),
                xytext=(6, dy), textcoords="offset points", va="center",
                fontsize=7.5, color=INK)


def main():
    os.makedirs(OUT, exist_ok=True)
    rows = S.ladder_rows()
    ext = S.external_rows()
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.9))

    ax = axes[0]
    style(ax)
    p = np.array([q["params"] for q in rows], float)
    for key, color, marker, label, dashed in (
            ("from_empty", BLUE, "o", "empty source (local)", False),
            ("geometry", AQUA, "s", "geometry", False),
            ("leaves_check", ORANGE, "^", "leaves check (global)", False),
            ("correct_local_belief", YELLOW, "D",
             "despite correct local belief", True)):
        series(ax, p, np.array([q[key] for q in rows]), p, color, marker, label,
               dashed)
    ax.set_xlabel("parameters", fontsize=9, color=INK)
    ax.set_ylabel("failures per scored position", fontsize=9, color=INK)
    ax.set_title("(a) Our ladder: 6 sizes x 3 seeds, UCI", fontsize=9.5,
                 color=INK, loc="left")
    ax.set_xlim(p.min() * 0.8, p.max() * 4.5)

    ax = axes[1]
    style(ax)
    pe = np.array([r["params"] for r in ext], float)
    for key, color, marker, label, dy in (
            ("unreachable", BLUE, "o", "unreachable (local)", 9),
            ("leaves_check", ORANGE, "^", "leaves check (global)", -9)):
        series(ax, pe, np.array([r["absolute"][key] for r in ext]), pe, color,
               marker, label, dy=dy)
    ax.set_xlabel("parameters", fontsize=9, color=INK)
    ax.set_title("(b) Independent ladder: 3 checkpoints, character SAN",
                 fontsize=9.5, color=INK, loc="left")
    ax.set_xlim(pe.min() * 0.7, pe.max() * 6)

    for a in axes:
        a.legend(fontsize=7, frameon=False, loc="lower left")
    fig.tight_layout()
    path = os.path.join(OUT, "scaling.png")
    fig.savefig(path, dpi=220, facecolor="white")
    print("wrote", path)


if __name__ == "__main__":
    main()
