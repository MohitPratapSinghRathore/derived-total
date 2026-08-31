"""Stage 6: aggregate results/*.json -> tables + figures for the paper."""
import os, json, glob, re
import numpy as np

RES = os.path.join(os.path.dirname(__file__), "..", "results")
FIG = os.path.join(os.path.dirname(__file__), "..", "figures")

COND = [
    ("attn_8L256",   "Attention-only 8L/256"),
    ("attnpm_8L256", "Attention-only, param-matched"),
    ("alsb_l0_8L256", "ALSB ($\\lambda=0$)"),
    ("alsb_l1_8L256", "ALSB ($\\lambda=1$)"),
    ("attn_12L256",  "Attention-only 12L/256"),
    ("attn_8L384",   "Attention-only 8L/384"),
]


def load(cond):
    out = []
    for f in sorted(glob.glob(os.path.join(RES, f"{cond}_s*.json"))):
        if "_probes" in f or "_eplan" in f or "_patch" in f or "zablate" in f:
            continue
        out.append(json.load(open(f)))
    return out


def ms(vals):
    v = np.array(vals, dtype=float)
    return v.mean(0), (v.std(0, ddof=1) if len(v) > 1 else np.zeros_like(v.mean(0)))


def main():
    os.makedirs(FIG, exist_ok=True)
    rows, curves = [], {}
    for cond, label in COND:
        rs = load(cond)
        if not rs:
            continue
        bl = [r["best_layer"] for r in rs]
        fid = [np.array(r["fidelity_exact"])[r["best_layer"]] for r in rs]
        sq = [np.array(r["fidelity_sq"])[r["best_layer"]] for r in rs]
        ill = [r["illegal"] for r in rs]
        lm = [r["legal_mass"] for r in rs]
        h05 = [r["H05"] for r in rs]
        h20 = [r["H20"] for r in rs]
        fm, fs = ms(fid); im, isd = ms(ill); lmm, _ = ms(lm); sm, _ = ms(sq)
        buckets = rs[0]["buckets"]
        bi40 = [i for i, (lo, hi) in enumerate(buckets) if lo == 40]
        bi40 = bi40[0] if bi40 else 4
        val = [r["train_log"][-1]["val"] for r in rs]
        rows.append(dict(cond=cond, label=label, n_seed=len(rs),
                         params=rs[0]["params"], best_layer=float(np.mean(bl)),
                         val=float(np.mean(val)), val_sd=float(np.std(val, ddof=1)) if len(val) > 1 else 0.0,
                         H05=float(np.mean(h05)), H05_sd=float(np.std(h05, ddof=1)) if len(h05) > 1 else 0.0,
                         H20=float(np.mean(h20)),
                         fid40=float(fm[bi40]), fid40_sd=float(fs[bi40]),
                         sq40=float(sm[bi40]),
                         ill40=float(im[bi40]), ill40_sd=float(isd[bi40]),
                         legalmass40=float(lmm[bi40])))
        curves[label] = (buckets, fm, fs, im, sm)
        if "control_random_exact" in rs[0]:
            r0 = rs[0]
            curves[label + " [random-weight ctrl]"] = (
                buckets, np.array(r0["control_random_exact"])[r0["best_layer"]],
                None, None, np.array(r0["control_random_sq"])[r0["best_layer"]])

    json.dump(rows, open(os.path.join(RES, "summary.json"), "w"), indent=1)

    # ---- console table
    print(f"{'condition':34s} {'params':>9s} {'val':>6s} {'H05':>5s} {'F@40':>7s} "
          f"{'sqacc@40':>9s} {'illegal@40':>11s} {'legalmass@40':>12s}")
    for r in rows:
        print(f"{r['label'][:34]:34s} {r['params']:9d} {r['val']:6.3f} {r['H05']:5.0f} "
              f"{r['fid40']:7.4f} {r['sq40']:9.4f} {r['ill40']:11.4f} {r['legalmass40']:12.4f}")

    # ---- figures
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        mids = None
        fig, axes = plt.subplots(1, 3, figsize=(16, 4.4))
        for label, (buckets, fm, fs, im, sm) in curves.items():
            mids = [(lo + hi) / 2 for lo, hi in buckets]
            ls = "--" if "ctrl" in label else "-"
            axes[0].plot(mids, fm, ls, marker="o", ms=3, label=label)
            axes[1].plot(mids, sm, ls, marker="o", ms=3, label=label)
            if im is not None:
                axes[2].plot(mids, im, ls, marker="o", ms=3, label=label)
        axes[0].set_ylabel("exact-position fidelity $F(t)$")
        axes[1].set_ylabel("per-square state accuracy")
        axes[2].set_ylabel("illegal top-1 move rate  ($E_{state}$)")
        for ax in axes:
            ax.set_xlabel("ply $t$"); ax.grid(alpha=.3)
        axes[0].legend(fontsize=6)
        plt.tight_layout()
        plt.savefig(os.path.join(FIG, "fidelity_curves.png"), dpi=160)
        print("wrote figures/fidelity_curves.png")

        # per-layer heatmap for the base condition
        rs = load("attn_8L256")
        if rs:
            f = np.array(rs[0]["fidelity_sq"])
            plt.figure(figsize=(7, 3.6))
            plt.imshow(f, aspect="auto", origin="lower", cmap="viridis")
            plt.colorbar(label="per-square state accuracy")
            plt.yticks(range(f.shape[0]), [f"L{i}" for i in range(f.shape[0])], fontsize=6)
            plt.xticks(range(len(rs[0]["buckets"])),
                       [f"{lo}-{hi}" for lo, hi in rs[0]["buckets"]], rotation=45, fontsize=6)
            plt.xlabel("ply bucket"); plt.title("state decodability by layer and depth")
            plt.tight_layout()
            plt.savefig(os.path.join(FIG, "layer_heatmap.png"), dpi=160)
            print("wrote figures/layer_heatmap.png")
    except ImportError:
        print("matplotlib missing; skipped figures")


if __name__ == "__main__":
    main()
