"""Generate every figure in the paper from results/*.json."""
import os, json, glob
import numpy as np
import matplotlib
from horizon import interp_horizon
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RES = os.path.join(os.path.dirname(__file__), "..", "results")
FIG = os.path.join(os.path.dirname(__file__), "..", "paper", "figures")
os.makedirs(FIG, exist_ok=True)

plt.rcParams.update({
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
    "legend.fontsize": 6.5, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "axes.grid": True, "grid.alpha": 0.25, "figure.dpi": 200,
    "axes.spines.top": False, "axes.spines.right": False,
})


def load(n):
    p = os.path.join(RES, f"{n}.json")
    return json.load(open(p)) if os.path.exists(p) else None


def mids(buckets):
    return [(lo + hi) / 2 for lo, hi in buckets]


def fig_decay():
    r = load("attn_8L256_s0")
    if not r:
        return
    b = r["buckets"]; x = mids(b); bl = r["best_layer"]
    occ = np.array(r["fidelity_occ"])[bl]
    fig, ax = plt.subplots(1, 2, figsize=(6.8, 2.5))
    ax[0].plot(x, occ, "o-", ms=3, color="#1f4e79", label="trained model")
    if "control_random_occ" in r:
        ax[0].plot(x, np.array(r["control_random_occ"])[bl], "s--", ms=2.5,
                   color="#888", label="randomised weights")
        ax[0].plot(x, np.array(r["control_shuffled_occ"])[bl], "^--", ms=2.5,
                   color="#bbb", label="label permutation")
    ax[0].plot(x, r["majority_occ"], ":", color="#c00", label="majority predictor")
    ax[0].set_xlabel("ply"); ax[0].set_ylabel("occupied-square state accuracy")
    ax[0].set_title("(a) internal state decays with depth")
    ax[0].legend(frameon=False)

    ax[1].plot(x, r["illegal"], "o-", ms=3, color="#8b0000", label="illegal top-1 rate")
    ax[1].plot(x, r["legal_mass"], "s-", ms=3, color="#1f4e79",
               label="probability mass on legal moves")
    ax[1].set_xlabel("ply"); ax[1].set_ylabel("rate")
    ax[1].set_title("(b) behaviour degrades in step")
    ax[1].legend(frameon=False)
    plt.tight_layout(); plt.savefig(os.path.join(FIG, "decay.png")); plt.close()
    print("figures/decay.png")


def fig_layers():
    r = load("attn_8L256_s0")
    if not r:
        return
    f = np.array(r["fidelity_occ"])
    plt.figure(figsize=(4.2, 2.4))
    plt.imshow(f, aspect="auto", origin="lower", cmap="viridis")
    plt.colorbar(label="occupied-square accuracy")
    plt.yticks(range(f.shape[0]), [f"L{i}" for i in range(f.shape[0])])
    plt.xticks(range(len(r["buckets"])), [f"{lo}" for lo, _ in r["buckets"]])
    plt.xlabel("ply"); plt.ylabel("layer")
    plt.title("state decodability rises through the stack")
    plt.tight_layout(); plt.savefig(os.path.join(FIG, "layers.png")); plt.close()
    print("figures/layers.png")


def fig_coupling():
    c = load("attn_8L256_s0_coupling")
    if not c:
        return
    fig, ax = plt.subplots(1, 2, figsize=(6.8, 2.5))
    corr = [v if v is not None else np.nan
            for v in c["within_bucket_corr_wrongsquares_illegal"]]
    ax[0].bar(range(len(corr)), corr, color="#999")
    ax[0].axhline(0, color="k", lw=0.6)
    ax[0].set_ylim(-0.05, 1.0)
    ax[0].set_xticks(range(len(c["buckets"])))
    ax[0].set_xticklabels([f"{lo}" for lo, _ in c["buckets"]])
    ax[0].set_xlabel("ply"); ax[0].set_ylabel("correlation")
    ax[0].set_title("(a) aggregate state error vs illegality\n(within depth)")

    L = c["locality"]
    it = L["illegal_touched_wrong"] + L["illegal_touched_ok"]
    lt = L["legal_touched_wrong"] + L["legal_touched_ok"]
    vals = [L["illegal_touched_wrong"] / max(it, 1), L["legal_touched_wrong"] / max(lt, 1)]
    ax[1].bar(["illegal moves", "legal moves"], vals, color=["#8b0000", "#1f4e79"], width=.55)
    for i, v in enumerate(vals):
        ax[1].text(i, v + .02, f"{v:.3f}", ha="center", fontsize=7)
    ax[1].set_ylim(0, 1)
    ax[1].set_ylabel("P(state wrong on the squares used)")
    ax[1].set_title("(b) action-relevant state error")
    plt.tight_layout(); plt.savefig(os.path.join(FIG, "coupling.png")); plt.close()
    print("figures/coupling.png")


def fig_belief():
    b = load("attn_8L256_s0_belief")
    if not b:
        return
    x = mids(b["buckets"])
    # suppress buckets with too few illegal cases to estimate a rate
    MINN = 30
    keep = [i for i, n in enumerate(b["n_illegal"]) if n >= MINN]
    xk = [x[i] for i in keep]
    def k(v):
        return [v[i] for i in keep]
    fig, ax = plt.subplots(1, 2, figsize=(6.8, 2.5))
    ax[0].plot(xk, k(b["illegal_legal_in_belief"]), "o-", ms=3, color="#8b0000",
               label="model's illegal moves")
    ax[0].plot(xk, k(b["legal_legal_in_belief"]), "s-", ms=3, color="#1f4e79",
               label="model's legal moves (ceiling)")
    ax[0].plot(xk, k(b["mismatched_belief_legal"]), "^--", ms=3, color="#888",
               label="same move, other belief")
    ax[0].plot(xk, k(b["randomillegal_legal_in_belief"]), ":", color="#000",
               label="random illegal move")
    ax[0].set_xlabel("ply"); ax[0].set_ylabel("legal in the believed board")
    ax[0].set_title("(a) belief consistency against depth")
    ax[0].legend(frameon=False)

    o = b["overall"]
    keys = ["illegal_legal_in_belief", "mismatched_belief_legal",
            "randomillegal_legal_in_belief"]
    labs = ["own belief", "mismatched\nbelief", "random\nmove"]
    vals = [o[k] for k in keys]
    ax[1].bar(labs, vals, color=["#8b0000", "#888", "#ccc"], width=.55)
    ax[1].axhline(o["legal_legal_in_belief"], color="#1f4e79", ls="--", lw=1)
    ax[1].text(1.0, o["legal_legal_in_belief"] + .012, "decoding ceiling",
               ha="center", fontsize=6, color="#1f4e79")
    ax[1].set_ylim(0, o["legal_legal_in_belief"] + .09)
    for i, v in enumerate(vals):
        ax[1].text(i, v + .015, f"{v:.3f}", ha="center", fontsize=7)
    ax[1].set_ylabel("illegal move legal in that board")
    ax[1].set_title("(b) the errors are coherent")
    plt.tight_layout(); plt.savefig(os.path.join(FIG, "belief.png")); plt.close()
    print("figures/belief.png")


def fig_patch():
    sw = os.path.join(RES, "attn_8L256_s0_patch_sweep.json")
    if not os.path.exists(sw):
        return
    d = json.load(open(sw))
    a = sorted(d, key=float)
    x = [float(v) for v in a]
    plt.figure(figsize=(4.6, 2.6))
    plt.plot(x, [d[k]["flip_rate"] for k in a], "o-", ms=3, label="belief flipped (check)")
    plt.plot(x, [1 - d[k]["state_mass_after"] / d[k]["state_mass_before"] for k in a],
             "s-", ms=3, color="#8b0000", label="mass removed, state edit")
    plt.plot(x, [1 - d[k]["random_mass_after"] / d[k]["state_mass_before"] for k in a],
             "^--", ms=3, color="#888", label="mass removed, random edit")
    plt.plot(x, [d[k]["random_frac_decreased"] for k in a], ":", color="#000",
             label="random edit: fraction decreased")
    plt.axvspan(0.25, 1.0, color="#cceeff", alpha=.45)
    plt.text(0.42, 0.05, "usable window", fontsize=6, color="#046")
    plt.xscale("log"); plt.xlabel("edit strength (residual-stream magnitudes)")
    plt.ylabel("effect"); plt.title("calibrating the causal intervention")
    plt.legend(frameon=False, loc="center right")
    plt.tight_layout(); plt.savefig(os.path.join(FIG, "patch_calibration.png")); plt.close()
    print("figures/patch_calibration.png")


def fig_scale():
    rungs = []
    for f in sorted(glob.glob(os.path.join(RES, "attn_*_s0.json"))):
        n = os.path.basename(f)[:-5]
        if any(t in n for t in ("coupling", "belief", "patch", "eplan")):
            continue
        r = load(n)
        if not r or "H_ill_05" not in r:
            continue
        bel = load(f"{n}_belief")
        rungs.append((r["params"], interp_horizon(r["illegal"], r["buckets"], 0.05),
                      bel["overall"]["illegal_legal_in_belief"] if bel else None,
                      bel["overall"]["mismatched_belief_legal"] if bel else None,
                      bel["overall"]["legal_legal_in_belief"] if bel else None))
    if len(rungs) < 2:
        return
    rungs.sort()
    p = [r[0] / 1e6 for r in rungs]
    fig, ax = plt.subplots(1, 2, figsize=(6.8, 2.5))
    ax[0].plot(p, [r[1] for r in rungs], "o-", ms=4, color="#1f4e79")
    ax[0].set_xscale("log"); ax[0].set_xlabel("parameters (M)")
    ax[0].set_ylabel("horizon $H_{ill}(0.05)$ (ply)")
    ax[0].set_title("(a) horizon against scale")
    idx = [i for i, r in enumerate(rungs) if r[2] is not None]
    if idx:
        ax[1].plot([p[i] for i in idx], [rungs[i][2] for i in idx], "o-", ms=4,
                   color="#8b0000", label="own belief")
        ax[1].plot([p[i] for i in idx], [rungs[i][3] for i in idx], "^--", ms=4,
                   color="#888", label="mismatched control")
        # normalised by each model's own decoding ceiling, since the ceiling
        # itself rises with scale and would otherwise inflate the trend
        ceil = [rungs[i][4] for i in idx]
        ax[1].plot([p[i] for i in idx],
                   [rungs[i][2] / c for i, c in zip(idx, ceil)], "s-", ms=4,
                   color="#1f4e79", label="own belief / ceiling")
        ax[1].set_xscale("log"); ax[1].set_xlabel("parameters (M)")
        ax[1].set_ylabel("illegal move legal in belief")
        ax[1].set_title("(b) coherence of errors against scale")
        ax[1].legend(frameon=False)
    plt.tight_layout(); plt.savefig(os.path.join(FIG, "scale.png")); plt.close()
    print("figures/scale.png")


if __name__ == "__main__":
    for f in (fig_decay, fig_layers, fig_coupling, fig_belief, fig_patch, fig_scale):
        try:
            f()
        except Exception as e:
            print(f"{f.__name__} skipped: {e}")
