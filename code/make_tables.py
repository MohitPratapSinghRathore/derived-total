"""Emit LaTeX tables directly from results/*.json.

This is the only path from measurement to manuscript. No table cell is typed by
hand, and an absent value prints n/a rather than a guess.
"""
import os, json, glob
import numpy as np
from horizon import interp_horizon

RES = os.path.join(os.path.dirname(__file__), "..", "results")
OUT = os.path.join(os.path.dirname(__file__), "..", "paper", "tables.tex")

CORE = [("attn_8L256", "Attention-only"),
        ("attnpm_8L256", "Attention-only, parameter-matched"),
        ("alsb_l0_8L256", "Recurrent channel ($\\lambda{=}0$)"),
        ("alsb_l1_8L256", "Recurrent channel ($\\lambda{=}1$)")]
LADDER = [("attn_6L192", "6L/192"), ("attn_8L256", "8L/256"),
          ("attn_12L256", "12L/256"), ("attn_8L384", "8L/384"),
          ("attn_12L384", "12L/384"), ("attn_12L512", "12L/512")]


def load(n):
    p = os.path.join(RES, f"{n}.json")
    return json.load(open(p)) if os.path.exists(p) else None


def seeds(cond):
    return [r for r in (load(f"{cond}_s{s}") for s in (0, 1, 2)) if r]


def pm(v, fmt="{:.3f}"):
    v = [x for x in v if x is not None]
    if not v:
        return "n/a"
    a = np.array(v, float)
    if len(a) == 1:
        return fmt.format(a[0])
    return (fmt + "\\,$\\pm$\\," + fmt).format(a.mean(), a.std(ddof=1))



def fnum(n):
    """Format an integer with thousands separators protected for LaTeX.

    Applying a comma replacement to a whole table row also corrupts the thin-space
    macros produced by pm(), which silently broke the build for a day.
    """
    return "{:,}".format(int(n)).replace(",", "{,}")


def controls_table():
    r = load("attn_8L256_s0")
    if not r or "control_random_occ" not in r:
        return []
    bl = r["best_layer"]
    occ = np.array(r["fidelity_occ"])[bl]
    cs = np.array(r["control_shuffled_occ"])[bl]
    cr = np.array(r["control_random_occ"])[bl]
    mj = np.array(r["majority_occ"])
    L = ["\\begin{table}[t]", "\\centering\\small",
         "\\caption{State decodability against every trivial explanation, for the",
         "baseline model. Occupied-square accuracy at the most decodable layer,",
         "beside a per-square per-depth majority predictor, a label-permutation",
         "control, and a randomised-weight model of identical architecture. The gap",
         "is what training put there.}",
         "\\label{tab:controls}", "\\begin{tabular}{lccccc}", "\\toprule",
         "Ply & model & majority & label perm. & random weights & illegal rate \\\\",
         "\\midrule"]
    for i, (lo, hi) in enumerate(r["buckets"]):
        L.append(f"{lo}--{hi} & {occ[i]:.3f} & {mj[i]:.3f} & {cs[i]:.3f} & "
                 f"{cr[i]:.3f} & {r['illegal'][i]:.4f} \\\\")
    L += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    return L


def mechanism_table():
    """The paper's central table: does state loss explain behaviour, per model."""
    rows = []
    for cond, label in [(c, l) for c, l in CORE] + \
                       [("attn_6L192", "6L/192"), ("attn_12L384", "12L/384"),
                        ("attn_12L512", "12L/512")]:
        c = load(f"{cond}_s0_coupling")
        b = load(f"{cond}_s0_belief")
        r = load(f"{cond}_s0")
        if not (c and b and r):
            continue
        Lc = c["locality"]
        it = Lc["illegal_touched_wrong"] + Lc["illegal_touched_ok"]
        lt = Lc["legal_touched_wrong"] + Lc["legal_touched_ok"]
        cors = [x for x in c["within_bucket_corr_wrongsquares_illegal"] if x is not None]
        o = b["overall"]
        rows.append((label, r["params"],
                     f"{min(cors):.2f} to {max(cors):.2f}" if cors else "n/a",
                     Lc["illegal_touched_wrong"] / max(it, 1),
                     Lc["legal_touched_wrong"] / max(lt, 1),
                     o["illegal_legal_in_belief"],
                     o.get("mismatched_belief_legal"),
                     o["randomillegal_legal_in_belief"],
                     o["legal_legal_in_belief"]))
    if not rows:
        return []
    L = ["\\begin{table}[t]", "\\centering\\small",
         "\\caption{Does state loss explain behaviour? Aggregate error is the",
         "within-depth correlation between misremembered squares and playing an",
         "illegal move. Action-relevant error is the probability the probe is wrong",
         "on the squares the move uses. Belief consistency is the share of illegal",
         "moves that are legal in the model's own decoded board, beside the same",
         "move under another position's belief, an arbitrary illegal move, and the",
         "decoding ceiling given by genuinely legal moves.}",
         "\\label{tab:mechanism}",
         "\\begin{tabular}{lrccccccc}", "\\toprule",
         "& & aggregate & \\multicolumn{2}{c}{action-relevant}"
         " & \\multicolumn{4}{c}{belief consistency} \\\\",
         "\\cmidrule(lr){4-5}\\cmidrule(lr){6-9}",
         "Model & Params & corr. & illegal & legal & own & mismatch & random & ceiling \\\\",
         "\\midrule"]
    for (lab, prm, cor, li, ll, bi, bm, br, bc) in rows:
        bm_s = "n/a" if bm is None else f"{bm:.3f}"
        L.append(f"{lab} & {fnum(prm)} & {cor} & {li:.3f} & {ll:.3f} & "
                 f"\\textbf{{{bi:.3f}}} & {bm_s} & {br:.3f} & {bc:.3f} \\\\")
    L += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    return L


def horizon_table():
    L = ["\\begin{table}[t]", "\\centering\\small",
         "\\caption{Horizon and state fidelity by condition. $H_{\\mathrm{ill}}(.05)$ is",
         "the ply at which the illegal-move rate first crosses five percent,",
         "interpolated between bucket midpoints because the bucketed read-off returns",
         "the same bucket for every condition and so discriminates nothing.",
         "Mean\\,$\\pm$\\,s.d.\\ over seeds where more than one was run.}",
         "\\label{tab:horizon}", "\\begin{tabular}{lrcccc}", "\\toprule",
         "Model & Params & seeds & occ.\\ acc.\\ @40--50 & illegal @40--50 "
         "& $H_{\\mathrm{ill}}(.05)$ \\\\", "\\midrule"]
    any_ = False
    for cond, label in CORE:
        rs = seeds(cond)
        if not rs:
            continue
        any_ = True
        L.append(f"{label} & {fnum(rs[0]['params'])} & {len(rs)} & "
                 f"{pm([np.array(r['fidelity_occ'])[r['best_layer']][4] for r in rs])} & "
                 f"{pm([r['illegal'][4] for r in rs])} & "
                 f"{pm([interp_horizon(r['illegal'], r['buckets'], 0.05) for r in rs], '{:.1f}')}"
                 f" \\\\")
    zs = [load(f"alsb_l1_8L256_s{s}_zablate") for s in (0, 1, 2)]
    zs = [z for z in zs if z]
    if zs:
        L.append("\\midrule")
        L.append(f"\\quad read-out ablated at test & n/a & {len(zs)} & "
                 f"{pm([np.array(z['fidelity_occ'])[z['best_layer']][4] for z in zs])} & "
                 f"{pm([z['illegal'][4] for z in zs])} & "
                 f"{pm([interp_horizon(z['illegal'], z['buckets'], 0.05) for z in zs], '{:.1f}')}"
                 f" \\\\")
    L += ["\\midrule"]
    from ladder_stats import rung_stats, fmt as lfmt
    for cond, label in LADDER:
        st = rung_stats(cond)
        if not st:
            continue
        any_ = True
        L.append(f"\\quad {label} & {fnum(st['params'])} & {st['n']} & "
                 f"{lfmt(st['occ40'], 3)} & {lfmt(st['ill40'], 3)} & "
                 f"{lfmt(st['h_ill'], 1)} \\\\")
    L += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    return L if any_ else []


def eplan_table():
    rows = []
    for cond, label in CORE:
        d = load(f"{cond}_s0_eplan")
        if d:
            rows.append((label, d["eplan"]))
    if not rows:
        return []
    keys = list(rows[0][1].keys())
    L = ["\\begin{table}[t]", "\\centering\\small",
         "\\caption{Planning error: mean engine centipawn loss, scored only where the",
         "move is legal and the probe independently recovers the exact position.",
         "This isolates judgement from memory.}",
         "\\label{tab:eplan}", "\\begin{tabular}{l" + "c" * len(keys) + "}", "\\toprule",
         "Model & " + " & ".join(k.replace("-", "--") for k in keys) + " \\\\", "\\midrule"]
    for label, d in rows:
        cells = ["n/a" if d[k]["mean_cp_loss"] is None else f"{d[k]['mean_cp_loss']:.0f}"
                 for k in keys]
        L.append(f"{label} & " + " & ".join(cells) + " \\\\")
    L += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    return L


def patch_table():
    rows = []
    for cond, label in CORE:
        d = load(f"{cond}_s0_patch")
        if d and "state_edit_all" in d:
            rows.append((label, d))
    if not rows:
        return []
    L = ["\\begin{table}[t]", "\\centering\\small",
         "\\caption{Causal intervention at the calibrated edit strength. Editing the",
         "state direction removes probability mass from moves out of the affected",
         "square; a norm-matched random direction does not. The manipulation check",
         "reports how often the edit actually changed the decoded belief.}",
         "\\label{tab:patch}", "\\begin{tabular}{lccccc}", "\\toprule",
         "Model & $n$ & belief flipped & mass before & mass after "
         "& random edit after \\\\", "\\midrule"]
    for label, d in rows:
        se, rd = d["state_edit_all"], d["norm_matched_random_edit"]
        L.append(f"{label} & {d['n']} & {d['manipulation_check_flip_rate']:.3f} & "
                 f"{se['mass_before']:.3f} & \\textbf{{{se['mass_after']:.3f}}} & "
                 f"{rd['mass_after']:.3f} \\\\")
    L += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    return L


def scale_table():
    """The seeded ladder: both horizons and the ceiling-normalised belief rate."""
    from ladder_stats import RUNGS, rung_stats, fmt as lfmt
    rows = [(lab, rung_stats(base)) for base, lab in RUNGS]
    rows = [(lab, st) for lab, st in rows if st]
    if not rows:
        return []
    L = ["\\begin{table}[t]", "\\centering\\small",
         "\\caption{The scale ladder, three seeds per rung. $H_{\\mathrm{ill}}$ is read",
         "off behaviour, the ply at which the illegal-move rate crosses five percent.",
         "$H_{\\mathrm{fid}}$ is read off the representation, the depth at which the",
         "probe first fails on the squares the chosen move touches more than half the",
         "time, and so cannot be moved by hedging. Belief consistency is normalised by",
         "each model's own decoding ceiling, because the ceiling itself rises with",
         "size. Mean\\,$\\pm$\\,s.d.\\ over seeds.}",
         "\\label{tab:scale}", "\\begin{tabular}{lrccccc}", "\\toprule",
         "Model & Params & seeds & $H_{\\mathrm{ill}}(.05)$ & $H_{\\mathrm{fid}}(.5)$ "
         "& belief\\,/\\,ceiling & mismatch \\\\", "\\midrule"]
    for lab, st in rows:
        L.append(f"{lab} & {fnum(st['params'])} & {st['n']} & "
                 f"{lfmt(st['h_ill'], 1)} & {lfmt(st['h_fid'], 1)} & "
                 f"{lfmt(st['bel_ratio'], 3)} & {lfmt(st['bel_mis'], 3)} \\\\")
    L += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    return L


def auc_table():
    """Aggregate against action-relevant, scored the same way."""
    d = load("attn_8L256_s0_auc")
    if not d:
        return []
    L = ["\\begin{table}[t]", "\\centering\\small",
         "\\caption{Aggregate and action-relevant state error compared like for",
         "like, as predictors of the same binary outcome on the same samples:",
         "whether the model's top-1 move is illegal. AUC is computed within each",
         "depth bucket so neither predictor can simply be tracking depth. The",
         "standardised coefficients come from a logistic fit containing both",
         "predictors. The aggregate measure is informative and it is the weaker of",
         "the two, and the gap widens with depth.}",
         "\\label{tab:auc}", "\\begin{tabular}{lccccr}", "\\toprule",
         "Ply & AUC aggregate & AUC action-relevant & $\\beta$ aggregate "
         "& $\\beta$ action-relevant & $n$ \\\\", "\\midrule"]
    for k, v in d["per_bucket"].items():
        lo, hi = k.split("-")
        L.append(f"{lo}--{hi} & {v['auc_aggregate']:.3f} & "
                 f"\\textbf{{{v['auc_action_relevant']:.3f}}} & "
                 f"{v['beta_aggregate']:+.3f} & "
                 f"\\textbf{{{v['beta_action_relevant']:+.3f}}} & {fnum(v['n'])} \\\\")
    p = d.get("pooled_with_depth_covariate")
    if p:
        L.append("\\midrule")
        L.append(f"pooled, depth as covariate & n/a & n/a & "
                 f"{p['beta_aggregate']:+.3f} & "
                 f"\\textbf{{{p['beta_action_relevant']:+.3f}}} & {fnum(p['n'])} \\\\")
    L += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    return L


def strictness_table():
    """Planning error as the conditioning on state is tightened."""
    d = load("attn_8L256_s0_eplanstrict")
    if not d:
        return []
    tols = d["tolerances"]
    L = ["\\begin{table}[t]", "\\centering\\small",
         "\\caption{Planning error against how strictly we condition on state. Each",
         "cell gives the median engine centipawn loss, the blunder rate, and the",
         "sample size, over legal moves whose touched squares the probe recovers and",
         "with at most the stated number of other squares wrong. Tightening the",
         "conditioning lowers the loss, so part of what looks like degraded judgement",
         "is state loss elsewhere on the board. Loss still grows with depth at fixed",
         "conditioning, so part of it is not. Cells with fewer than ten samples are",
         "reported as n/a.}",
         "\\label{tab:strict}",
         "\\begin{tabular}{l" + "c" * len(tols) + "}", "\\toprule",
         "Ply & " + " & ".join(("any" if t >= 64 else "$\\leq " + str(t) + "$")
                               for t in tols) + " \\\\",
         "\\midrule"]
    for lo, hi in d["buckets"]:
        cells = []
        for t in tols:
            c = d["cells"].get(f"{lo}-{hi}|tol{t}")
            if not c or c["median_cp"] is None or c["n"] < 10:
                cells.append("n/a")
            else:
                cells.append(f"{c['median_cp']:.0f} / {c['blunder_rate']:.2f} "
                             f"({c['n']})")
        L.append(f"{lo}--{hi} & " + " & ".join(cells) + " \\\\")
    L += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    return L


def boxes_table():
    """Domain 2: the same decomposition on permutation tracking."""
    rows = []
    for cond, label in [("attn", "Attention-only"),
                        ("alsb_l1", "Recurrent channel ($\\lambda{=}1$)")]:
        rs = [load(f"boxes_{cond}_s{s}") for s in (0, 1, 2)]
        rs = [r for r in rs if r]
        if rs:
            rows.append((label, rs))
    if not rows:
        return []
    L = ["\\begin{table}[t]", "\\centering\\small",
         "\\caption{Domain 2, permutation tracking. Answer accuracy at the shallowest",
         "and deepest depths against a chance rate of one in four, the probe's",
         "recovery of the queried object at depth, and belief consistency beside its",
         "mismatched control. The attention-only model solves the task well above",
         "chance while its state stays at chance under the probe, so it is not",
         "maintaining a linearly decodable assignment. The supervised channel is, and",
         "its errors are coherent with it.}",
         "\\label{tab:boxes}", "\\begin{tabular}{lccccc}", "\\toprule",
         "Model & seeds & acc.\\ shallow & acc.\\ deep & probe deep "
         "& belief / control \\\\", "\\midrule"]
    for label, rs in rows:
        L.append(f"{label} & {len(rs)} & "
                 f"{pm([r['ans_acc'][0] for r in rs])} & "
                 f"{pm([r['ans_acc'][-1] for r in rs])} & "
                 f"{pm([np.array(r['qobj_acc'])[r['best_layer']][-1] for r in rs])} & "
                 f"{pm([r['belief_overall'] for r in rs])} / "
                 f"{pm([r['belief_mismatched_overall'] for r in rs])} \\\\")
    L += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    return L



def abstain_table():
    """Selective prediction: what to abstain on."""
    d = load("attn_8L256_s0_abstain")
    if not d:
        return []
    cols = [("model_confidence", "model conf."), ("model_entropy", "entropy"),
            ("probe_uncertainty", "probe cert."), ("combined", "combined"),
            ("action_relevant_oracle", "oracle bound")]
    L = ["\\begin{table}[t]", "\\centering\\small",
         "\\caption{Selective prediction. Each column ranks positions by one signal",
         "and reports the illegal-move rate among those retained at each coverage.",
         "Probe certainty is computable at inference; the oracle column is not, and",
         "is shown only as an upper bound on what a perfect state monitor could",
         "achieve. Positions from ply 20 onward.}",
         "\\label{tab:abstain}",
         "\\begin{tabular}{l" + "c" * len(cols) + "}", "\\toprule",
         "Coverage & " + " & ".join(c[1] for c in cols) + " \\\\", "\\midrule"]
    for cov, row in d["coverage"].items():
        cells = [f"{row[k]:.4f}" if row.get(k) is not None else "n/a"
                 for k, _ in cols]
        pct = int(round(float(cov) * 100))
        L.append(f"{pct}\\% & " + " & ".join(cells) + " \\\\")
    L += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    return L


def main():
    parts = [("controls", controls_table), ("mechanism", mechanism_table),
             ("horizon", horizon_table), ("patch", patch_table),
             ("eplan", eplan_table), ("scale", scale_table),
             ("auc", auc_table), ("strictness", strictness_table),
             ("boxes", boxes_table),
             ("abstain", abstain_table)]
    L = ["% AUTO-GENERATED by code/make_tables.py. Do not edit by hand.", ""]
    outdir = os.path.dirname(OUT)
    for name, fn in parts:
        block = fn()
        L += block
        # also emit each table alone, so a shorter paper can include a subset
        with open(os.path.join(outdir, f"tab_{name}.tex"), "w",
                  encoding="utf-8") as fh:
            fh.write("\n".join(block) if block else "")
    open(OUT, "w", encoding="utf-8").write("\n".join(L))
    print(f"wrote {OUT} ({len(L)} lines) and {len(parts)} individual tables")


if __name__ == "__main__":
    main()
