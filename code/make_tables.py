"""Stage 7: emit LaTeX tables straight from results/*.json.

Nothing in the paper's tables is typed by hand; this file is the only path from
measurement to manuscript, so a number in the PDF can always be traced to a run.
"""
import os, json, glob
import numpy as np

RES = os.path.join(os.path.dirname(__file__), "..", "results")
OUT = os.path.join(os.path.dirname(__file__), "..", "paper", "tables.tex")

CHESS = [("attn_8L256", "Attention-only, 8L/256"),
         ("attnpm_8L256", "Attention-only, param-matched"),
         ("alsb_l0_8L256", "ALSB ($\\lambda{=}0$)"),
         ("alsb_l1_8L256", "ALSB ($\\lambda{=}1$)"),
         ("attn_12L256", "Attention-only, 12L/256"),
         ("attn_8L384", "Attention-only, 8L/384")]
SYNTH = [("attn", "Attention-only"),
         ("attnpm", "Attention-only, param-matched"),
         ("alsb_l0", "ALSB ($\\lambda{=}0$)"),
         ("alsb_l1", "ALSB ($\\lambda{=}1$)")]


def seeds(pattern, extra=""):
    out = []
    for f in sorted(glob.glob(os.path.join(RES, pattern))):
        b = os.path.basename(f)
        if any(t in b for t in ("_probes", "_eplan", "_patch", "summary")):
            continue
        if not extra and "zablate" in b:
            continue
        out.append(json.load(open(f)))
    return out


def pm(vals, fmt="{:.3f}"):
    v = np.array(vals, dtype=float)
    if len(v) == 0:
        return "---"
    if len(v) == 1:
        return fmt.format(v[0])
    return (fmt + "\\,$\\pm$\\," + fmt).format(v.mean(), v.std(ddof=1))


def chess_table():
    L = ["\\begin{table}[t]", "\\centering", "\\small",
         "\\caption{Chess arm. Occupied-square state accuracy (occ), illegal top-1 move",
         "rate, and behavioural horizon $H_{\\mathrm{ill}}$ (first ply bucket where the",
         "illegal rate exceeds the threshold). Mean\\,$\\pm$\\,s.d.\\ over seeds.}",
         "\\label{tab:chess}",
         "\\begin{tabular}{lrccccc}", "\\toprule",
         "Model & Params & seeds & occ@40--50 & illegal@40--50 & $H_{\\mathrm{ill}}(.05)$ & $H_{\\mathrm{ill}}(.10)$ \\\\",
         "\\midrule"]
    for cond, label in CHESS:
        rs = seeds(f"{cond}_s*.json")
        if not rs:
            continue
        bi = 4
        occ = [np.array(r["fidelity_occ"])[r["best_layer"]][bi] for r in rs]
        ill = [r["illegal"][bi] for r in rs]
        h5 = [r["H_ill_05"] for r in rs]
        h10 = [r["H_ill_10"] for r in rs]
        L.append(f"{label} & {rs[0]['params']:,} & {len(rs)} & {pm(occ)} & {pm(ill)} & "
                 f"{pm(h5, '{:.0f}')} & {pm(h10, '{:.0f}')} \\\\".replace(",", "{,}"))
    # z-ablation
    rs = seeds("alsb_l1_8L256_s*_zablate.json", extra="z")
    rs = [r for r in rs if r.get("ablate_z")]
    if rs:
        bi = 4
        occ = [np.array(r["fidelity_occ"])[r["best_layer"]][bi] for r in rs]
        ill = [r["illegal"][bi] for r in rs]
        h5 = [r["H_ill_05"] for r in rs]
        h10 = [r["H_ill_10"] for r in rs]
        L.append("\\midrule")
        L.append(f"ALSB ($\\lambda{{=}}1$), $z$ ablated at test & --- & {len(rs)} & {pm(occ)} & "
                 f"{pm(ill)} & {pm(h5,'{:.0f}')} & {pm(h10,'{:.0f}')} \\\\")
    L += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    return L


def controls_table():
    rs = seeds("attn_8L256_s0.json")
    if not rs or "control_random_occ" not in rs[0]:
        return []
    r = rs[0]; bl = r["best_layer"]; b = r["buckets"]
    occ = np.array(r["fidelity_occ"])[bl]
    cs = np.array(r["control_shuffled_occ"])[bl]
    cr = np.array(r["control_random_occ"])[bl]
    mj = np.array(r["majority_occ"])
    ill = np.array(r["illegal"])
    L = ["\\begin{table}[t]", "\\centering", "\\small",
         "\\caption{Chess arm, baseline model: state decodability against every trivial",
         "explanation. Occupied-square accuracy at the best layer, versus a per-(square,",
         "ply) majority predictor, a cross-game label control, and a randomised-weight",
         "model of identical architecture. The gap is what the trained model knows.}",
         "\\label{tab:controls}",
         "\\begin{tabular}{lccccc}", "\\toprule",
         "Ply & model & majority & cross-game & random-weight & illegal rate \\\\",
         "\\midrule"]
    for i, (lo, hi) in enumerate(b):
        L.append(f"{lo}--{hi} & {occ[i]:.3f} & {mj[i]:.3f} & {cs[i]:.3f} & {cr[i]:.3f} "
                 f"& {ill[i]:.4f} \\\\")
    L += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    return L


def eplan_table():
    rows = []
    for cond, label in CHESS[:4]:
        f = os.path.join(RES, f"{cond}_s0_eplan.json")
        if not os.path.exists(f):
            continue
        d = json.load(open(f))["eplan"]
        rows.append((label, d))
    if not rows:
        return []
    keys = list(rows[0][1].keys())
    L = ["\\begin{table}[t]", "\\centering", "\\small",
         "\\caption{Chess arm, $E_{\\mathrm{plan}}$: engine centipawn loss CONDITIONED on the",
         "top-1 move being legal and the probe recovering the exact position. This is the",
         "H4 test -- the intervention should move $E_{\\mathrm{state}}$, not this.}",
         "\\label{tab:eplan}",
         "\\begin{tabular}{l" + "c" * len(keys) + "}", "\\toprule",
         "Model & " + " & ".join(k.replace("-", "--") for k in keys) + " \\\\",
         "\\midrule"]
    for label, d in rows:
        cells = []
        for k in keys:
            v = d[k]["mean_cp_loss"]
            cells.append("---" if v is None else f"{v:.0f}")
        L.append(f"{label} & " + " & ".join(cells) + " \\\\")
    L += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    return L


def synth_table():
    L = ["\\begin{table}[t]", "\\centering", "\\small",
         "\\caption{Domain 2 (variable-state tracking). $E_{\\mathrm{state}}$ is the rate at",
         "which the two queried variables are not decodable at the query; $E_{\\mathrm{plan}}$",
         "is the answer-error rate GIVEN both are decodable. Deepest bucket (32--40 steps).",
         "Mean\\,$\\pm$\\,s.d.\\ over seeds.}",
         "\\label{tab:synth}",
         "\\begin{tabular}{lrcccc}", "\\toprule",
         "Model & Params & seeds & answer acc. & $E_{\\mathrm{state}}$ & $E_{\\mathrm{plan}}$ \\\\",
         "\\midrule"]
    any_ = False
    for cond, label in SYNTH:
        rs = seeds(f"synth_{cond}_s*.json")
        if not rs:
            continue
        any_ = True
        bi = -1
        acc = [r["ans_acc"][bi] for r in rs]
        es = [r["estate"][bi] for r in rs]
        ep = [r["eplan"][bi] for r in rs]
        L.append(f"{label} & {rs[0]['params']:,} & {len(rs)} & {pm(acc)} & {pm(es)} & "
                 f"{pm(ep)} \\\\".replace(",", "{,}"))
    L += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    return L if any_ else []


def patch_table():
    rows = []
    for cond, label in CHESS[:4]:
        f = os.path.join(RES, f"{cond}_s0_patch.json")
        if os.path.exists(f):
            rows.append((label, json.load(open(f))))
    if not rows:
        return []
    L = ["\\begin{table}[t]", "\\centering", "\\small",
         "\\caption{Causal patching. Editing probe-identified state directions toward",
         "``square empty'' should reduce the probability mass on moves originating from",
         "that square. A representation that is merely correlated would not respond.}",
         "\\label{tab:patch}",
         "\\begin{tabular}{lccc}", "\\toprule",
         "Model & $n$ & frac.\\ mass decreased & mean mass drop \\\\", "\\midrule"]
    for label, d in rows:
        L.append(f"{label} & {d['n']} & {d['frac_mass_decreased']:.3f} & "
                 f"{d['mean_mass_drop']:.4f} \\\\")
    L += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    return L


def main():
    L = ["% AUTO-GENERATED by code/make_tables.py -- do not edit by hand", ""]
    for f in (controls_table, chess_table, eplan_table, patch_table, synth_table):
        L += f()
    open(OUT, "w").write("\n".join(L))
    print(f"wrote {OUT} ({len(L)} lines)")


if __name__ == "__main__":
    main()
