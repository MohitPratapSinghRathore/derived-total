"""Generate paper/results_macros.tex from results/*.json.

Playbook section 3: every number in the prose enters through \result{key}, and
every key traces back to the run that produced it.  Nothing here is typed by
hand, and a key with no measured value resolves to n/a rather than to a guess.
"""
import os, json, glob
import numpy as np

RES = os.path.join(os.path.dirname(__file__), "..", "results")
OUT = os.path.join(os.path.dirname(__file__), "..", "paper", "results_macros.tex")

M = {}
PROV = {}


def put(key, val, src, fmt="{:.3f}"):
    if val is None or (isinstance(val, float) and not np.isfinite(val)):
        M[key] = "n/a"
    elif isinstance(val, str):
        M[key] = val
    else:
        M[key] = fmt.format(val)
    PROV[key] = src


def load(name):
    p = os.path.join(RES, f"{name}.json")
    return json.load(open(p)) if os.path.exists(p) else None


def seed_set(cond):
    out = []
    for s in (0, 1, 2):
        r = load(f"{cond}_s{s}")
        if r:
            out.append(r)
    return out


def pm(key, vals, src, fmt="{:.3f}"):
    v = np.array([x for x in vals if x is not None], dtype=float)
    if len(v) == 0:
        put(key, None, src); put(key + "SD", None, src); return
    put(key, float(v.mean()), src, fmt)
    put(key + "SD", float(v.std(ddof=1)) if len(v) > 1 else 0.0, src, fmt)
    put(key + "N", len(v), src, "{:.0f}")


# ---------------------------------------------------------------- dataset facts
proc = os.path.join(os.path.dirname(__file__), "..", "data", "proc")
try:
    import numpy as _np
    tot = 0
    for sp in ("train_lm", "train_probe", "eval"):
        d = _np.load(os.path.join(proc, f"{sp}.npz"))
        put(f"n{sp.replace('_','')}", len(d["lens"]), "data/proc", "{:,.0f}")
        tot += len(d["lens"])
        if sp == "eval":
            put("meanPly", float(d["lens"].mean()), "data/proc", "{:.1f}")
    put("nGames", tot, "data/proc", "{:,.0f}")
    put("nVocab", len(json.load(open(os.path.join(proc, "vocab.json")))),
        "data/proc", "{:,.0f}")
except Exception as e:
    print("dataset macros unavailable:", e)

# ------------------------------------------------------------------ chess arm
BI = 4          # the 40-50 ply bucket
for cond, tag in [("attn_8L256", "Attn"), ("attnpm_8L256", "AttnPM"),
                  ("alsb_l0_8L256", "AlsbZero"), ("alsb_l1_8L256", "AlsbOne"),
                  ("attn_12L256", "AttnDeep"), ("attn_8L384", "AttnWide")]:
    rs = seed_set(cond)
    if not rs:
        continue
    src = f"results/{cond}_s*.json"
    put(f"params{tag}", rs[0]["params"], src, "{:,.0f}")
    pm(f"occ40{tag}", [np.array(r["fidelity_occ"])[r["best_layer"]][BI] for r in rs], src)
    pm(f"ill40{tag}", [r["illegal"][BI] for r in rs], src)
    pm(f"hill05{tag}", [r["H_ill_05"] for r in rs], src, "{:.0f}")
    pm(f"hill10{tag}", [r["H_ill_10"] for r in rs], src, "{:.0f}")
    pm(f"val{tag}", [r["train_log"][-1]["val"] for r in rs], src)
    pm(f"bestLayer{tag}", [r["best_layer"] for r in rs], src, "{:.1f}")

# z-ablation
rs = [load(f"alsb_l1_8L256_s{s}_zablate") for s in (0, 1, 2)]
rs = [r for r in rs if r]
if rs:
    src = "results/alsb_l1_8L256_s*_zablate.json"
    pm("occ40Zab", [np.array(r["fidelity_occ"])[r["best_layer"]][BI] for r in rs], src)
    pm("ill40Zab", [r["illegal"][BI] for r in rs], src)
    pm("hill05Zab", [r["H_ill_05"] for r in rs], src, "{:.0f}")

# baseline controls + curve endpoints
r = load("attn_8L256_s0")
if r:
    src = "results/attn_8L256_s0.json"
    bl = r["best_layer"]
    occ = np.array(r["fidelity_occ"])[bl]
    put("occFirst", occ[0], src)
    put("occLast", occ[-1], src)
    put("illFirst", r["illegal"][0], src, "{:.4f}")
    put("illLast", r["illegal"][-1], src, "{:.3f}")
    put("massFirst", r["legal_mass"][0], src)
    put("massLast", r["legal_mass"][-1], src)
    put("nLayers", len(r["fidelity_occ"]), src, "{:.0f}")
    put("bestLayerBase", bl, src, "{:.0f}")
    if "control_random_occ" in r:
        put("ctlRand40", np.array(r["control_random_occ"])[bl][BI], src)
        put("ctlShuf40", np.array(r["control_shuffled_occ"])[bl][BI], src)
    put("ctlMaj40", r["majority_occ"][BI], src)
    lp = np.array(r["fidelity_occ"]).mean(1)
    put("layerProfFirst", lp[0], src)
    put("layerProfLast", lp[-1], src)
    put("exactPly", r["H_exact_05"], src, "{:.0f}")

# E_plan
for cond, tag in [("attn_8L256", "Attn"), ("attnpm_8L256", "AttnPM"),
                  ("alsb_l0_8L256", "AlsbZero"), ("alsb_l1_8L256", "AlsbOne")]:
    d = load(f"{cond}_s0_eplan")
    if not d:
        continue
    src = f"results/{cond}_s0_eplan.json"
    ks = list(d["eplan"].keys())
    for k in ks:
        kk = k.replace("-", "to")
        put(f"cp{kk}{tag}", d["eplan"][k]["mean_cp_loss"], src, "{:.0f}")
        put(f"blunder{kk}{tag}", d["eplan"][k]["blunder_rate"], src)

# patching
for cond, tag in [("attn_8L256", "Attn"), ("alsb_l1_8L256", "AlsbOne")]:
    d = load(f"{cond}_s0_patch")
    if not d:
        continue
    src = f"results/{cond}_s0_patch.json"
    put(f"patchFrac{tag}", d["frac_mass_decreased"], src)
    put(f"patchDrop{tag}", d["mean_mass_drop"], src, "{:.4f}")
    put(f"patchN{tag}", d["n"], src, "{:.0f}")

# ------------------------------------------------- coupling and belief analyses
for cond, tag in [("attn_8L256", "Attn"), ("attnpm_8L256", "AttnPM"),
                  ("alsb_l0_8L256", "AlsbZero"), ("alsb_l1_8L256", "AlsbOne"),
                  ("attn_6L192", "Rung1"), ("attn_12L384", "Rung3"),
                  ("attn_12L512", "Rung4")]:
    d = load(f"{cond}_s0_coupling")
    if d:
        src = f"results/{cond}_s0_coupling.json"
        L = d["locality"]
        it = L["illegal_touched_wrong"] + L["illegal_touched_ok"]
        lt = L["legal_touched_wrong"] + L["legal_touched_ok"]
        put(f"locIll{tag}", L["illegal_touched_wrong"] / max(it, 1), src)
        put(f"locLeg{tag}", L["legal_touched_wrong"] / max(lt, 1), src)
        cs = [c for c in d["within_bucket_corr_wrongsquares_illegal"] if c is not None]
        if cs:
            put(f"corrGlobalMin{tag}", min(cs), src, "{:.3f}")
            put(f"corrGlobalMax{tag}", max(cs), src, "{:.3f}")
        ages = d["error_by_age"]
        put(f"ageFirst{tag}", ages[0], src)
        put(f"agePeak{tag}", max(ages), src)
        put(f"ageLast{tag}", ages[-1], src)

    d = load(f"{cond}_s0_belief")
    if d:
        src = f"results/{cond}_s0_belief.json"
        o = d["overall"]
        put(f"belIll{tag}", o["illegal_legal_in_belief"], src)
        put(f"belLeg{tag}", o["legal_legal_in_belief"], src)
        put(f"belRand{tag}", o["randomillegal_legal_in_belief"], src)
        put(f"belMis{tag}", o.get("mismatched_belief_legal"), src)
        if o.get("illegal_legal_in_belief") and o.get("randomillegal_legal_in_belief"):
            put(f"belRatio{tag}",
                o["illegal_legal_in_belief"] / max(o["randomillegal_legal_in_belief"], 1e-9),
                src, "{:.0f}")
        if o.get("mismatched_belief_legal"):
            put(f"belRatioMis{tag}",
                o["illegal_legal_in_belief"] / max(o["mismatched_belief_legal"], 1e-9),
                src, "{:.1f}")
        if "illegal_ci95" in o:
            put(f"belIllLo{tag}", o["illegal_ci95"][0], src)
            put(f"belIllHi{tag}", o["illegal_ci95"][1], src)

# ------------------------------------------------------------------ domain 2
for cond, tag in [("attn", "Attn"), ("attnpm", "AttnPM"),
                  ("alsb_l0", "AlsbZero"), ("alsb_l1", "AlsbOne")]:
    rs = [load(f"synth_{cond}_s{s}") for s in (0, 1, 2)]
    rs = [r for r in rs if r]
    if not rs:
        continue
    src = f"results/synth_{cond}_s*.json"
    put(f"synParams{tag}", rs[0]["params"], src, "{:,.0f}")
    pm(f"synAccDeep{tag}", [r["ans_acc"][-1] for r in rs], src)
    pm(f"synAccShallow{tag}", [r["ans_acc"][0] for r in rs], src)
    pm(f"synStateDeep{tag}", [r["estate"][-1] for r in rs], src)
    pm(f"synPlanDeep{tag}", [r["eplan"][-1] for r in rs], src)
    pm(f"synStateShallow{tag}", [r["estate"][0] for r in rs], src)
    pm(f"synPlanShallow{tag}", [r["eplan"][0] for r in rs], src)


def main():
    lines = ["% AUTO-GENERATED by code/make_macros.py. Do not edit by hand.",
             "% Every value traces to the results file named in the comment.",
             r"\makeatletter",
             r"\newcommand{\result}[1]{%",
             r"  \expandafter\ifx\csname result@#1\endcsname\relax",
             r"    \textbf{??}\PackageWarning{results}{undefined result key: #1}%",
             r"  \else\csname result@#1\endcsname\fi}",
             r"\newcommand{\defresult}[2]{\expandafter\def\csname result@#1\endcsname{#2}}",
             r"\makeatother", ""]
    for k in sorted(M):
        lines.append(f"\\defresult{{{k}}}{{{M[k]}}}   % {PROV[k]}")
    open(OUT, "w", encoding="utf-8").write("\n".join(lines) + "\n")
    na = sum(1 for v in M.values() if v == "n/a")
    print(f"wrote {OUT}: {len(M)} keys ({na} n/a)")
    json.dump({k: {"value": M[k], "source": PROV[k]} for k in M},
              open(os.path.join(RES, "macro_provenance.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
