"""Generate paper/results_macros.tex from results/*.json.

Playbook section 3: every number in the prose enters through \result{key}, and
every key traces back to the run that produced it.  Nothing here is typed by
hand, and a key with no measured value resolves to n/a rather than to a guess.
"""
import os, json, glob
import numpy as np
from horizon import interp_horizon

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
    pm(f"hint{tag}", [interp_horizon(r["illegal"], r["buckets"], 0.05) for r in rs],
       src, "{:.1f}")
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
    pm("hintZab", [interp_horizon(r["illegal"], r["buckets"], 0.05) for r in rs],
       src, "{:.1f}")

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

# patching (causal intervention)
for cond, tag in [("attn_8L256", "Attn"), ("alsb_l1_8L256", "AlsbOne"),
                  ("attn_12L512", "Rung4")]:
    d = load(f"{cond}_s0_patch")
    if not d:
        continue
    src = f"results/{cond}_s0_patch.json"
    put(f"patchAlpha{tag}", d["alpha"], src, "{:.2f}")
    put(f"patchN{tag}", d["n"], src, "{:.0f}")
    put(f"patchFlip{tag}", d["manipulation_check_flip_rate"], src)
    se, rd = d["state_edit_all"], d["norm_matched_random_edit"]
    put(f"patchMassBefore{tag}", se["mass_before"], src)
    put(f"patchMassAfter{tag}", se["mass_after"], src)
    put(f"patchDec{tag}", se["frac_decreased"], src)
    put(f"patchRandAfter{tag}", rd["mass_after"], src)
    put(f"patchRandDec{tag}", rd["frac_decreased"], src)
    if se["mass_before"] > 0:
        put(f"patchCut{tag}", 1 - se["mass_after"] / se["mass_before"], src)
        put(f"patchRandCut{tag}", 1 - rd["mass_after"] / se["mass_before"], src)

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



# ---------------------------------------- planning error vs conditioning strictness
d = load("attn_8L256_s0_eplanstrict")
if d:
    src = "results/attn_8L256_s0_eplanstrict.json"
    for (lo, hi) in d["buckets"]:
        for t in d["tolerances"]:
            c = d["cells"].get(f"{lo}-{hi}|tol{t}")
            if not c:
                continue
            tag = f"{lo}to{hi}tol{t}"
            put(f"epsMed{tag}", c["median_cp"], src, "{:.0f}")
            put(f"epsBl{tag}", c["blunder_rate"], src)
            put(f"epsN{tag}", c["n"], src, "{:.0f}")

# ------------------------------------------------------------ domain 2 (boxes)
for cond, tag in [("attn", "BoxAttn"), ("alsb_l1", "BoxAlsb")]:
    rs = [load(f"boxes_{cond}_s{s}") for s in (0, 1, 2)]
    rs = [r for r in rs if r]
    if not rs:
        continue
    src = f"results/boxes_{cond}_s*.json"
    put(f"chance{tag}", rs[0]["chance"], src, "{:.2f}")
    pm(f"ansShallow{tag}", [r["ans_acc"][0] for r in rs], src)
    pm(f"ansDeep{tag}", [r["ans_acc"][-1] for r in rs], src)
    pm(f"probeDeep{tag}",
       [np.array(r["qobj_acc"])[r["best_layer"]][-1] for r in rs], src)
    pm(f"belief{tag}", [r["belief_overall"] for r in rs], src)
    pm(f"beliefMis{tag}", [r["belief_mismatched_overall"] for r in rs], src)
    pm(f"estateDeep{tag}", [r["estate"][-1] for r in rs], src)
    pm(f"eplanDeep{tag}", [r["eplan"][-1] for r in rs], src)



# ------------------------------------- like-for-like AUC and the read-out ablation
d = load("attn_8L256_s0_auc")
if d:
    src = "results/attn_8L256_s0_auc.json"
    for k, v in d["per_bucket"].items():
        tag = k.replace("-", "to")
        put(f"aucAgg{tag}", v["auc_aggregate"], src)
        put(f"aucAct{tag}", v["auc_action_relevant"], src)
    pooled = d["pooled_with_depth_covariate"]
    put("betaAgg", pooled["beta_aggregate"], src)
    put("betaAct", pooled["beta_action_relevant"], src)
    aa = [v["auc_aggregate"] for v in d["per_bucket"].values()]
    ac = [v["auc_action_relevant"] for v in d["per_bucket"].values()]
    put("aucAggMin", min(aa), src); put("aucAggMax", max(aa), src)
    put("aucActMin", min(ac), src); put("aucActMax", max(ac), src)

d = load("readout_ablation_check")
if d:
    src = "results/readout_ablation_check.json"
    vals = list(d.values())
    pm("lmIntact", [v["lm_loss_intact"] for v in vals], src)
    pm("lmAblated", [v["lm_loss_ablated"] for v in vals], src)
    pm("lmDelta", [v["delta"] for v in vals], src)
    pm("injRatio", [v["injection_ratio"] for v in vals], src)

# belief normalised by each model's own decoding ceiling, across the ladder
_rungs = []
for _n in ("attn_6L192_s0", "attn_8L256_s0", "attn_12L256_s0", "attn_8L384_s0",
           "attn_12L384_s0", "attn_12L512_s0"):
    _b = load(f"{_n}_belief"); _r = load(_n)
    if _b and _r:
        _o = _b["overall"]
        _rungs.append((_r["params"], _o["illegal_legal_in_belief"] /
                       max(_o["legal_legal_in_belief"], 1e-9)))
if len(_rungs) >= 2:
    _rungs.sort()
    put("belRatioSmall", _rungs[0][1], "results/*_belief.json")
    put("belRatioLarge", _rungs[-1][1], "results/*_belief.json")
    put("paramsSmall", _rungs[0][0], "results/*.json", "{:,.0f}")
    put("paramsLarge", _rungs[-1][0], "results/*.json", "{:,.0f}")

# legal probability mass, intact against ablated read-out
_i, _a = [], []
for _s in (0, 1, 2):
    _x = load(f"alsb_l1_8L256_s{_s}"); _y = load(f"alsb_l1_8L256_s{_s}_zablate")
    if _x: _i.append(_x["legal_mass"][4])
    if _y: _a.append(_y["legal_mass"][4])
if _i: pm("massIntact", _i, "results/alsb_l1_8L256_s*.json")
if _a: pm("massAblated", _a, "results/alsb_l1_8L256_s*_zablate.json")



# ------------------------------- horizon defined on action-relevant fidelity
_TAGMAP = {"attn_6L192_s0": "Rung1", "attn_8L256_s0": "Attn",
           "attn_12L256_s0": "AttnDeep", "attn_8L384_s0": "AttnWide",
           "attn_12L384_s0": "Rung5", "attn_12L512_s0": "Rung6"}
for _n, _tag in _TAGMAP.items():
    _d = load(f"{_n}_auc")
    if not _d:
        continue
    _src = f"results/{_n}_auc.json"
    _ks = list(_d["per_bucket"].keys())
    _err = [_d["per_bucket"][k]["action_relevant_error"] for k in _ks]
    _bk = [tuple(int(x) for x in k.split("-")) for k in _ks]
    put(f"hfid{_tag}", interp_horizon(_err, _bk, 0.5), _src, "{:.1f}")
    put(f"actErrFirst{_tag}", _err[0], _src)
    put(f"actErrLast{_tag}", _err[-1], _src)



# behavioural horizon for the ladder rungs, so both horizons can be compared
for _n, _tag in {"attn_6L192_s0": "Rung1", "attn_12L384_s0": "Rung5",
                 "attn_12L512_s0": "Rung6"}.items():
    _r = load(_n)
    if _r:
        put(f"hint{_tag}", interp_horizon(_r["illegal"], _r["buckets"], 0.05),
            f"results/{_n}.json", "{:.1f}")
        put(f"params{_tag}", _r["params"], f"results/{_n}.json", "{:,.0f}")



# ------------------------------------------- seeded ladder (three seeds per rung)
from ladder_stats import RUNGS as _RUNGS, rung_stats as _rs, ms as _ms
_TAG = {"attn_6L192": "Rung1", "attn_8L256": "Attn", "attn_12L256": "AttnDeep",
        "attn_8L384": "AttnWide", "attn_12L384": "Rung5", "attn_12L512": "Rung6"}
for _base, _lab in _RUNGS:
    _st = _rs(_base)
    if not _st:
        continue
    _t = _TAG[_base]
    _src = f"results/{_base}_s*.json"
    put(f"nseed{_t}", _st["n"], _src, "{:.0f}")
    for _key, _name, _dp in (("h_ill", "hint", "{:.1f}"), ("h_fid", "hfid", "{:.1f}"),
                             ("bel_ratio", "belRatioN", "{:.3f}"),
                             ("act_first", "actErrFirst", "{:.3f}")):
        _m, _sd = _ms(_st[_key])
        if _m is not None:
            put(f"{_name}{_t}", _m, _src, _dp)
            put(f"{_name}{_t}SD", _sd, _src, _dp)



# ---------------------------------------------------- selective prediction
_ab = load("attn_8L256_s0_abstain")
if _ab:
    _src = "results/attn_8L256_s0_abstain.json"
    put("abBase", _ab["base_illegal_rate"], _src)
    put("abAucConf", _ab["auc"]["model_confidence"], _src)
    put("abAucEnt", _ab["auc"]["model_entropy"], _src)
    put("abAucProbe", _ab["auc"]["probe_uncertainty"], _src)
    _c50 = _ab["coverage"].get("0.5", {})
    put("abConf50", _c50.get("model_confidence"), _src)
    put("abEnt50", _c50.get("model_entropy"), _src)
    put("abProbe50", _c50.get("probe_uncertainty"), _src)
    put("abComb50", _c50.get("combined"), _src)
    put("abOracle50", _c50.get("action_relevant_oracle"), _src)
    put("abBetaConf", _ab["joint_fit"]["beta_model_confidence"], _src)
    put("abBetaProbe", _ab["joint_fit"]["beta_probe_uncertainty"], _src)



# ------------------------------------------------- square controls, exact fidelity
_sq = load("attn_8L256_s0_sqcontrol")
if _sq:
    _src = "results/attn_8L256_s0_sqcontrol.json"
    _m = {"whole_board": "sqWholeBoard", "two_random": "sqTwoRandom",
          "two_random_occupied": "sqTwoRandomOcc", "source_only": "sqSourceOnly",
          "destination_only": "sqDestOnly", "move_touched": "sqMoveTouched"}
    for _k, _name in _m.items():
        _v = _sq["pooled"].get(_k)
        if _v:
            put(_name, _v["auc"], _src)
            put(_name + "Lo", _v["ci_lo"], _src)
            put(_name + "Hi", _v["ci_hi"], _src)

_r = load("attn_8L256_s0")
if _r:
    _ex = np.array(_r["fidelity_exact"])[_r["best_layer"]]
    put("exactFirst", _ex[0], "results/attn_8L256_s0.json")
    put("exactMid", _ex[2], "results/attn_8L256_s0.json")


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
