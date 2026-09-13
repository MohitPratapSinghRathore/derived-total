"""Generate paper_scaling/results_macros.tex.

No number reaches the manuscript except through a \result{} macro written here,
each tagged with the results file it came from, so every figure in the prose is
traceable and none can go stale silently.
"""
import os, json
import numpy as np
import ds_stats as S

OUT_DIR = os.path.join(S.ROOT, "paper_scaling")
OUT = os.path.join(OUT_DIR, "results_macros.tex")
M, PROV = {}, {}


def put(k, v, src, fmt="{:.3f}"):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        M[k] = "n/a"
    elif isinstance(v, str):
        M[k] = v
    else:
        M[k] = fmt.format(v)
    PROV[k] = src


def sgn(v):
    return f"{v:+.3f}".replace("-", "$-$", 1) if v < 0 else f"{v:+.3f}"


def params_fmt(n):
    return f"{n / 1e6:.1f}M"


def pval(p):
    m, e = f"{p:.1e}".split("e")
    return f"${m}\\times10^{{{int(e)}}}$"


# ======================================================== Act 1: our ladder
rows = S.ladder_rows()
LAD = "results/attn_*_{structure,natdiv}.json via ds_stats"
put("nRuns", len(rows), LAD, "{:d}")
put("nRungs", len(S.RUNGS), LAD, "{:d}")
put("nSeeds", len(S.SEEDS), LAD, "{:d}")
put("paramsSmall", params_fmt(S.PARAMS[S.RUNGS[0]]), LAD)
put("paramsLarge", params_fmt(S.PARAMS[S.RUNGS[-1]]), LAD)
put("spanLadder", S.PARAMS[S.RUNGS[-1]] / S.PARAMS[S.RUNGS[0]], LAD, "{:.1f}")

KEYS = {"fromEmpty": "from_empty", "fromOpp": "from_opponent", "toOwn": "to_own",
        "geometry": "geometry", "leavesCheck": "leaves_check",
        "corrLocal": "correct_local_belief", "illegal": "illegal_rate"}
EXPS = {}
for tag, key in ((t[0].upper() + t[1:], k) for t, k in KEYS.items()):
    m = S.rung_means(rows, key)
    e = S.exponent(rows, key)
    EXPS[key] = e
    put(f"abs{tag}First", m[0], LAD, "{:.4f}")
    put(f"abs{tag}Last", m[-1], LAD, "{:.4f}")
    put(f"ratio{tag}", m[0] / m[-1], LAD, "{:.2f}")
    put(f"exp{tag}", sgn(e["point"]), LAD)
    put(f"exp{tag}Lo", sgn(e["lo"]), LAD)
    put(f"exp{tag}Hi", sgn(e["hi"]), LAD)

for tag, (a, b) in {"ProbeFree": ("leaves_check", "from_empty"),
                    "Corr": ("correct_local_belief", "from_empty")}.items():
    d = S.differential(rows, a, b)
    put(f"diff{tag}", sgn(d["point"]), LAD)
    put(f"diff{tag}Lo", sgn(d["lo"]), LAD)
    put(f"diff{tag}Hi", sgn(d["hi"]), LAD)
    put(f"diff{tag}Pct", 100 * d["frac_pos"], LAD, "{:.0f}")

for tag, key in {"FromEmpty": "share_from_empty", "Geometry": "share_geometry",
                 "Leaves": "share_leaves_check", "Policy": "share_policy"}.items():
    m = S.rung_means(rows, key)
    put(f"share{tag}First", m[0], LAD, "{:.3f}")
    put(f"share{tag}Last", m[-1], LAD, "{:.3f}")
put("shareOtherMax", max(q["share_other"] for q in rows), LAD, "{:.4f}")

rises = 0
for r in S.RUNGS:
    sub = [q for q in rows if q["rung"] == r and q["lc_early"] is not None]
    if sub and np.mean([q["lc_late"] for q in sub]) > np.mean([q["lc_early"] for q in sub]):
        rises += 1
put("lcDepthRises", rises, LAD, "{:d}")
big = [q for q in rows if q["rung"] == S.RUNGS[-1]]
put("lcEarlyLarge", np.mean([q["lc_early"] for q in big]), LAD, "{:.3f}")
put("lcLateLarge", np.mean([q["lc_late"] for q in big]), LAD, "{:.3f}")
ok = [q["lc_when_local_ok"] for q in rows if q["lc_when_local_ok"] is not None]
put("lcWhenLocalOk", float(np.mean(ok)), LAD, "{:.3f}")
_lg = 0
_geo = []
for q in S.RUNGS:
    for sd in S.SEEDS:
        _d = S.load(f"attn_{q}_s{sd}_structure.json")
        if not _d:
            continue
        _v = {c: _d["overall"][c].get("share_when_belief_ok") for c in S.CLASSES}
        if None in _v.values():
            continue
        _lg += int(max(_v, key=_v.get) == "leaves_check")
        _geo.append(_v["geometry"])
put("lcLargestCount", _lg, LAD, "{:d}")
put("geomWhenLocalOk", float(np.mean(_geo)), LAD, "{:.3f}")

# illustrative projection, clearly labelled in the text as extrapolation
last = {k: S.rung_means(rows, k)[-1] for k in ("correct_local_belief", "leaves_check",
                                                "from_empty")}
scale = 1e9 / S.PARAMS[S.RUNGS[-1]]
put("projFactor", scale, LAD, "{:.0f}")
put("projCorr", last["correct_local_belief"] * scale ** EXPS["correct_local_belief"]["point"], LAD, "{:.4f}")
put("projLeaves", last["leaves_check"] * scale ** EXPS["leaves_check"]["point"], LAD, "{:.4f}")
put("projFromEmpty", last["from_empty"] * scale ** EXPS["from_empty"]["point"], LAD, "{:.4f}")

# ======================================================== external validity
ext = S.external_rows()
EX = "results/chessgpt_structure_lichess_{6L,8L,16L}.json"
for i, r in enumerate(ext):
    t = ["Six", "Eight", "Sixteen"][i]
    put(f"kParams{t}", params_fmt(r["params"]), EX)
    put(f"kIll{t}", r["illegal_rate"], EX, "{:.4f}")
    put(f"kFail{t}", r["n_failures"], EX, "{:,d}")
    put(f"kPos{t}", r["n_positions"], EX, "{:,d}")
    put(f"kAbsUnr{t}", r["absolute"]["unreachable"], EX, "{:.5f}")
    put(f"kAbsLc{t}", r["absolute"]["leaves_check"], EX, "{:.5f}")
    put(f"kShareUnr{t}", r["shares"]["unreachable"]["mean"], EX, "{:.3f}")
    put(f"kShareLc{t}", r["shares"]["leaves_check"]["mean"], EX, "{:.3f}")
put("kSpan", ext[-1]["params"] / ext[0]["params"], EX, "{:.0f}")
ee = S.external_exponents(ext)
for tag, key in (("Unr", "unreachable"), ("Lc", "leaves_check"), ("Own", "to_own")):
    put(f"kExp{tag}", sgn(ee[key]["point"]), EX)
    put(f"kExp{tag}Lo", sgn(ee[key]["lo"]), EX)
    put(f"kExp{tag}Hi", sgn(ee[key]["hi"]), EX)
put("kDiff", sgn(ee["differential"]["point"]), EX)
put("kDiffLo", sgn(ee["differential"]["lo"]), EX)
put("kDiffHi", sgn(ee["differential"]["hi"]), EX)
st6 = ext[0].get("strata", {})
put("kSixLcEarly", st6.get("20-40", {}).get("leaves_check"), EX, "{:.3f}")
put("kSixLcLate", st6.get("60-120", {}).get("leaves_check"), EX, "{:.3f}")
put("kSixUnrEarly", st6.get("20-40", {}).get("unreachable"), EX, "{:.3f}")
put("kSixUnrLate", st6.get("60-120", {}).get("unreachable"), EX, "{:.3f}")

oth = S.load("othello_structure_synthetic_model.json")
OT = "results/othello_structure_synthetic_model.json"
put("othPos", oth["n_positions"], OT, "{:,d}")
put("othFail", oth["n_failures"], OT, "{:d}")
put("othRate", oth["illegal_rate"], OT, "{:.5f}")
put("othOccupied", oth["counts"]["occupied"], OT, "{:d}")
put("othNoFlank", oth["counts"]["no_flank"], OT, "{:d}")
put("othShareNoFlank", oth["shares"]["no_flank"], OT, "{:.3f}")
for band, tag in (("5-20", "A"), ("20-40", "B"), ("40-59", "C")):
    put(f"othDepth{tag}", oth["strata"].get(band, {}).get("n"), OT, "{:d}")

# Chess-GPT load-bearing measurements
CG = "results/chessgpt2_{probe,belief_*,edit_*}.json"
pr = S.load("chessgpt2_probe.json")
bl = pr["best_layer"]
buckets = [tuple(b) for b in pr["buckets"]]
acc = pr["acc_by_layer"][bl]
put("cgProbeFirst", acc[buckets.index((20, 30))], CG, "{:.3f}")
put("cgProbeLast", acc[buckets.index((80, 120))], CG, "{:.3f}")
put("cgLayer", bl, CG, "{:d}")
ent = S.load("attn_8L256_s0_entangle.json")
eb = [tuple(b) for b in ent["buckets"]]
put("ourProbeFirst", ent["curves"]["linear"][eb.index((20, 30))], "results/attn_8L256_s0_entangle.json", "{:.3f}")
put("ourProbeLast", ent["curves"]["linear"][eb.index((80, 120))], "results/attn_8L256_s0_entangle.json", "{:.3f}")

ob = S.load("attn_8L256_s0_belief.json")["clustered"]
put("ourBelOwn", ob["own"]["mean"], "results/attn_8L256_s0_belief.json")
put("ourBelMis", ob["mis"]["mean"], "results/attn_8L256_s0_belief.json")
put("ourBelRand", ob["rnd"]["mean"], "results/attn_8L256_s0_belief.json")
put("ourBelRatio", ob["own"]["mean"] / ob["mis"]["mean"], "results/attn_8L256_s0_belief.json", "{:.1f}")
oa = S.load("attn_8L256_s0_auc.json")["per_bucket"]
agg = np.mean([v["auc_aggregate"] for v in oa.values()])
act = np.mean([v["auc_action_relevant"] for v in oa.values()])
put("ourAucAgg", agg, "results/attn_8L256_s0_auc.json")
put("ourAucAct", act, "results/attn_8L256_s0_auc.json")
put("ourAucDiff", sgn(act - agg), "results/attn_8L256_s0_auc.json")

for dist, T in (("human", "H"), ("random", "R")):
    b = S.load(f"chessgpt2_belief_{dist}.json")
    put(f"cgIll{T}", b["illegal_rate"], CG, "{:.4f}")
    put(f"cgNIll{T}", b["n_illegal"], CG, "{:,d}")
    for k, tag in (("own", "Own"), ("mismatched", "Mis"), ("random", "Rand"),
                   ("legal_fraction_reference", "Ref")):
        put(f"cgBel{tag}{T}", b[k]["mean"], CG, "{:.3f}")
        put(f"cgBel{tag}{T}Lo", b[k]["ci_lo"], CG, "{:.3f}")
        put(f"cgBel{tag}{T}Hi", b[k]["ci_hi"], CG, "{:.3f}")
    put(f"cgBelRatio{T}", b["own"]["mean"] / b["mismatched"]["mean"], CG, "{:.1f}")
    put(f"cgAucAgg{T}", b["auc_whole_board"]["auc"], CG)
    put(f"cgAucAct{T}", b["auc_move_touched"]["auc"], CG)
    put(f"cgAucDiff{T}", sgn(b["auc_difference"]), CG)
    e = S.load(f"chessgpt2_edit_{dist}.json")
    put(f"cgEdit{T}", sgn(e["state"]["mean"]), CG)
    put(f"cgEditRand{T}", sgn(e["rand"]["mean"]), CG)
    put(f"cgEditDiff{T}", sgn(e["state_minus_random"]["mean"]), CG)
    put(f"cgEditDiff{T}Lo", sgn(e["state_minus_random"]["ci_lo"]), CG)
    put(f"cgEditDiff{T}Hi", sgn(e["state_minus_random"]["ci_hi"]), CG)
    put(f"cgEditGames{T}", e["n_games"], CG, "{:d}")
    put(f"cgFlip{T}", e["manipulation_flip"], CG, "{:.3f}")

# ======================================================== Act 2: mechanism
kill = S.repair_meta("correct_minus_wrong_target_entropy_matched")
loc = S.repair_meta("correct_minus_irrelevant_entropy_matched")
RM = "results/attn_*_repair.json"
for tag, m in (("Kill", kill), ("Loc", loc)):
    put(f"meta{tag}Pos", m["pos"], RM, "{:d}")
    put(f"meta{tag}N", m["n"], RM, "{:d}")
    put(f"meta{tag}Sig", m["sig"], RM, "{:d}")
    put(f"meta{tag}P", pval(m["p"]), RM)
    put(f"meta{tag}Mean", sgn(m["mean"]), RM)
    put(f"meta{tag}Min", sgn(m["min"]), RM)
    put(f"meta{tag}Max", sgn(m["max"]), RM)

rp = S.load("attn_8L256_s0_repair.json")
RP = "results/attn_8L256_s0_repair.json"
put("repN", rp["n"], RP, "{:d}")
put("repGames", rp["n_games"], RP, "{:d}")
for c, tag in (("correct", "Correct"), ("irrelevant", "Irr"),
               ("wrong_target", "Wrong"), ("random", "Rand"), ("null", "Null")):
    cc = rp["conditions"][c]
    put(f"dr{tag}", sgn(cc["dr_mean"]) if cc["dr_mean"] != 0 else "0.000", RP)
    put(f"nl{tag}", cc["now_legal"], RP, "{:.3f}")
    put(f"dent{tag}", sgn(cc["dentropy"]) if cc["dentropy"] != 0 else "0.000", RP)
    put(f"flip{tag}", cc["manipulation_flip"], RP, "{:.3f}")
put("rawCW", sgn(rp["correct_minus_wrong_target"]["mean"]), RP)
mcw = rp["correct_minus_wrong_target_entropy_matched"]
put("matchedCW", sgn(mcw["mean"]), RP)
put("matchedCWLo", sgn(mcw["ci_lo"]), RP)
put("matchedCWHi", sgn(mcw["ci_hi"]), RP)
put("inflation", rp["correct_minus_wrong_target"]["mean"] / mcw["mean"], RP, "{:.1f}")
put("noflat", sgn(rp["conditions"]["correct"]["dr_noflatten"]), RP)
put("noflatN", rp["conditions"]["correct"]["n_noflatten"], RP, "{:d}")
for mlt, tag in (("0.25", "Q"), ("1.0", "One"), ("2.0", "Two"), ("4.0", "Four")):
    put(f"sweep{tag}", sgn(rp["sweep"][mlt]["dr"]), RP)

iv = S.inversion_meta()
put("invN", iv["n"], RM, "{:d}")
put("invCount", iv["inv"], RM, "{:d}")
put("invRandBest", iv["rand_best"], RM, "{:d}")
put("invCorrBest", iv["corr_best"], RM, "{:d}")
put("invP", pval(iv["p"]), RM)
put("invGapMean", sgn(iv["gap_mean"]), RM)
put("invGapMin", sgn(iv["gap_min"]), RM)
put("invGapMax", sgn(iv["gap_max"]), RM)

ind = S.load("attn_8L256_s0_induced.json")
IN = "results/attn_8L256_s0_induced.json"
put("indRec", ind["recovered"]["mean"], IN)
put("indRecLo", ind["recovered"]["ci_lo"], IN)
put("indRecHi", ind["recovered"]["ci_hi"], IN)
put("indRand", ind["recovered_rand"]["mean"], IN)
put("indDecode", ind["repaired_squares"]["after"] if "repaired_squares" in ind
    else ind["repair_decode_ok"]["mean"], IN)
dec = S.load("attn_8L256_s0_decompose.json")
DC = "results/attn_8L256_s0_decompose.json"
put("natRepaired", dec["memory_plus_mixed"], DC)
put("indNatRatio", ind["recovered"]["mean"] / dec["memory_plus_mixed"], IN, "{:.1f}")
for k, tag in (("memory", "Mem"), ("mixed", "Mix"), ("policy", "Pol"),
               ("unresolved", "Unr")):
    put(f"dec{tag}", dec["shares"][k]["mean"], DC)

FB = "results/attn_8L256_s0_repair_fidelity*.json"
full = S.load("attn_8L256_s0_repair_fidelity.json")
put("fidOccBefore", full["occupied squares only"]["before"], FB)
put("fidOccAfter", full["occupied squares only"]["after"], FB)
put("fidTargetAll", full["repaired_squares"]["after"], FB)
put("fidTargetPerSq", S.load("attn_8L256_s0_repair_fidelity_per_square_64.json")["repaired_squares"]["after"], FB)
put("fidTargetTwo", S.load("attn_8L256_s0_repair_fidelity_sum_renorm_2.json")["repaired_squares"]["after"], FB)

nd = S.load("attn_8L256_s0_natdiv.json")
ND = "results/attn_8L256_s0_natdiv.json"
put("natDivMedian", nd["ndiv_illegal"]["median"], ND, "{:.0f}")

ref = S.load("attn_8L256_s0_refresh.json")
RF = "results/attn_8L256_s0_refresh.json"
for m, tag in (("noop", "Noop"), ("repair", "Repair"), ("stale", "Stale"),
               ("random", "Random")):
    put(f"refresh{tag}", ref["illegal"][m][-1], RF, "{:.3f}")

# ======================================================== appendix: capacity
EN = "results/attn_8L256_s0_entangle.json"
gaps = [a - b for a, b in zip(ent["curves"]["mlp"], ent["curves"]["linear"])]
put("capGapMin", sgn(min(gaps)), EN)
put("capGapMax", sgn(max(gaps)), EN)
put("capCtlDeep", ent["curves"]["mlp_controltask"][-1], EN)
put("capRandDeep", ent["curves"]["mlp_randomweights"][-1], EN)

pw = S.load("s5_power.json")
PW = "results/s5_power.json"
put("powMdePct", 100 * pw["mde_pooled_9"] / pw["memory_fraction"], PW, "{:.0f}")
put("powSeedRel", pw["relative_seed_spread"], PW, "{:.3f}")


pre = open(os.path.join(S.ROOT, "PREREGISTRATION.sha256")).read().split()[0]
put("preregHash", pre[:16], "PREREGISTRATION.sha256")


ac = S.load("adapter_checks.json")
AC = "results/adapter_checks.json"
for k, tag in (("vocab_openings_legal", "acOpenLegal"), ("vocab_openings_total", "acOpenTotal"),
               ("vocab_size", "acVocab"), ("align_sites", "acAlignSites"),
               ("align_problems", "acAlignProblems"), ("nano16_legal", "acNanoLegal"),
               ("nano16_total", "acNanoTotal"), ("nanoRandom_legal", "acRandLegal"),
               ("nanoRandom_total", "acRandTotal"), ("cache_sites", "acCacheSites"),
               ("cache_mismatches", "acCacheMismatch"), ("san_cases", "acSanCases"),
               ("san_correct", "acSanCorrect"), ("othello_legal", "acOthLegal"),
               ("othello_total", "acOthTotal")):
    put(tag, int(ac[k]), AC, "{:,d}")
_allnano = all(ac.get(f"{t}_legal") == ac.get(f"{t}_total") for t in ("nano6", "nano8", "nano16"))
put("acAllNano", "all three" if _allnano else "not all", AC)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    L = ["% AUTO-GENERATED by code/make_macros_ds.py. Do not edit by hand.",
         r"\makeatletter",
         r"\newcommand{\result}[1]{%",
         r"  \expandafter\ifx\csname result@#1\endcsname\relax",
         r"    \textbf{??}\PackageWarning{results}{undefined key: #1}%",
         r"  \else\csname result@#1\endcsname\fi}",
         r"\newcommand{\defresult}[2]{\expandafter\def\csname result@#1\endcsname{#2}}",
         r"\makeatother", ""]
    for k in sorted(M):
        L.append(f"\\defresult{{{k}}}{{{M[k]}}}   % {PROV[k]}")
    open(OUT, "w", encoding="utf-8").write("\n".join(L) + "\n")
    json.dump({k: {"value": M[k], "source": PROV[k]} for k in M},
              open(os.path.join(S.RES, "macro_provenance_ds.json"), "w"), indent=1)
    na = [k for k in M if M[k] == "n/a"]
    print(f"wrote {OUT}: {len(M)} keys, {len(na)} n/a {na}")


if __name__ == "__main__":
    main()
