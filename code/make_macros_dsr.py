"""Generate paper_dsr/results_macros.tex.

Same discipline as the other two manuscripts: no number reaches the prose except
through a \\result{} macro written here and tagged with the results file it came
from. This paper reuses the measurements the scaling paper produced rather than
recomputing them, so the two manuscripts cannot disagree about a shared figure.
"""
import os, json
import numpy as np
import ds_stats as S

OUT_DIR = os.path.join(S.ROOT, "paper_dsr")
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
    """Negative numbers need a real minus. Never used inside math mode: the
    $-$ it emits would close the surrounding $...$ and silently degrade."""
    return f"{v:+.3f}".replace("-", "$-$", 1) if v < 0 else f"{v:+.3f}"


def mag(n):
    if n >= 1e12:
        return f"{n / 1e12:.0f}T"
    if n >= 1e9:
        return f"{n / 1e9:.1f}B"
    return f"{n / 1e6:.0f}M"


# ------------------------------------------------------------ the ladder
rows = S.ladder_rows()
LAD = "results/attn_*_{structure,natdiv}.json via ds_stats"
put("nRuns", len(rows), LAD, "{:d}")
put("nRungs", len(S.RUNGS), LAD, "{:d}")
put("nSeeds", len(S.SEEDS), LAD, "{:d}")
put("paramsSmall", f"{S.PARAMS[S.RUNGS[0]] / 1e6:.1f}M", LAD)
put("paramsLarge", f"{S.PARAMS[S.RUNGS[-1]] / 1e6:.1f}M", LAD)
put("spanLadder", S.PARAMS[S.RUNGS[-1]] / S.PARAMS[S.RUNGS[0]], LAD, "{:.1f}")

for tag, key in (("FromEmpty", "from_empty"), ("Geometry", "geometry"),
                 ("LeavesCheck", "leaves_check"), ("ToOwn", "to_own"),
                 ("FromOpp", "from_opponent"), ("Illegal", "illegal_rate"),
                 ("CorrLocal", "correct_local_belief")):
    e = S.exponent(rows, key)
    put(f"exp{tag}", sgn(e["point"]), LAD)
    put(f"exp{tag}Lo", sgn(e["lo"]), LAD)
    put(f"exp{tag}Hi", sgn(e["hi"]), LAD)

d = S.differential(rows, "leaves_check", "from_empty")
put("diffProbeFree", sgn(d["point"]), LAD)
put("diffProbeFreeLo", sgn(d["lo"]), LAD)
put("diffProbeFreeHi", sgn(d["hi"]), LAD)

# ------------------------------------------------- the closure diagnostic
cp = S.load("compositional.json")
CP = "results/compositional.json"
put("cmpAstar", sgn(cp["a_star"]), CP)
put("cmpBtotal", sgn(cp["b_total"]), CP)
put("cmpWeighted", sgn(cp["weighted_mean_exponent"]), CP)
put("cmpDiverge", sgn(cp["divergence_rate"]), CP)
put("cmpBreakdown", mag(cp["breakdown_scale"]), CP)
for tag, lab in (("ours_large", "Large"), ("proj_1e9", "Bn"), ("proj_1e12", "Tn")):
    put(f"cmpOver{lab}", cp["overspend"][tag]["ratio"], CP, "{:.2f}")
put("cmpOverBnLo", cp["overspend_1e9_lo"], CP, "{:.2f}")
put("cmpOverBnHi", cp["overspend_1e9_hi"], CP, "{:.2f}")
put("cmpOverBnFrac", 100 * cp["overspend_1e9_frac_above_one"], CP, "{:.0f}")
put("cmpRmseSimplex", cp["rmse_simplex"], CP, "{:.5f}")
put("cmpRmseIndep", cp["rmse_independent"], CP, "{:.5f}")
put("cmpRmseGain", 100 * (1 - cp["rmse_simplex"] / cp["rmse_independent"]), CP,
    "{:.1f}")
for tag, lab in (("ours_large", "Large"), ("proj_1e9", "Bn"), ("proj_1e12", "Tn")):
    for k, nm in (("leaves_check", "Check"), ("from_empty", "Empty"),
                  ("geometry", "Geom")):
        put(f"cmp{nm}{lab}", cp["composition"][tag][k], CP, "{:.3f}")
put("cmpCheckMax", cp["check_max_share"], CP, "{:.3f}")
put("cmpPolicySlope", sgn(cp["policy_logit_slope"]), CP)
for tag, lab in (("ours_large", "Large"), ("proj_1e9", "Bn"), ("proj_1e12", "Tn")):
    put(f"cmpPolicy{lab}", cp["policy_at"][tag], CP, "{:.3f}")
put("cmpPolicyObsLo", cp["policy_obs_min"], CP, "{:.3f}")
put("cmpPolicyObsHi", cp["policy_obs_max"], CP, "{:.3f}")
put("cmpExtDiverge", sgn(cp["ext_divergence_rate"]), CP)
put("cmpExtAstar", sgn(cp["ext_a_star"]), CP)
put("cmpExtB", sgn(cp["ext_b"]), CP)
put("cmpExtOverBn", cp["ext_overspend_1e9"], CP, "{:.2f}")
put("cmpExtBreakdown", mag(cp["ext_breakdown"]), CP)
put("cmpExtNclass", len(cp["ext_classes"]), CP, "{:d}")
put("cmpCompTV", cp["covariate_tv"]["competence"], CP, "{:.3f}")
put("cmpSizeTV", cp["covariate_tv"]["size"], CP, "{:.3f}")
put("cmpBaseTV", cp["covariate_tv"]["baseline"], CP, "{:.3f}")
put("admSizeMargin", cp["covariate_tv"]["baseline"] - cp["covariate_tv"]["size"],
    CP, "{:.3f}")

# ------------------------------------------- admissibility (corrected framing)
ad = S.load("admissibility.json")
AD = "results/admissibility.json"
put("admBfitted", sgn(ad["b_fitted"]), AD)
put("admAstar", sgn(ad["a_star"]), AD)
put("admAmin", sgn(ad["a_min"]), AD)
for tag, lab in (("small", "Small"), ("large", "Large"), ("bn", "Bn"), ("tn", "Tn")):
    put(f"admEff{lab}", sgn(ad["effective_exponent"][tag]), AD)
put("admDriftRange", ad["eff_drift_in_range"], AD, "{:.3f}")
put("admDriftBn", ad["eff_drift_to_bn"], AD, "{:.3f}")
put("admDriftPct", 100 * ad["drift_frac_of_span"], AD, "{:.0f}")
put("admSpan", ad["exponent_span"], AD, "{:.3f}")
for tag, lab in (("large", "Large"), ("bn", "Bn"), ("tn", "Tn")):
    put(f"admOver{lab}", ad["overspend"][tag], AD, "{:.2f}")
put("admBreakdown", mag(ad["breakdown"]), AD)
put("admBreakdownLo", mag(ad["breakdown_lo"]), AD)
put("admBreakdownHi", mag(ad["breakdown_hi"]), AD)
acc = ad["accuracy"]
put("admShareSimplex", acc["share_rmse_simplex"], AD, "{:.5f}")
put("admShareSum", acc["share_rmse_sumparts"], AD, "{:.5f}")
put("admRateSimplex", acc["rate_rmse_simplex"], AD, "{:.5f}")
put("admRateSum", acc["rate_rmse_sumparts"], AD, "{:.5f}")
put("admMaxShareGap", acc["max_share_gap"], AD, "{:.4f}")

# how far beyond the fitted range the companion forecast reached
_span = S.PARAMS[S.RUNGS[-1]] / S.PARAMS[S.RUNGS[0]]
put("admLadderWide", _span, LAD, "{:.0f}")
put("admForecastBeyond", 1e9 / S.PARAMS[S.RUNGS[-1]], LAD, "{:.0f}")

# ----------------------------------------------------- in-range curvature
cv = S.load("curvature.json")
CV = "results/curvature.json"
put("curvQ", f"{cv['q_ols']:+.4f}", CV)
put("curvQLo", f"{cv['q_boot_lo']:+.4f}", CV)
put("curvQHi", f"{cv['q_boot_hi']:+.4f}", CV)
put("curvQFracNeg", 100 * cv["q_boot_frac_neg"], CV, "{:.1f}")
put("curvT", cv["q_t"], CV, "{:.2f}")
put("curvDof", cv["q_dof"], CV, "{:d}")
put("curvRmseLin", cv["rmse_linear"], CV, "{:.4f}")
put("curvRmseQuad", cv["rmse_quadratic"], CV, "{:.4f}")
put("curvQImplied", f"{cv['implied_q']:+.4f}", CV)

# ----------------------------------------------------- literature audit
la = S.load("literature_audit.json")
LA = "results/literature_audit.json"
_lq = la["sources"][0]
put("audRows", _lq["n_rows"], LA, "{:d}")
put("audClosed", _lq["n_closed"], LA, "{:d}")
put("audWorst", _lq["max_abs_residual"], LA, "{:.2f}")
put("audSources", la["summary"]["sources_examined"], LA, "{:d}")
put("audWithAgg", la["summary"]["with_independent_aggregate"], LA, "{:d}")

# --------------------------------------------------------- external ladder
ext = S.external_rows()
EX = "results/chessgpt_structure_lichess_{6L,8L,16L}.json"
for i, r in enumerate(ext):
    t = ["Six", "Eight", "Sixteen"][i]
    put(f"kParams{t}", f"{r['params'] / 1e6:.1f}M", EX)
    put(f"kIll{t}", r["illegal_rate"], EX, "{:.4f}")
put("kSpan", ext[-1]["params"] / ext[0]["params"], EX, "{:.0f}")

# ------------------------------------------------------- the environment
env = S.load("environment.json")
EV = "results/environment.json"
put("envGpu", env["gpu_name"], EV)
put("envGpuMem", env["gpu_memory_gib"], EV, "{:.0f}")
put("envHostMem", env["host_memory_gb"], EV, "{:.0f}")
put("envOs", env["os"], EV)
for k, tag in (("python", "envPython"), ("torch", "envTorch"),
               ("numpy", "envNumpy"), ("scipy", "envScipy")):
    put(tag, env[k], EV)

# ----------------------------------------------------- adapter provenance
ac = S.load("adapter_checks.json")
AC = "results/adapter_checks.json"
for k, tag in (("align_sites", "acAlignSites"), ("align_problems", "acAlignProblems"),
               ("vocab_size", "acVocab"), ("othello_legal", "acOthLegal"),
               ("othello_total", "acOthTotal")):
    put(tag, int(ac[k]), AC, "{:,d}")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    L = ["% AUTO-GENERATED by code/make_macros_dsr.py. Do not edit by hand.",
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
              open(os.path.join(S.RES, "macro_provenance_dsr.json"), "w"), indent=1)
    na = [k for k in M if M[k] == "n/a"]
    print(f"wrote {OUT}: {len(M)} keys, {len(na)} n/a {na}")


if __name__ == "__main__":
    main()
