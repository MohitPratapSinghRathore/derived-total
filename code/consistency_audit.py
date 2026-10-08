"""Adversarial consistency audit over the built manuscript.

The last two review rounds both found that the damage was in consistency rather
than in the new work: a corrected table beside uncorrected prose, a figure drawn
from superseded data, a simplex column overwriting the largest class. None of
those were errors of analysis. They were errors of the analysis losing its
connection to what shipped.

The existing QA gates catch undefined macros, stale PDFs and hand-typed decimals.
They cannot catch a number that is correctly plumbed and semantically wrong. So
this checks invariants that must hold between quantities, each of which would have
caught at least one defect found so far:

  identity     quantities that must be equal because they count the same thing
  arithmetic   sums and ratios that must agree with their parts
  interval     point estimates that must lie inside their own intervals
  bounds       shares in the unit interval, counts not exceeding their totals
  prose        every "A of B" in the text where A must not exceed B
  reference    every cross-reference resolves, and to the right kind of object

A failure here is not necessarily a bug in the paper; it is a place where two
numbers disagree and somebody has to look. That is the point.
"""
from __future__ import annotations

import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ds_stats as S  # noqa: E402

TEX = os.path.join(S.ROOT, "paper_scaling", "main.tex")
MAC = os.path.join(S.ROOT, "paper_scaling", "results_macros.tex")
PDF = os.path.join(S.ROOT, "build_scaling", "main.pdf")

FAILS: list[str] = []
CHECKS = 0


def ok(cond, label, detail=""):
    global CHECKS
    CHECKS += 1
    if not cond:
        FAILS.append(f"{label}" + (f"  [{detail}]" if detail else ""))
    tag = "ok  " if cond else "FAIL"
    print(f"  [{tag}] {label}" + (f"  {detail}" if detail and not cond else ""))


def num(x):
    """Macro values carry LaTeX minus signs and thousands separators."""
    if x is None:
        return None
    t = str(x).replace("$-$", "-").replace(",", "").replace("\\%", "")
    t = t.replace("$", "").strip()
    m = re.match(r"^[+-]?\d*\.?\d+", t)
    return float(m.group()) if m else None


def macros():
    out = {}
    for k, v in re.findall(r"\\defresult\{([^}]+)\}\{([^}]*)\}",
                           open(MAC, encoding="utf-8").read()):
        out[k] = v
    return out


def main() -> int:
    M = macros()
    tex = open(TEX, encoding="utf-8").read()
    rg = S.load("regen_exponents.json") or {}
    rc = S.load("regen_compare.json") or {}
    tt = S.load("training_table.json") or {}
    li = S.load("locality_index.json") or {}
    cp = S.load("compositional.json") or {}
    rows = S.ladder_rows_corrected()

    print("IDENTITY: counts of the same thing must agree")
    run_counts = {k: num(M.get(k)) for k in
                  ("nRuns", "rgRuns", "trnRuns", "rcRuns")
                  if M.get(k) is not None}
    ok(len(set(run_counts.values())) == 1,
       "every reported run count agrees", str(run_counts))
    ok(num(M.get("rcFailures")) == 900 * num(M.get("rgRuns")),
       "recorded failures equal 900 per run",
       f"{M.get('rcFailures')} vs 900x{M.get('rgRuns')}")
    ok(len(rows) == int(num(M.get("rgRuns"))),
       "corrected rows match the reported run count", len(rows))

    print("\nARITHMETIC: sums and ratios must agree with their parts")
    if rc:
        parts = (rc["flow"].get("geometry->geometry_impossible", 0)
                 + rc["flow"].get("geometry->geometry_blocked", 0)
                 + rc["flow"].get("to_own->castling", 0))
        ok(parts == rc["n_relabelled"],
           "relabelled total equals its named components",
           f"{parts} vs {rc['n_relabelled']}")
        flow_tot = sum(rc["flow"].values())
        ok(flow_tot == rc["n_failures"],
           "class-flow cells sum to the failure total",
           f"{flow_tot} vs {rc['n_failures']}")
    if rows:
        worst = max(abs(sum(q[c] for c in S.CLASSES_CORRECTED) - q["illegal_rate"])
                    for q in rows)
        ok(worst < 1e-9, "corrected partition sums to the illegal rate",
           f"max dev {worst:.2e}")
    if rg:
        bad = []
        for c, e in rg["classes"].items():
            f, l, fold = e.get("first"), e.get("last"), e.get("fold")
            if None in (f, l) or not fold:
                continue
            if abs(fold - f / l) > 1e-6:
                bad.append(c)
            # A negative exponent means the rate fell, so the fold drop exceeds 1.
            if e["point"] < 0 and fold < 1:
                bad.append(c + ":sign")
        ok(not bad, "fold drops agree with endpoints and exponent signs", bad)
    if li:
        five = ["from_empty", "from_opponent", "to_own", "geometry",
                "leaves_check"]
        tot = sum(li["classes"][c]["n"] for c in five if c in li["classes"])
        ok(num(M.get("locMoves")) == tot,
           "locality sample total counts the partition only",
           f"{M.get('locMoves')} vs {tot}")
        g = li["classes"]
        if all(k in g for k in ("geometry", "geometry_impossible",
                                "geometry_blocked")):
            ok(g["geometry_impossible"]["n"] + g["geometry_blocked"]["n"]
               == g["geometry"]["n"],
               "geometry sub-classes partition geometry")

    print("\nINTERVAL: point estimates must lie inside their own intervals")
    bad = []
    for c, e in (rg.get("classes") or {}).items():
        if not (e["lo"] <= e["point"] <= e["hi"]):
            bad.append(c)
    for k, d in (rg.get("differentials") or {}).items():
        if not (d["lo"] <= d["point"] <= d["hi"]):
            bad.append(k)
    ok(not bad, "every exponent and differential sits inside its interval", bad)
    # And the same for what the manuscript prints, which is rounded separately.
    pairs = [("rgExp", "rgExpLo", "rgExpHi",
              ["Empty", "Opp", "Own", "Castle", "GeomImp", "GeomBlk", "Check",
               "AllIll", "CorrLoc"]),
             ("rgDiff", "rgDiffLo", "rgDiffHi",
              ["CheckEmpty", "CheckOpp", "CheckGeomImp"])]
    bad = []
    for base, lo, hi, tags in pairs:
        for t in tags:
            p, a, b = num(M.get(base + t)), num(M.get(lo + t)), num(M.get(hi + t))
            if None in (p, a, b):
                continue
            if not (a <= p <= b):
                bad.append(f"{base}{t}={p} not in [{a},{b}]")
    ok(not bad, "printed intervals contain their printed point estimates", bad)

    print("\nBOUNDS: shares in the unit interval, counts within totals")
    bad = []
    for tag in ("cmpCheckLarge", "cmpCheckBn", "cmpCheckTn", "cmpCheckMax",
                "cmpGeomImpLarge", "cmpGeomBlkTn", "cmpPolicyLarge",
                "cmpPolicyBn", "cmpPolicyTn", "shareOtherMax"):
        v = num(M.get(tag))
        if v is not None and not (0.0 <= v <= 1.0):
            bad.append(f"{tag}={v}")
    ok(not bad, "projected and observed shares lie in [0,1]", bad)
    if cp:
        ok(cp["a_star"] >= cp["weighted_mean_exponent"] - 1e-9,
           "a* is at least the rate-weighted mean, as the argument requires",
           f"{cp['a_star']:.4f} vs {cp['weighted_mean_exponent']:.4f}")
        ok(cp["divergence_rate"] > 0,
           "divergence rate is positive, so the closure argument applies")
    if tt:
        ok(tt["n_final_is_best"] <= tt["n_runs"],
           "final-is-best count cannot exceed the run count")
        vf = [r["val_final"] for r in tt["rows"]]
        ok(num(M.get("trnValMin")) <= num(M.get("trnValMax")),
           "validation loss min does not exceed max")
        ok(abs(min(vf) - num(M.get("trnValMin"))) < 5e-4,
           "printed validation minimum matches the artifacts")

    print("\nDECOMPOSITION: exponents that must add, do")
    jv = S.load("joint_vs_conditional.json") or {}
    if jv:
        add = abs((jv["marginal_correct"]["point"] + jv["conditional"]["point"])
                  - jv["joint"]["point"])
        ok(add < 1e-9,
           "marginal and conditional exponents sum to the joint exponent",
           f"error {add:.2e}")
        if rg:
            ok(abs(jv["joint"]["point"]
                   - rg["classes"]["corr_local"]["point"]) < 5e-4,
               "the decomposition's joint matches the Table 1 row",
               f"{jv['joint']['point']:.4f} vs "
               f"{rg['classes']['corr_local']['point']:.4f}")
    ag = S.load("aggregate_gate_check.json") or {}
    if ag:
        ok(ag["total_disagreements"] == 0,
           "the two legality gates agree on every scored position",
           ag["total_disagreements"])

    print("\nPROSE: every 'A of B' must have A no greater than B")
    # Resolve macros to their values first, then look for the pattern.
    prose = tex
    for k, v in M.items():
        prose = prose.replace("\\result{" + k + "}", str(v))
    bad = []
    for a, b in re.findall(r"([\d,]+(?:\.\d+)?)\s+of\s+([\d,]+(?:\.\d+)?)",
                           prose):
        x, y = num(a), num(b)
        if x is not None and y is not None and x > y:
            bad.append(f"{a} of {b}")
    ok(not bad, "no 'A of B' with A greater than B", bad)

    print("\nREFERENCE: cross-references resolve to a real label")
    labels = set(re.findall(r"\\label\{([^}]+)\}", tex))
    refs = set(re.findall(r"\\ref\{([^}]+)\}", tex))
    ok(not (refs - labels), "every \\ref has a matching \\label",
       sorted(refs - labels))
    unref = sorted(labels - refs)
    # Informational only: an unreferenced section label is untidy, not wrong.
    print(f"  [info] labels defined and never referenced: {unref}")

    print("\nSOURCE: analyses that feed the manuscript read the corrected pass")
    for f in ("robustness_checks", "compositional", "admissibility",
              "curvature", "curvature_decomp", "closure_check", "figures_ds"):
        p = os.path.join(HERE, f + ".py")
        if not os.path.exists(p):
            continue
        src = open(p, encoding="utf-8").read()
        uses_old = re.search(r"S\.ladder_rows\(\)", src) is not None
        ok(not uses_old, f"{f} does not read the superseded ladder rows")

    print(f"\n{CHECKS} checks, {len(FAILS)} failed")
    for f in FAILS:
        print("  FAILED:", f)
    return 1 if FAILS else 0


if __name__ == "__main__":
    raise SystemExit(main())
