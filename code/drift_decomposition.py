"""Separate the three things that changed between the published differential and
the current one.

The manuscript attributed the residual movement in the headline differential to
floating-point precision. That attribution was not established, because more than
precision changed between the two numbers: the classifier changed, and so did the
estimator, from sampled shares rescaled by the aggregate rate to direct counts
over every scored position. A comparison that moves three things cannot assign the
movement to one of them.

So the three are varied one at a time:

  A  published      old classifier, float16 on GPU, 900-failure shares x rate
  B  precision      old classifier, float32 on CPU, 900-failure shares x rate
  C  estimator      old classifier, float32 on CPU, full-set direct counts
  D  classifier     new classifier, float32 on CPU, full-set direct counts

B - A isolates precision, holding the classifier and the estimator fixed. C - B
isolates the estimator, holding the classifier and precision fixed. D - C isolates
the classifier. The three differences sum to D - A by construction, which is worth
checking rather than assuming.

One honesty note about B - A. Changing precision can change which move is top-1 at
a near-tie, so the realised set of 900 failures is not identical between A and B
even though the sampling rule is. That is the precision effect rather than a
confound: there is no way to hold the sampled set fixed while changing the
arithmetic that selects it. What is held fixed is the classifier, the sampling
rule and the estimator.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ds_stats as S  # noqa: E402

OLD_PARTITION = ["from_empty", "from_opponent", "to_own", "geometry",
                 "leaves_check", "other"]


def slope(xs, ys):
    return float(np.polyfit(np.log(xs), np.log(np.maximum(ys, 1e-12)), 1)[0])


def diff_from(rows, a, b):
    """Paired differential of two class rates across the ladder."""
    xs = [q["params"] for q in rows]
    return (slope(xs, [q[a] for q in rows]) - slope(xs, [q[b] for q in rows]))


def published():
    """A: the superseded artifacts exactly as the paper reported them."""
    rows = S.ladder_rows()
    if not rows:
        return None, 0
    return diff_from(rows, "leaves_check", "from_empty"), len(rows)


def precision_only():
    """B: old labels from the regenerated pass, same sampling rule and estimator."""
    rows = []
    for r in S.RUNGS:
        for sd in S.SEEDS:
            name = f"attn_{r}_s{sd}"
            p = os.path.join(S.RES, f"{name}_regen.json")
            nd = S.load(f"{name}_natdiv.json")
            if not (os.path.exists(p) and nd):
                continue
            recs = json.load(open(p))["records"]
            n = len(recs)
            if not n:
                continue
            ill = nd["illegal_rate"]
            row = {"params": S.PARAMS[r]}
            for c in OLD_PARTITION:
                row[c] = (sum(1 for q in recs if q["old_label"] == c) / n) * ill
            rows.append(row)
    if not rows:
        return None, 0
    return diff_from(rows, "leaves_check", "from_empty"), len(rows)


def estimator_only():
    """C: old labels, full-set direct counts."""
    full = S.load("full_classification.json")
    if not full:
        return None, 0
    rows = []
    for r in full["rows"]:
        row = {"params": S.PARAMS[r["rung"]]}
        for c in OLD_PARTITION:
            row[c] = r["old_counts"].get(c, 0) / r["n_positions"]
        rows.append(row)
    return diff_from(rows, "leaves_check", "from_empty"), len(rows)


def current():
    """D: new labels, full-set direct counts."""
    rows = S.ladder_rows_corrected()
    return diff_from(rows, "leaves_check", "from_empty"), len(rows)


def main() -> int:
    A, nA = published()
    B, nB = precision_only()
    C, nC = estimator_only()
    D, nD = current()
    if None in (A, B, C, D):
        print("one of the four configurations is unavailable")
        return 1

    out = {"A_published": A, "B_precision": B, "C_estimator": C,
           "D_classifier": D,
           "d_precision": B - A, "d_estimator": C - B, "d_classifier": D - C,
           "total": D - A}
    out["sum_check"] = abs((out["d_precision"] + out["d_estimator"]
                            + out["d_classifier"]) - out["total"])
    print("check-minus-empty differential, one change at a time\n")
    print(f"  A published   old classifier, fp16 GPU, sampled   {A:+.4f}")
    print(f"  B precision   old classifier, fp32 CPU, sampled   {B:+.4f}"
          f"   ({B - A:+.4f})")
    print(f"  C estimator   old classifier, fp32 CPU, full set  {C:+.4f}"
          f"   ({C - B:+.4f})")
    print(f"  D classifier  new classifier, fp32 CPU, full set  {D:+.4f}"
          f"   ({D - C:+.4f})")
    print(f"\n  total movement {out['total']:+.4f}; "
          f"parts sum to within {out['sum_check']:.2e}")
    biggest = max(("precision", abs(out["d_precision"])),
                  ("estimator", abs(out["d_estimator"])),
                  ("classifier", abs(out["d_classifier"])),
                  key=lambda kv: kv[1])
    out["largest_term"] = biggest[0]
    print(f"  largest single term: {biggest[0]} ({biggest[1]:.4f})")
    json.dump(out, open(os.path.join(S.RES, "drift_decomposition.json"), "w"),
              indent=1)
    print("\nwrote results/drift_decomposition.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
