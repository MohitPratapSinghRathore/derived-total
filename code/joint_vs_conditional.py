"""Separate the two factors behind the flat residual rate.

The paper reports that failures occurring despite correctly decoded local state
have an exponent indistinguishable from zero, and reads that as the underlying
propensity being untouched by scale. That reading does not follow from the
quantity measured. What is measured is a joint event:

    P(illegal AND local state correctly decoded)
        = P(local state correctly decoded) x P(illegal | correctly decoded)

If the first factor rises with scale, as it plainly might, the joint rate can sit
flat while the conditional failure rate falls. A flat joint is then evidence of
two effects cancelling, not of one effect being absent, and the paper would be
claiming the opposite of what the data support.

So both factors are computed here. The marginal comes from the per-bucket
action-relevant decoding error already recorded in the AUC files; the joint is
the quantity the paper already reports; the conditional is their ratio. Each gets
its own exponent on the same ladder and the same bootstrap, so the three are
directly comparable.

The decomposition is exact by construction, which is worth stating: the point of
the exercise is not to estimate a new quantity but to show how an existing one
splits, and whether the paper's interpretation survives the split.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ds_stats as S  # noqa: E402

N_BOOT = 4000


def marginal_correct(name: str) -> float | None:
    """P(local state correctly decoded), pooled over the scored depth range.

    `action_relevant_error` is the fraction of scored positions whose
    action-relevant squares are not all decoded correctly, so one minus it is
    the marginal. Buckets are weighted by their own n, because they differ in
    size and an unweighted mean would let a thin deep bucket dominate.
    """
    d = S.load(f"{name}_auc.json")
    if not d or "per_bucket" not in d:
        return None
    num = den = 0.0
    for _b, r in d["per_bucket"].items():
        e, n = r.get("action_relevant_error"), r.get("n")
        if e is None or not n:
            continue
        num += (1.0 - e) * n
        den += n
    return (num / den) if den else None


def rows_with_decomposition():
    out = []
    for r in S.RUNGS:
        for s in S.SEEDS:
            name = f"attn_{r}_s{s}"
            st = S.load(f"{name}_structure.json")
            nd = S.load(f"{name}_natdiv.json")
            if not (st and nd):
                continue
            m = marginal_correct(name)
            if m is None or m <= 0:
                continue
            joint = st["policy_share"] * nd["illegal_rate"]
            out.append({"rung": r, "seed": s, "params": S.PARAMS[r],
                        "marginal_correct": m,
                        "joint": joint,
                        "conditional": joint / m,
                        "illegal_rate": nd["illegal_rate"]})
    return out


def exponent(rows, key):
    def slope(sub):
        x = np.log([q["params"] for q in sub])
        y = np.log([max(q[key], 1e-12) for q in sub])
        return float(np.polyfit(x, y, 1)[0]) if len(set(x)) > 1 else np.nan

    pt = slope(rows)
    bs = np.array([v for v in (slope(sub) for sub in S._rung_resamples(rows, n=N_BOOT))
                   if np.isfinite(v)])
    return {"point": pt, "lo": float(np.percentile(bs, 2.5)),
            "hi": float(np.percentile(bs, 97.5)),
            "excludes_zero": bool(np.percentile(bs, 2.5) > 0
                                  or np.percentile(bs, 97.5) < 0)}


def main() -> int:
    rows = rows_with_decomposition()
    if not rows:
        print("no runs carry both the structure and the AUC artifact")
        return 1
    out = {"n_runs": len(rows)}
    print(f"{len(rows)} runs carry both artifacts\n")
    print(f"{'rung':8s} {'P(correct)':>11s} {'joint':>9s} {'conditional':>12s}")
    for r in S.RUNGS:
        sub = [q for q in rows if q["rung"] == r]
        if not sub:
            continue
        print(f"{r:8s} {np.mean([q['marginal_correct'] for q in sub]):11.4f} "
              f"{np.mean([q['joint'] for q in sub]):9.4f} "
              f"{np.mean([q['conditional'] for q in sub]):12.4f}")
    print()
    for key, label in (("marginal_correct", "P(correct local decoding)"),
                       ("joint", "joint: illegal AND correct"),
                       ("conditional", "P(illegal | correct)")):
        e = exponent(rows, key)
        out[key] = e
        flag = "excludes zero" if e["excludes_zero"] else "includes zero"
        print(f"{label:28s} exponent {e['point']:+.4f} "
              f"[{e['lo']:+.4f}, {e['hi']:+.4f}]  {flag}")
    # The identity the split rests on, checked rather than assumed.
    err = max(abs(q["joint"] - q["marginal_correct"] * q["conditional"])
              for q in rows)
    out["identity_max_abs_error"] = err
    print(f"\ndecomposition identity holds to {err:.2e}")
    out["interpretation_supported"] = bool(
        not out["conditional"]["excludes_zero"])
    print("conditional exponent indistinguishable from zero:",
          out["interpretation_supported"])
    json.dump(out, open(os.path.join(S.RES, "joint_vs_conditional.json"), "w"),
              indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
