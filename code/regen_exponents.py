"""Exponents under the repaired taxonomy, from the full-set classification.

Every failure at every scored position is classified, so each class rate here is a
direct count over the evaluation set. An earlier version of this analysis used the
first 900 failures per model and rescaled their shares by the aggregate rate; that
made each class rate an estimate resting on the first 900 failures being
representative of all of them, and corpus order is not random. The counts replace
the estimate, so the assumption is gone rather than documented.

Every class is refitted with the same rung-level bootstrap as the rest of the
paper, including the two geometry sub-classes and castling. The shares, depth
bands and correct-decoding composition are computed from the same counts, so no
quantity the manuscript prints is produced by a different pass.
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
CLASSES = list(S.CLASSES_CORRECTED)
# The classes a verdict can reach from the move's own two squares. Castling and
# blocked paths are excluded: both need further squares.
LOCAL = ["from_empty", "from_opponent", "to_own", "geometry_impossible"]


def rows():
    rs = S.ladder_rows_corrected()
    for q in rs:
        q["all_illegal"] = q["illegal_rate"]
        q["corr_local"] = q["correct_local_belief"]
        q["local_pooled"] = sum(q[c] for c in LOCAL)
    return rs


def endpoints(rs, key):
    first = float(np.mean([q[key] for q in rs if q["rung"] == S.RUNGS[0]]))
    last = float(np.mean([q[key] for q in rs if q["rung"] == S.RUNGS[-1]]))
    return {"first": first, "last": last,
            "fold": (first / last) if last > 0 else None}


def _slope(sub, key):
    x = np.log([q["params"] for q in sub])
    y = np.log([max(q[key], 1e-12) for q in sub])
    return float(np.polyfit(x, y, 1)[0]) if len(set(x)) > 1 else np.nan


def exponent(rs, key):
    pt = _slope(rs, key)
    bs = np.array([v for v in (_slope(s, key)
                               for s in S._rung_resamples(rs, n=N_BOOT))
                   if np.isfinite(v)])
    lo, hi = float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))
    return {"point": pt, "lo": lo, "hi": hi,
            "excludes_zero": bool(lo > 0 or hi < 0)}


def differential(rs, a, b):
    def d(sub):
        return _slope(sub, a) - _slope(sub, b)

    pt = d(rs)
    bs = np.array([v for v in (d(s) for s in S._rung_resamples(rs, n=N_BOOT))
                   if np.isfinite(v)])
    lo, hi = float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))
    return {"point": pt, "lo": lo, "hi": hi,
            "excludes_zero": bool(lo > 0 or hi < 0)}


def main() -> int:
    rs = rows()
    out = {"n_runs": len(rs), "classes": {},
           "n_failures": int(sum(q["n"] for q in rs)),
           "n_positions": int(sum(q["n_positions"] for q in rs)),
           "source": "full_classification.json, every failure classified"}
    print(f"{len(rs)} runs, {out['n_failures']:,} failures over "
          f"{out['n_positions']:,} positions\n")
    print(f"{'class':22s} {'exponent':>9s} {'95% CI':>20s}  zero?")
    for c in CLASSES + ["local_pooled", "all_illegal", "corr_local"]:
        if all(q[c] == 0 for q in rs):
            print(f"{c:22s}    (never observed)")
            continue
        e = exponent(rs, c)
        e.update(endpoints(rs, c))
        out["classes"][c] = e
        print(f"{c:22s} {e['point']:+9.4f} "
              f"[{e['lo']:+.4f}, {e['hi']:+.4f}]  "
              f"{'excl' if e['excludes_zero'] else 'incl'}")

    shares, depth, pol = {}, {}, {}
    for r in S.RUNGS:
        sub = [q for q in rs if q["rung"] == r]
        if not sub:
            continue
        shares[r] = {c: float(np.mean([q["share_" + c] for q in sub]))
                     for c in CLASSES}
        depth[r] = {
            "early": float(np.mean([q["lc_early"] for q in sub
                                    if q["lc_early"] is not None])),
            "late": float(np.mean([q["lc_late"] for q in sub
                                   if q["lc_late"] is not None]))}
        pol[r] = {c: float(np.mean([q["pol_" + c] for q in sub
                                    if q["pol_" + c] is not None]))
                  for c in CLASSES}
    out["shares_by_rung"] = shares
    out["depth_by_rung"] = depth
    out["depth_rises"] = sum(1 for d in depth.values() if d["late"] > d["early"])

    allpol = {c: float(np.mean([q["pol_" + c] for q in rs
                                if q["pol_" + c] is not None]))
              for c in CLASSES}
    allpol["geometry"] = (allpol["geometry_impossible"]
                          + allpol["geometry_blocked"])
    out["policy_conditioned"] = allpol
    largest = 0
    for q in rs:
        geom = q["pol_geometry_impossible"] + q["pol_geometry_blocked"]
        others = [q["pol_" + c] for c in CLASSES
                  if not c.startswith("geometry") and c != "leaves_check"]
        if q["pol_leaves_check"] >= max([geom] + others):
            largest += 1
    out["policy_check_largest"] = largest
    out["policy_n_models"] = len(rs)

    first, last = S.RUNGS[0], S.RUNGS[-1]
    print("\nSHARES AND DEPTH, full-set counts")
    for c in ("from_empty", "leaves_check", "geometry_impossible",
              "geometry_blocked"):
        print(f"  {c:22s} share {shares[first][c]:.3f} -> {shares[last][c]:.3f}")
    print(f"  check share rises with depth in {out['depth_rises']} of "
          f"{len(depth)} rungs; largest rung "
          f"{depth[last]['early']:.3f} -> {depth[last]['late']:.3f}")
    print(f"  where local belief is correct: check {allpol['leaves_check']:.3f},"
          f" geometry {allpol['geometry']:.3f},"
          f" check largest in {largest} of {len(rs)}")

    print("\nHEADLINE DIFFERENTIALS, paired bootstrap")
    for a, b, lab in (("leaves_check", "local_pooled", "check - local pooled"),
                      ("leaves_check", "from_empty", "check - empty"),
                      ("leaves_check", "from_opponent", "check - opponent"),
                      ("leaves_check", "geometry_impossible",
                       "check - geom impossible"),
                      ("corr_local", "from_empty", "corr local - empty")):
        d = differential(rs, a, b)
        out.setdefault("differentials", {})[lab] = d
        print(f"  {lab:26s} {d['point']:+.4f} [{d['lo']:+.4f}, {d['hi']:+.4f}]  "
              f"{'excludes zero' if d['excludes_zero'] else 'includes zero'}")

    json.dump(out, open(os.path.join(S.RES, "regen_exponents.json"), "w"),
              indent=1)
    print("\nwrote results/regen_exponents.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
