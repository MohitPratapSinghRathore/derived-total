"""Exponents under the repaired taxonomy, and whether the headline survives.

The reviewer's requirement is that the scaling analysis use the corrected
classification rather than only disclosing that it was wrong, and that the paper
establish which results are unaffected instead of leaving that open. So every
class is refitted from the regenerated records, including the two geometry
sub-classes and castling, with the same rung-level bootstrap as the paper.

Absolute rates multiply a class's share of failures by the run's illegal-move
rate, which is unchanged: no recorded failure turned out to be legal, so the
denominator and the rate are the same objects they were before.

The comparison against the published exponents is reported for every class, not
only the ones that moved, because "unchanged" is a claim that needs a number
attached to it too.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ds_stats as S  # noqa: E402
from structure_regen import NEW_CLASSES  # noqa: E402

N_BOOT = 4000


def rows():
    out = []
    for r in S.RUNGS:
        for s in S.SEEDS:
            nm = f"attn_{r}_s{s}"
            p = os.path.join(S.RES, f"{nm}_regen.json")
            nd = S.load(f"{nm}_natdiv.json")
            if not (os.path.exists(p) and nd):
                continue
            recs = json.load(open(p))["records"]
            n = len(recs)
            ill = nd["illegal_rate"]
            rec = {"rung": r, "seed": s, "params": S.PARAMS[r],
                   "illegal_rate": ill, "n": n}
            for c in NEW_CLASSES:
                rec[c] = (sum(1 for q in recs if q["new_label"] == c) / n) * ill
            rec["policy_share"] = float(np.mean([q["is_policy"] for q in recs]))
            out.append(rec)
    return out


def endpoints(rs, key):
    """Mean rate at the smallest and largest rung, and the fold drop between."""
    first = float(np.mean([q[key] for q in rs if q["rung"] == S.RUNGS[0]]))
    last = float(np.mean([q[key] for q in rs if q["rung"] == S.RUNGS[-1]]))
    return {"first": first, "last": last,
            "fold": (first / last) if last > 0 else None}


def exponent(rs, key):
    def slope(sub):
        v = [max(q[key], 1e-12) for q in sub]
        x = np.log([q["params"] for q in sub])
        y = np.log(v)
        return float(np.polyfit(x, y, 1)[0]) if len(set(x)) > 1 else np.nan

    pt = slope(rs)
    bs = np.array([v for v in (slope(s) for s in S._rung_resamples(rs, n=N_BOOT))
                   if np.isfinite(v)])
    lo, hi = float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))
    return {"point": pt, "lo": lo, "hi": hi,
            "excludes_zero": bool(lo > 0 or hi < 0)}


def differential(rs, a, b):
    def d(sub):
        xa = np.log([q["params"] for q in sub])
        fa = np.polyfit(xa, np.log([max(q[a], 1e-12) for q in sub]), 1)[0]
        fb = np.polyfit(xa, np.log([max(q[b], 1e-12) for q in sub]), 1)[0]
        return float(fa - fb)

    pt = d(rs)
    bs = np.array([v for v in (d(s) for s in S._rung_resamples(rs, n=N_BOOT))
                   if np.isfinite(v)])
    lo, hi = float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))
    return {"point": pt, "lo": lo, "hi": hi,
            "excludes_zero": bool(lo > 0 or hi < 0)}


def main() -> int:
    rs = rows()
    if not rs:
        print("no regenerated records found")
        return 1
    out = {"n_runs": len(rs), "classes": {}}
    print(f"{len(rs)} runs\n")
    print(f"{'class':22s} {'exponent':>9s} {'95% CI':>20s}  zero?")
    for c in NEW_CLASSES:
        if all(q[c] == 0 for q in rs):
            print(f"{c:22s}    (never observed)")
            continue
        e = exponent(rs, c)
        e.update(endpoints(rs, c))
        out["classes"][c] = e
        print(f"{c:22s} {e['point']:+9.4f} "
              f"[{e['lo']:+.4f}, {e['hi']:+.4f}]  "
              f"{'excl' if e['excludes_zero'] else 'incl'}")

    # The aggregate and the correct-local row, recomputed from the same records
    # so that every row of the table comes from one consistent pass.
    for q in rs:
        q["all_illegal"] = q["illegal_rate"]
        q["corr_local"] = q["policy_share"] * q["illegal_rate"]
    for c in ("all_illegal", "corr_local"):
        e = exponent(rs, c)
        e.update(endpoints(rs, c))
        out["classes"][c] = e
        print(f"{c:22s} {e['point']:+9.4f} "
              f"[{e['lo']:+.4f}, {e['hi']:+.4f}]  "
              f"{'excl' if e['excludes_zero'] else 'incl'}")

    # Shares and depth strata, recomputed here so that no quantity the
    # manuscript prints still comes from the superseded pass. The records carry
    # the ply of every failure, so the depth bands need no separate artifact.
    import json as _j
    shares, depth = {}, {}
    for r in S.RUNGS:
        acc, early, late, n_e, n_l = {}, 0.0, 0.0, 0, 0
        runs = 0
        for sd in S.SEEDS:
            p = os.path.join(S.RES, f"attn_{r}_s{sd}_regen.json")
            if not os.path.exists(p):
                continue
            runs += 1
            recs = _j.load(open(p))["records"]
            n = len(recs)
            for c in NEW_CLASSES:
                acc[c] = acc.get(c, 0.0) + sum(
                    1 for q in recs if q["new_label"] == c) / n
            e = [q for q in recs if 20 <= q["ply"] < 40]
            l = [q for q in recs if q["ply"] >= 60]
            if e:
                early += sum(1 for q in e
                             if q["new_label"] == "leaves_check") / len(e)
                n_e += 1
            if l:
                late += sum(1 for q in l
                            if q["new_label"] == "leaves_check") / len(l)
                n_l += 1
        if not runs:
            continue
        shares[r] = {c: v / runs for c, v in acc.items()}
        depth[r] = {"early": (early / n_e) if n_e else None,
                    "late": (late / n_l) if n_l else None}
    out["shares_by_rung"] = shares
    out["depth_by_rung"] = depth
    out["depth_rises"] = sum(
        1 for r, d in depth.items()
        if d["early"] is not None and d["late"] is not None
        and d["late"] > d["early"])
    print("\nSHARES AND DEPTH, corrected pass")
    first, last = S.RUNGS[0], S.RUNGS[-1]
    for c in ("from_empty", "leaves_check", "geometry_impossible",
              "geometry_blocked"):
        print(f"  {c:22s} share {shares[first][c]:.3f} -> "
              f"{shares[last][c]:.3f}")
    print(f"  check share rises with depth in {out['depth_rises']} of "
          f"{len(depth)} rungs; largest rung "
          f"{depth[last]['early']:.3f} -> {depth[last]['late']:.3f}")

    # Class composition among failures where local belief was already correct.
    # These were the last quantities in the manuscript still produced by the
    # superseded classifier.
    pol = {}
    largest = 0
    n_models = 0
    for r in S.RUNGS:
        for sd in S.SEEDS:
            p = os.path.join(S.RES, f"attn_{r}_s{sd}_regen.json")
            if not os.path.exists(p):
                continue
            recs = [q for q in _j.load(open(p))["records"] if q["is_policy"]]
            if not recs:
                continue
            n_models += 1
            fr = {}
            for c in NEW_CLASSES:
                fr[c] = sum(1 for q in recs if q["new_label"] == c) / len(recs)
            for c, v in fr.items():
                pol[c] = pol.get(c, 0.0) + v
            geom = fr["geometry_impossible"] + fr["geometry_blocked"]
            if fr["leaves_check"] >= max(geom, *[fr[c] for c in NEW_CLASSES
                                                 if not c.startswith("geometry")
                                                 and c != "leaves_check"]):
                largest += 1
    if n_models:
        out["policy_conditioned"] = {
            c: v / n_models for c, v in pol.items()}
        out["policy_conditioned"]["geometry"] = (
            out["policy_conditioned"]["geometry_impossible"]
            + out["policy_conditioned"]["geometry_blocked"])
        out["policy_check_largest"] = largest
        out["policy_n_models"] = n_models
        pc = out["policy_conditioned"]
        print("\nWHERE LOCAL BELIEF IS CORRECT, corrected labels")
        print(f"  leaves_check {pc[chr(108)+chr(101)+chr(97)+chr(118)+chr(101)+chr(115)+chr(95)+chr(99)+chr(104)+chr(101)+chr(99)+chr(107)]:.3f}"
              f"  geometry {pc[chr(103)+chr(101)+chr(111)+chr(109)+chr(101)+chr(116)+chr(114)+chr(121)]:.3f}"
              f"  check largest in {largest} of {n_models}")

    print("\nHEADLINE DIFFERENTIALS, paired bootstrap")
    for a, b, lab in (("corr_local", "from_empty", "corr local - empty"),
                      ("leaves_check", "from_empty", "check - empty"),
                      ("leaves_check", "from_opponent", "check - opponent"),
                      ("leaves_check", "geometry_impossible",
                       "check - geom impossible")):
        if all(q[b] == 0 for q in rs):
            continue
        d = differential(rs, a, b)
        out.setdefault("differentials", {})[lab] = d
        print(f"  {lab:26s} {d['point']:+.4f} [{d['lo']:+.4f}, {d['hi']:+.4f}]  "
              f"{'excludes zero' if d['excludes_zero'] else 'includes zero'}")

    # Against the published values, for every class, so that "unchanged" carries
    # a number. The published geometry exponent has no single counterpart now, so
    # both sub-classes are listed against it.
    pub = S.load("exponents.json") or {}
    if pub:
        print("\nAGAINST THE PUBLISHED EXPONENTS")
        for c in NEW_CLASSES:
            if c not in out["classes"]:
                continue
            src = c
            if c.startswith("geometry"):
                src = "geometry"
            if c == "castling":
                continue
            p = pub.get(src, {}).get("point") if isinstance(pub.get(src), dict) \
                else None
            if p is None:
                continue
            delta = out["classes"][c]["point"] - p
            print(f"  {c:22s} published {p:+.4f}  now "
                  f"{out['classes'][c]['point']:+.4f}  delta {delta:+.4f}")
            out["classes"][c]["published"] = p
            out["classes"][c]["delta_vs_published"] = delta
    json.dump(out, open(os.path.join(S.RES, "regen_exponents.json"), "w"),
              indent=1)
    print("\nwrote results/regen_exponents.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
