"""Single source of truth for the differential-scaling paper's statistics.

Macros, tables and figures all import from here, so a number in the prose, the
same number in a table, and the point it plots cannot disagree. Paper 1 learned
this the hard way; ladder_stats.py exists for the same reason.

Conventions, applied everywhere:

  absolute rate   a failure category's share of illegal top-1 moves multiplied by
                  the illegal-move rate, so every quantity is a fraction of ALL
                  scored positions at ply >= 20. Never a share. Shares of a
                  failure base that itself halves across the ladder can rise while
                  the absolute rate falls, which is how an earlier draft of this
                  argument came to imply that models got worse at check.

  exponent        slope of log absolute rate on log parameter count, fitted to all
                  runs. Its interval resamples RUNGS with replacement, because the
                  rung is the unit of variation and seeds within a rung are not
                  independent draws of the architecture.

  differential    a paired bootstrap: both exponents are refitted on the SAME
                  resample, so the interval is on the difference itself rather
                  than on two independent intervals.
"""
import os, json
from math import comb
import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
RES = os.path.join(ROOT, "results")

RUNGS = ["6L192", "8L256", "12L256", "8L384", "12L384", "12L512"]
SEEDS = [0, 1, 2]
PARAMS = {"6L192": 3_440_064, "8L256": 7_345_920, "12L256": 10_504_960,
          "8L384": 15_737_472, "12L384": 22_835_328, "12L512": 39_884_288}
CLASSES = ["from_empty", "from_opponent", "to_own", "geometry", "leaves_check"]
N_BOOT = 4000


def load(name):
    p = os.path.join(RES, name)
    return json.load(open(p)) if os.path.exists(p) else None


# ------------------------------------------------------------------ our ladder
def ladder_rows():
    rows = []
    for r in RUNGS:
        for s in SEEDS:
            st = load(f"attn_{r}_s{s}_structure.json")
            nd = load(f"attn_{r}_s{s}_natdiv.json")
            if not (st and nd):
                continue
            ill = nd["illegal_rate"]
            row = {"rung": r, "seed": s, "params": PARAMS[r],
                   "illegal_rate": ill,
                   "correct_local_belief": st["policy_share"] * ill,
                   "share_policy": st["policy_share"],
                   "share_other": st["overall"].get("other", {}).get("mean", 0.0)}
            for c in CLASSES:
                row[c] = st["overall"][c]["mean"] * ill
                row["share_" + c] = st["overall"][c]["mean"]
            lc_ok = st["overall"]["leaves_check"].get("share_when_belief_ok")
            row["lc_when_local_ok"] = lc_ok
            strata = st.get("strata", {})
            row["lc_early"] = strata.get("20-40", {}).get("leaves_check")
            row["lc_late"] = strata.get("60-120", {}).get("leaves_check")
            rows.append(row)
    return rows


def _slope(sub, key):
    x = np.log([q["params"] for q in sub])
    y = np.log([max(q[key], 1e-12) for q in sub])
    if len(set(x)) < 2:
        return np.nan
    return float(np.polyfit(x, y, 1)[0])


def _rung_resamples(rows, n=N_BOOT, seed=0):
    rng = np.random.default_rng(seed)
    by = {r: [q for q in rows if q["rung"] == r] for r in RUNGS}
    present = [r for r in RUNGS if by[r]]
    for _ in range(n):
        yield [q for r in rng.choice(present, len(present), replace=True)
               for q in by[r]]


def exponent(rows, key):
    pt = _slope(rows, key)
    bs = np.array([v for v in (_slope(sub, key) for sub in _rung_resamples(rows))
                   if np.isfinite(v)])
    return {"point": pt, "lo": float(np.percentile(bs, 2.5)),
            "hi": float(np.percentile(bs, 97.5))}


def differential(rows, a, b):
    """exponent(a) - exponent(b), paired on the same rung resample."""
    d = []
    for sub in _rung_resamples(rows):
        ea, eb = _slope(sub, a), _slope(sub, b)
        if np.isfinite(ea) and np.isfinite(eb):
            d.append(ea - eb)
    d = np.array(d)
    return {"point": _slope(rows, a) - _slope(rows, b),
            "lo": float(np.percentile(d, 2.5)), "hi": float(np.percentile(d, 97.5)),
            "frac_pos": float((d > 0).mean())}


def rung_means(rows, key):
    return [float(np.mean([q[key] for q in rows if q["rung"] == r]))
            for r in RUNGS if any(q["rung"] == r for q in rows)]


def monotone_decreasing(rows, key):
    m = rung_means(rows, key)
    return all(m[i] >= m[i + 1] for i in range(len(m) - 1))


# ------------------------------------------------------------ repair meta-test
def repair_meta(key):
    vals, sig = [], 0
    for r in RUNGS:
        for s in SEEDS:
            d = load(f"attn_{r}_s{s}_repair.json")
            if d and key in d:
                vals.append(d[key]["mean"])
                sig += int(d[key]["ci_lo"] > 0)
    n = len(vals)
    pos = sum(1 for v in vals if v > 0)
    k = max(pos, n - pos)
    p = min(1.0, 2 * sum(comb(n, i) for i in range(k, n + 1)) / 2 ** n) if n else 1.0
    return {"n": n, "pos": pos, "sig": sig, "p": p,
            "mean": float(np.mean(vals)) if vals else float("nan"),
            "min": float(min(vals)) if vals else float("nan"),
            "max": float(max(vals)) if vals else float("nan")}


# ------------------------------------------------------------ external ladder
EXT_LADDER = ["lichess_6L", "lichess_8L", "lichess_16L"]


def external_rows():
    out = []
    for lab in EXT_LADDER:
        d = load(f"chessgpt_structure_{lab}.json")
        if d:
            out.append(d)
    return out


def external_exponents(rows, keys=("unreachable", "leaves_check", "to_own"),
                       n=N_BOOT, seed=0):
    """Three checkpoints admit no fit uncertainty; this propagates only the
    sampling noise of each checkpoint's class shares, and must be reported as
    such rather than as a confidence statement about the slope."""
    rng = np.random.default_rng(seed)
    x = np.log([r["params"] for r in rows])
    res = {}
    draws = {k: [] for k in keys}
    for _ in range(n):
        for k in keys:
            y = []
            for r in rows:
                s = r["shares"][k]
                sd = max((s["ci_hi"] - s["ci_lo"]) / 3.92, 1e-6)
                y.append(np.log(max(rng.normal(s["mean"], sd), 1e-6)
                                * r["illegal_rate"]))
            draws[k].append(np.polyfit(x, y, 1)[0])
    for k in keys:
        y = np.log([max(r["absolute"][k], 1e-12) for r in rows])
        arr = np.array(draws[k])
        res[k] = {"point": float(np.polyfit(x, y, 1)[0]),
                  "lo": float(np.percentile(arr, 2.5)),
                  "hi": float(np.percentile(arr, 97.5)), "draws": arr}
    d = res["leaves_check"]["draws"] - res["unreachable"]["draws"]
    res["differential"] = {
        "point": res["leaves_check"]["point"] - res["unreachable"]["point"],
        "lo": float(np.percentile(d, 2.5)), "hi": float(np.percentile(d, 97.5)),
        "frac_pos": float((d > 0).mean())}
    return res


if __name__ == "__main__":
    rows = ladder_rows()
    print(f"{len(rows)} runs")
    for k in CLASSES + ["correct_local_belief", "illegal_rate"]:
        e = exponent(rows, k)
        print(f"  {k:22s} {e['point']:+.3f} [{e['lo']:+.3f}, {e['hi']:+.3f}]"
              f"  monotone={monotone_decreasing(rows, k)}")
    for a, b in (("leaves_check", "from_empty"),
                 ("correct_local_belief", "from_empty")):
        d = differential(rows, a, b)
        print(f"  diff {a} - {b}: {d['point']:+.3f} [{d['lo']:+.3f}, "
              f"{d['hi']:+.3f}] pos={d['frac_pos']:.3f}")
    ext = external_exponents(external_rows())
    print(f"  external diff {ext['differential']['point']:+.3f} "
          f"[{ext['differential']['lo']:+.3f}, {ext['differential']['hi']:+.3f}]")


def inversion_meta():
    """Does illegal-move rate rank the edits backwards, in every run?

    The full inversion is: the norm-matched random direction leaves the model
    with MORE legal top-1 moves than correct repair, while correct repair produces
    the larger shift in forced-choice preference. Counted per run, with an exact
    two-sided sign test across runs.
    """
    conds = ("correct", "irrelevant", "wrong_target", "random")
    n = inv = rand_best = corr_best = 0
    gaps = []
    for r in RUNGS:
        for s in SEEDS:
            d = load(f"attn_{r}_s{s}_repair.json")
            if not d:
                continue
            c = d["conditions"]
            n += 1
            nl = {k: c[k]["now_legal"] for k in conds}
            dr = {k: c[k]["dr_mean"] for k in conds}
            rand_best += int(max(nl, key=nl.get) == "random")
            corr_best += int(max(dr, key=dr.get) == "correct")
            inv += int(nl["random"] > nl["correct"] and dr["correct"] > dr["random"])
            gaps.append(nl["random"] - nl["correct"])
    k = max(inv, n - inv)
    p = min(1.0, 2 * sum(comb(n, i) for i in range(k, n + 1)) / 2 ** n)
    return {"n": n, "inv": inv, "rand_best": rand_best, "corr_best": corr_best,
            "p": p, "gap_mean": float(np.mean(gaps)),
            "gap_min": float(min(gaps)), "gap_max": float(max(gaps))}
