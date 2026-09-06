"""Absolute failure rates per category, and how each one scales.

An earlier framing reported the composition of failures -- leaves_check rising
from 0.307 to 0.448 of failures across the ladder -- and described scaling as
leaving global incoherence behind. That framing is wrong, or at least it invites
a reading the data does not support, because the failure base itself halves
across the ladder. A category can grow as a share while falling in absolute
terms, and here every category falls. A reader who multiplies through and finds
the prose claiming the model got worse at check will treat it as spin.

So everything here is an absolute rate: the fraction of ALL scored positions at
ply >= 20 at which the model plays an illegal move of that kind, which is the
category share multiplied by the illegal-move rate.

The claim survives the correction and is stronger stated this way. Local state
errors fall steeply with scale. Failures that occur DESPITE correct belief at the
two squares the move touches are close to flat. Two categories with different
scaling exponents is a differential scaling result, immune to the composition
objection, and it says something a share shift cannot: the component our
instrument cannot reach is the component scale does not fix.

Exponents are fitted as log rate against log parameters over all eighteen runs.
Intervals come from resampling the six rungs with replacement, since the rung is
the unit of variation and seeds within a rung are not independent draws of the
architecture. Six clusters is few, so the intervals are wide and are reported
rather than smoothed over.
"""
import os, json
import numpy as np

RES = os.path.join(os.path.dirname(__file__), "..", "results")
RUNGS = ["6L192", "8L256", "12L256", "8L384", "12L384", "12L512"]
SEEDS = [0, 1, 2]
PARAMS = {"6L192": 3_440_064, "8L256": 7_345_920, "12L256": 10_504_960,
          "8L384": 15_737_472, "12L384": 22_835_328, "12L512": 39_884_288}
CLS = ["from_empty", "from_opponent", "to_own", "geometry", "leaves_check"]


def load(name, suffix):
    p = os.path.join(RES, f"{name}{suffix}")
    return json.load(open(p)) if os.path.exists(p) else None


def gather():
    """Absolute rate per category per run: share of failures x failure rate."""
    rows = []
    for r in RUNGS:
        for s in SEEDS:
            n = f"attn_{r}_s{s}"
            st, nd = load(n, "_structure.json"), load(n, "_natdiv.json")
            if not (st and nd):
                continue
            ill = nd["illegal_rate"]
            rec = {"rung": r, "seed": s, "params": PARAMS[r],
                   "illegal_rate": ill}
            for c in CLS:
                rec[c] = st["overall"][c]["mean"] * ill
            # failures despite correct belief at both action squares
            rec["correct_local_belief"] = st["policy_share"] * ill
            rows.append(rec)
    return rows


def fit_exponent(rows, key, n_boot=2000, seed=0):
    """log rate ~ a + b log params. Bootstrap resamples RUNGS, not runs."""
    def do(sub):
        x = np.log([r["params"] for r in sub])
        y = np.log([max(r[key], 1e-9) for r in sub])
        if len(set(x)) < 2:
            return np.nan
        return float(np.polyfit(x, y, 1)[0])

    b = do(rows)
    by = {r: [x for x in rows if x["rung"] == r] for r in RUNGS}
    rng = np.random.default_rng(seed)
    bs = []
    for _ in range(n_boot):
        pick = rng.choice(RUNGS, len(RUNGS), replace=True)
        sub = [x for r in pick for x in by[r]]
        v = do(sub)
        if np.isfinite(v):
            bs.append(v)
    return b, float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)), bs


def main():
    rows = gather()
    print(f"absolute failure rates, {len(rows)} runs "
          f"(category share x illegal-move rate)\n")
    keys = ["illegal_rate"] + CLS + ["correct_local_belief"]
    print(f"{'rung':>8} {'params':>11} " +
          " ".join(f"{k[:13]:>14}" for k in keys))
    for r in RUNGS:
        sub = [x for x in rows if x["rung"] == r]
        if not sub:
            continue
        print(f"{r:>8} {PARAMS[r]:>11,} " +
              " ".join(f"{np.mean([x[k] for x in sub]):14.4f}" for k in keys))

    print(f"\n{'':>8} {'ratio 6L192 -> 12L512 (>1 means improved)':>52}")
    first = [x for x in rows if x["rung"] == "6L192"]
    last = [x for x in rows if x["rung"] == "12L512"]
    for k in keys:
        a, b = np.mean([x[k] for x in first]), np.mean([x[k] for x in last])
        print(f"{k:>26} {a:8.4f} -> {b:8.4f}   {a/max(b,1e-9):6.2f}x")

    print(f"\nscaling exponents, log rate against log parameters")
    print(f"(more negative means the category is fixed faster by scale)")
    print(f"{'category':>26} {'exponent':>10}  {'95% CI (rung-clustered)':>26}"
          f" {'monotone?':>10}")
    fits = {}
    for k in keys:
        b, lo, hi, bs = fit_exponent(rows, k)
        fits[k] = bs
        means = [np.mean([x[k] for x in rows if x["rung"] == r])
                 for r in RUNGS]
        mono = all(means[i] >= means[i + 1] for i in range(len(means) - 1))
        print(f"{k:>26} {b:10.4f}  [{lo:+.4f}, {hi:+.4f}] {str(mono):>10}")

    print(f"\ndifferential scaling: is local state fixed faster than "
          f"failure-despite-correct-local-belief?")
    d = np.array(fits["correct_local_belief"][:len(fits["from_empty"])]) - \
        np.array(fits["from_empty"][:len(fits["correct_local_belief"])])
    print(f"  exponent(correct_local_belief) - exponent(from_empty) = "
          f"{np.mean(d):+.4f} [{np.percentile(d,2.5):+.4f}, "
          f"{np.percentile(d,97.5):+.4f}]")
    print(f"  fraction of resamples with the difference > 0: "
          f"{float((d > 0).mean()):.4f}")
    print(f"  (positive means correct-local-belief failures scale away MORE "
          f"slowly)")

    out = {"rows": rows,
           "exponents": {k: {"point": float(np.mean(fits[k])),
                             "ci_lo": float(np.percentile(fits[k], 2.5)),
                             "ci_hi": float(np.percentile(fits[k], 97.5))}
                         for k in keys},
           "differential": {"mean": float(np.mean(d)),
                            "ci_lo": float(np.percentile(d, 2.5)),
                            "ci_hi": float(np.percentile(d, 97.5)),
                            "frac_positive": float((d > 0).mean())}}
    json.dump(out, open(os.path.join(RES, "scaling.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
