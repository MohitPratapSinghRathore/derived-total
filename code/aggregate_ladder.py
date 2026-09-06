"""Summarise the ladder replication: does each claim hold across rungs and seeds?

Three claims are worth replicating and they are not equally secure, so they are
reported separately rather than averaged into a single verdict.

  kill criterion   correct repair beats wrong-target, entropy-matched. This is
                   the causal claim and the pre-registered gate.
  locality         correct repair beats an identical edit budget spent on
                   divergent squares the move does not touch. This carries the
                   Balogh reconciliation.
  structure        the failure classes, and in particular whether leaves_check
                   rises with depth and whether anything lands in "other".

For each claim we report every rung and seed, then how many runs have an
interval excluding zero, then the consistency of sign across runs. A claim that
holds at large widths and fails at small ones would be capacity-dependent and
would have to be written that way, so the per-rung view matters.

Read the count of significant runs together with the sign test, never alone.
Locality has an interval excluding zero in only about three fifths of runs,
which on first look suggested it was capacity-dependent and shaky. It is not.
Every run is positive, because each run's matched contrast retains only about
300 positions and is underpowered by construction, not because the effect comes
and goes. Dichotomising runs into significant and not discards exactly the
information that settles it.

Seeds are aggregated by a fixed-effect mean across runs within a rung. That is
not a substitute for a proper hierarchical model over runs, and the per-run
intervals are clustered on games rather than pooled across seeds, so the
across-seed spread is reported as a range rather than as a confidence statement.
"""
import os, json, glob
from math import comb
import numpy as np

RES = os.path.join(os.path.dirname(__file__), "..", "results")
RUNGS = ["6L192", "8L256", "12L256", "8L384", "12L384", "12L512"]
SEEDS = [0, 1, 2]
PARAMS = {"6L192": 3_440_064, "8L256": 7_345_920, "12L256": 10_504_960,
          "8L384": 15_737_472, "12L384": 22_835_328, "12L512": 39_884_288}


def load(name, suffix):
    p = os.path.join(RES, f"{name}{suffix}")
    return json.load(open(p)) if os.path.exists(p) else None


def meta(key):
    """Consistency of sign across independent runs.

    Each run's entropy-matched contrast keeps only the positions where the two
    edits flattened comparably, roughly 300 of 600, so an individual run is
    underpowered by construction and its interval often includes zero. Counting
    how many runs are "significant" throws away the sign information, which is
    where the evidence actually lives when the same effect reappears in every
    architecture and seed.

    The sign test addresses sampling and seed variation. It does NOT address a
    systematic bias in how the edit is built, which would push every run the same
    way; the defence against that is the wrong-target and irrelevant conditions
    being matched to correct repair in magnitude and construction.
    """
    v = []
    for r in RUNGS:
        for s in SEEDS:
            d = load(f"attn_{r}_s{s}", "_repair.json")
            if d and key in d:
                v.append(d[key]["mean"])
    if len(v) < 2:
        return "meta: too few runs"
    n = len(v)
    pos = sum(1 for x in v if x > 0)
    p = 2 * sum(comb(n, i) for i in range(max(pos, n - pos), n + 1)) / 2 ** n
    return (f"meta: {pos}/{n} runs positive, sign test p={min(p,1.0):.2e}, "
            f"mean {np.mean(v):+.4f}, range {min(v):+.4f} to {max(v):+.4f}")


def fmt(v):
    if v is None:
        return f"{'--':>24}"
    m, lo, hi = v
    star = "*" if lo > 0 else " "
    return f"{m:+.4f} [{lo:+.4f},{hi:+.4f}]{star}"


def main():
    rows = {}
    for r in RUNGS:
        for s in SEEDS:
            n = f"attn_{r}_s{s}"
            rows[(r, s)] = {"repair": load(n, "_repair.json"),
                            "structure": load(n, "_structure.json"),
                            "natdiv": load(n, "_natdiv.json")}

    have = sum(1 for v in rows.values() if v["repair"])
    print(f"ladder replication: {have}/18 repair runs present\n")

    for key, label in (("correct_minus_wrong_target_entropy_matched",
                        "KILL CRITERION  correct - wrong_target "
                        "(entropy-matched)"),
                       ("correct_minus_irrelevant_entropy_matched",
                        "LOCALITY        correct - irrelevant "
                        "(entropy-matched)")):
        print(label)
        print(f"{'rung':>8} {'params':>11} " +
              " ".join(f"{'seed '+str(s):>25}" for s in SEEDS))
        n_excl = n_tot = 0
        for r in RUNGS:
            cells = []
            for s in SEEDS:
                d = rows[(r, s)]["repair"]
                if d and key in d:
                    v = d[key]
                    cells.append((v["mean"], v["ci_lo"], v["ci_hi"]))
                    n_tot += 1
                    if v["ci_lo"] > 0:
                        n_excl += 1
                else:
                    cells.append(None)
            print(f"{r:>8} {PARAMS[r]:>11,} " +
                  " ".join(fmt(c) for c in cells))
        if n_tot:
            print(f"  intervals excluding zero: {n_excl}/{n_tot}"
                  f"   (* marks those)")
            print("  " + meta(key) + "\n")

    print("STRUCTURE  share of illegal top-1 by class, mean over seeds")
    cls = ["from_empty", "from_opponent", "to_own", "geometry",
           "leaves_check", "other"]
    print(f"{'rung':>8} " + " ".join(f"{c[:12]:>13}" for c in cls)
          + f" {'recent':>8}")
    for r in RUNGS:
        vals = {c: [] for c in cls}
        rec = []
        for s in SEEDS:
            d = rows[(r, s)]["structure"]
            if d:
                for c in cls:
                    vals[c].append(d["overall"][c]["mean"])
                rec.append(d["legal_recently"])
        if rec:
            print(f"{r:>8} " +
                  " ".join(f"{np.mean(vals[c]):13.4f}" for c in cls)
                  + f" {np.mean(rec):8.4f}")

    print("\nLEAVES_CHECK BY DEPTH  (the mechanism claim), mean over seeds")
    print(f"{'rung':>8} {'ply 20-40':>11} {'40-60':>11} {'60-120':>11}"
          f"   {'rises?':>7}")
    for r in RUNGS:
        got = {k: [] for k in ("20-40", "40-60", "60-120")}
        for s in SEEDS:
            d = rows[(r, s)]["structure"]
            if d and "strata" in d:
                for k in got:
                    if k in d["strata"]:
                        got[k].append(d["strata"][k]["leaves_check"])
        if all(got.values()):
            a, b, c = (np.mean(got[k]) for k in ("20-40", "40-60", "60-120"))
            print(f"{r:>8} {a:11.4f} {b:11.4f} {c:11.4f}   "
                  f"{'yes' if c > a else 'NO':>7}")

    print("\nCEILING AND POLICY SHARE, mean over seeds")
    print(f"{'rung':>8} {'illegal rate':>13} {'policy share':>13} "
          f"{'median div':>11}")
    for r in RUNGS:
        ir, ps, md = [], [], []
        for s in SEEDS:
            d = rows[(r, s)]["natdiv"]
            if d:
                ir.append(d["illegal_rate"])
                ps.append(d["policy_share"]["mean"])
                md.append(d["ndiv_illegal"]["median"])
        if ir:
            print(f"{r:>8} {np.mean(ir):13.4f} {np.mean(ps):13.4f} "
                  f"{np.mean(md):11.1f}")

    out = {"n_repair_runs": have,
           "rungs": RUNGS, "seeds": SEEDS}
    json.dump(out, open(os.path.join(RES, "ladder_summary.json"), "w"),
              indent=1)


if __name__ == "__main__":
    main()
