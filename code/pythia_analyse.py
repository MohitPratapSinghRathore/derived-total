"""The preregistered analysis of results/pythia_partition.csv.

Runs the four ordered checks from pythia_partition.py's header, then the curvature
test, exactly as PREREGISTERED.md fixes them. It refuses to report anything if
check 1 or 1b fails, because the preregistration says a rung that fails check 1 is
discarded and nothing is reported from an incomplete or faulty ladder.

Nothing here is tuned. The breakdown tolerance is 10 percent, the bootstrap
resamples size rungs 4000 times, and curvature is called detected only if the
bootstrap interval excludes zero.
"""
import os, csv, json, sys
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
RES = os.path.join(HERE, "..", "results")
CSV = os.path.join(RES, "pythia_partition.csv")

TOL_CHECK1 = 1e-6
BREAKDOWN_TOL = 1.10
N_BOOT = 4000


def load():
    rows = defaultdict(dict)
    meta = {}
    with open(CSV, encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            rows[r["model"]][r["source"]] = {
                "sum_nll": float(r["sum_nll"]), "tokens": int(r["tokens"]),
                "n_docs": int(r["n_docs"])}
            meta[r["model"]] = {"params": int(r["params"]),
                                "device": r["device"], "dtype": r["dtype"]}
    return rows, meta


def main():
    if not os.path.exists(CSV):
        print(f"no {CSV}; nothing to analyse")
        return 1
    rows, meta = load()
    models = sorted(rows, key=lambda m: meta[m]["params"])
    sources = sorted(set.intersection(*[set(rows[m]) for m in models]))
    out = {"models": models, "sources": sources,
           "params": {m: meta[m]["params"] for m in models},
           "dtype": {m: meta[m]["dtype"] for m in models},
           "device": {m: meta[m]["device"] for m in models}}
    print(f"{len(models)} rungs, {len(sources)} sources")

    # ---- check 1b: token counts identical at every rung (shared tokenizer)
    ref = {s: rows[models[0]][s]["tokens"] for s in sources}
    mismatches = [(m, s) for m in models for s in sources
                  if rows[m][s]["tokens"] != ref[s]]
    out["check1b_pass"] = not mismatches
    out["check1b_mismatches"] = [f"{m}/{s}" for m, s in mismatches[:8]]
    print(f"check 1b (token counts identical across rungs): "
          f"{'PASS' if out['check1b_pass'] else 'FAIL'}")

    # ---- check 1: pooled aggregate equals the token-weighted mean
    tot_tok = sum(ref[s] for s in sources)
    weights = {s: ref[s] / tot_tok for s in sources}
    out["weights"] = weights
    out["check1"] = {}
    agg = {}
    for m in models:
        pooled = sum(rows[m][s]["sum_nll"] for s in sources) / \
            sum(rows[m][s]["tokens"] for s in sources)
        weighted = sum(weights[s] * (rows[m][s]["sum_nll"] / rows[m][s]["tokens"])
                       for s in sources)
        rel = abs(pooled - weighted) / pooled
        out["check1"][m] = {"pooled": pooled, "weighted": weighted, "rel": rel,
                            "pass": bool(rel <= TOL_CHECK1)}
        agg[m] = pooled
        print(f"  {m:34s} pooled {pooled:.6f} weighted {weighted:.6f} "
              f"rel {rel:.2e} {'ok' if rel <= TOL_CHECK1 else 'FAIL'}")
    out["check1_pass"] = all(v["pass"] for v in out["check1"].values())

    if not (out["check1_pass"] and out["check1b_pass"]):
        out["verdict"] = "VOID: check 1 or 1b failed; nothing is reported"
        json.dump(out, open(os.path.join(RES, "pythia_analysis.json"), "w"),
                  indent=1)
        print("\n" + out["verdict"])
        return 1

    # ---- checks 2 to 4: the consistency condition
    N = np.array([meta[m]["params"] for m in models], float)
    x = np.log(N)
    # weighted per-source loss, so the parts sum to the aggregate
    parts = {s: np.array([weights[s] * rows[m][s]["sum_nll"] /
                          rows[m][s]["tokens"] for m in models]) for s in sources}
    total = np.array([agg[m] for m in models])

    def fit(y):
        return np.polyfit(x, np.log(y), 1)

    pf = {s: fit(v) for s, v in parts.items()}
    tf = fit(total)
    a_star_src = max(pf, key=lambda s: pf[s][0])
    a_star, b = float(pf[a_star_src][0]), float(tf[0])
    out["exponents"] = {s: float(v[0]) for s, v in pf.items()}
    out["a_star"], out["a_star_source"], out["b"] = a_star, a_star_src, b
    out["divergence"] = a_star - b
    print(f"\ncheck 2: a* {a_star:+.4f} ({a_star_src}) vs aggregate b {b:+.4f}"
          f"  divergence {a_star - b:+.4f}")

    def overspend(NN, pfits=pf, tfit=tf):
        s = sum(np.exp(np.polyval(v, np.log(NN))) for v in pfits.values())
        return float(s / np.exp(np.polyval(tfit, np.log(NN))))

    def breakdown(pfits=pf, tfit=tf, lo=None):
        lo = lo or float(N.max())
        hi = 1e30
        if overspend(hi, pfits, tfit) < BREAKDOWN_TOL:
            return float("nan")
        if overspend(lo, pfits, tfit) >= BREAKDOWN_TOL:
            return lo
        for _ in range(200):
            mid = np.exp((np.log(lo) + np.log(hi)) / 2)
            if overspend(mid, pfits, tfit) < BREAKDOWN_TOL:
                lo = mid
            else:
                hi = mid
        return float(np.exp((np.log(lo) + np.log(hi)) / 2))

    out["breakdown"] = breakdown()
    out["overspend_top_of_range"] = overspend(float(N.max()))
    out["overspend_1e12"] = overspend(1e12)
    print(f"check 3: breakdown {out['breakdown']:.3g}; overspend at top of range "
          f"{out['overspend_top_of_range']:.3f}, at 1e12 {out['overspend_1e12']:.3f}")
    out["breakdown_within_observed"] = bool(
        np.isfinite(out["breakdown"]) and out["breakdown"] <= N.max())

    # ---- check 4 plus the curvature test, both bootstrapped over rungs
    xc = x - x.mean()
    q_pt = float(np.polyfit(xc, np.log(total), 2)[0])
    rng = np.random.default_rng(0)
    qs, divs, bds = [], [], []
    idx = np.arange(len(models))
    for _ in range(N_BOOT):
        take = rng.choice(idx, len(idx), replace=True)
        if len(set(take)) < 3:
            continue
        xx, tt = x[take], total[take]
        xxc = xx - xx.mean()
        try:
            qs.append(np.polyfit(xxc, np.log(tt), 2)[0])
            pfb = {s: np.polyfit(xx, np.log(parts[s][take]), 1) for s in sources}
            tfb = np.polyfit(xx, np.log(tt), 1)
            divs.append(max(v[0] for v in pfb.values()) - tfb[0])
            bds.append(breakdown(pfb, tfb, lo=float(N[take].max())))
        except Exception:
            continue
    qs, divs = np.array(qs), np.array(divs)
    bds = np.array([v for v in bds if np.isfinite(v)])
    out["curvature_q"] = q_pt
    out["curvature_lo"] = float(np.percentile(qs, 2.5))
    out["curvature_hi"] = float(np.percentile(qs, 97.5))
    out["curvature_detected"] = bool(out["curvature_lo"] > 0 or
                                     out["curvature_hi"] < 0)
    out["divergence_lo"] = float(np.percentile(divs, 2.5))
    out["divergence_hi"] = float(np.percentile(divs, 97.5))
    out["divergence_frac_pos"] = float((divs > 0).mean())
    if len(bds):
        out["breakdown_lo"] = float(np.percentile(bds, 2.5))
        out["breakdown_hi"] = float(np.percentile(bds, 97.5))
    out["n_boot"] = int(len(qs))

    print(f"check 4: curvature q {q_pt:+.5f} "
          f"[{out['curvature_lo']:+.5f}, {out['curvature_hi']:+.5f}] "
          f"detected={out['curvature_detected']}")
    print(f"         divergence [{out['divergence_lo']:+.4f}, "
          f"{out['divergence_hi']:+.4f}], positive in "
          f"{out['divergence_frac_pos']:.3f} of resamples")

    # the preregistered verdict
    if out["divergence"] > 0 and np.isfinite(out["breakdown"]) \
            and out["breakdown"] <= 1e12:
        out["verdict"] = "POSITIVE: inconsistent at scales of practical interest"
    elif out["divergence"] <= 0:
        out["verdict"] = ("NEGATIVE: divergence rate is not positive; the parts do "
                          "not outgrow the fitted whole on this ladder")
    else:
        out["verdict"] = ("NEGATIVE: divergence positive but the breakdown scale "
                          "lies beyond any scale anyone extrapolates to")
    print("\n" + out["verdict"])

    json.dump(out, open(os.path.join(RES, "pythia_analysis.json"), "w"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
