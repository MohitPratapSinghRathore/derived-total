"""Where does the aggregate's curvature come from?

The aggregate's measured log-log curvature is about +0.040. If every category were
a power law, the curvature the aggregate must show is fixed by the category
exponents and intercepts alone, and that implied value is about +0.015. The total
bends roughly 2.7 times more than the decomposition can produce.

That gap has to be resolved before the paper's central subsection can stand. Two
possibilities:

  (a) The gap is sampling noise. The implied curvature is itself estimated from the
      same 18 runs and carries its own interval; if the measured value sits inside
      it, there is nothing to explain.

  (b) The gap is real, in which case at least one category is not a power law
      either, and the paper cannot say "the categories may each be perfectly well
      specified, what cannot be sustained is the claim about their total". The
      honest claim becomes the weaker one: even if the categories WERE power laws
      the aggregate could not be, and here they are not either.

This script decides between them. It bootstraps the implied curvature over the size
rungs, tests the measured-minus-implied difference on the SAME resample so the
comparison is paired, and fits every category quadratically to say which ones bend.
"""
import os, json, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ds_stats as S
import admissibility as A

N_BOOT = 4000


def _x(rws):
    x = np.log(np.array([q["params"] for q in rws], float))
    return x - x.mean()


def measured_q(rws, key="illegal_rate"):
    """Quadratic term of log(rate) on log(params), centred."""
    x = _x(rws)
    y = np.log(np.array([max(q[key], 1e-12) for q in rws], float))
    return float(np.polyfit(x, y, 2)[0])


def implied_q(rws):
    """Curvature the aggregate MUST have if every category is a power law.

    Fit the categories linearly, sum them to get the implied total, then read the
    quadratic term of that implied total at the same design points. This is the
    curvature the decomposition can produce, and nothing more.
    """
    fits = A.parts_fits(rws)
    x = _x(rws)
    xs = np.log(np.array([q["params"] for q in rws], float))
    y = np.log(np.array([A.T_sum(fits, np.exp(v)) for v in xs], float))
    return float(np.polyfit(x, y, 2)[0])


def main():
    rows = S.ladder_rows()
    out = {"n_runs": len(rows)}

    m, i = measured_q(rows), implied_q(rows)
    out["measured_q"], out["implied_q"] = m, i
    out["gap"] = m - i
    out["ratio"] = m / i if i else float("nan")

    # paired bootstrap on the gap: both quantities refitted on the same resample
    gaps, meas, impl = [], [], []
    for sub in S._rung_resamples(rows, n=N_BOOT):
        if len(set(q["params"] for q in sub)) < 3:
            continue
        try:
            a, b = measured_q(sub), implied_q(sub)
        except Exception:
            continue
        if np.isfinite(a) and np.isfinite(b):
            meas.append(a)
            impl.append(b)
            gaps.append(a - b)
    gaps, impl = np.array(gaps), np.array(impl)
    out["gap_lo"] = float(np.percentile(gaps, 2.5))
    out["gap_hi"] = float(np.percentile(gaps, 97.5))
    out["gap_frac_le_zero"] = float((gaps <= 0).mean())
    out["implied_lo"] = float(np.percentile(impl, 2.5))
    out["implied_hi"] = float(np.percentile(impl, 97.5))
    out["measured_inside_implied_interval"] = bool(
        out["implied_lo"] <= m <= out["implied_hi"])
    out["gap_is_noise"] = bool(out["gap_lo"] <= 0 <= out["gap_hi"])
    out["n_boot"] = int(len(gaps))

    # which categories are themselves not power laws?
    out["categories"] = {}
    for k in S.CLASSES:
        q_pt = measured_q(rows, k)
        bs = []
        for sub in S._rung_resamples(rows, n=N_BOOT):
            if len(set(z["params"] for z in sub)) < 3:
                continue
            try:
                v = measured_q(sub, k)
            except Exception:
                continue
            if np.isfinite(v):
                bs.append(v)
        bs = np.array(bs)
        lo, hi = float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))
        out["categories"][k] = {"q": q_pt, "lo": lo, "hi": hi,
                                "curved": bool(lo > 0 or hi < 0)}
    out["n_curved"] = sum(v["curved"] for v in out["categories"].values())

    json.dump(out, open(os.path.join(S.RES, "curvature_decomp.json"), "w"),
              indent=1)

    print(f"measured aggregate curvature {m:+.5f}")
    print(f"implied by the categories     {i:+.5f}  "
          f"[{out['implied_lo']:+.5f}, {out['implied_hi']:+.5f}]")
    print(f"gap {out['gap']:+.5f} [{out['gap_lo']:+.5f}, {out['gap_hi']:+.5f}], "
          f"{out['gap_frac_le_zero']:.3f} at or below zero, {out['n_boot']} resamples")
    print(f"gap explainable as noise: {out['gap_is_noise']}")
    print(f"measured inside implied interval: {out['measured_inside_implied_interval']}")
    print("\nper-category curvature:")
    for k, v in out["categories"].items():
        flag = "CURVED" if v["curved"] else "straight"
        print(f"  {k:16s} q={v['q']:+.5f} [{v['lo']:+.5f}, {v['hi']:+.5f}]  {flag}")
    print(f"\n{out['n_curved']} of {len(S.CLASSES)} categories are detectably curved")


if __name__ == "__main__":
    main()
