"""Is the aggregate's curvature detectable inside the observed range?

This decides how strong a claim the paper is entitled to make.

The derived total's slope moves by about 0.07 across our ladder. If that curvature
is statistically detectable on 18 runs spanning 12-fold in parameters, then a single
power law is a demonstrably wrong model of the aggregate ON THE DATA, not merely
asymptotically, and the paper can say the two reports cannot both be right. If it is
not detectable, then a power law is a perfectly serviceable local model of the
aggregate over the fitted range, the conflict is purely a statement about
extrapolation, and the abstract has to say "cannot both be extrapolated" instead.

Test: fit log T = c + b log N + q (log N)^2 to the measured aggregate and ask
whether q differs from zero. Two ways of asking, because 18 runs at 6 sizes are not
18 independent draws:

  1. A t-test on q from the ordinary least-squares fit, which treats runs as
     independent and is therefore optimistic.
  2. A rung bootstrap, resampling the six sizes with replacement, which is the
     convention used everywhere else in this project because the rung is the unit
     of variation.

We report both and take the second as the verdict.
"""
import os, json, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ds_stats as S

N_BOOT = 4000


def _design(rws):
    x = np.log(np.array([q["params"] for q in rws], float))
    x = x - x.mean()                      # centre, so q is not confounded with b
    y = np.log(np.array([q["illegal_rate"] for q in rws], float))
    return x, y


def quad_fit(rws):
    x, y = _design(rws)
    return np.polyfit(x, y, 2)            # [q, b, c]


def ols_t(rws):
    """t statistic on the quadratic term, runs treated as independent."""
    x, y = _design(rws)
    X = np.column_stack([x ** 2, x, np.ones_like(x)])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    dof = len(y) - X.shape[1]
    s2 = (resid @ resid) / dof
    cov = s2 * np.linalg.inv(X.T @ X)
    se = np.sqrt(cov[0, 0])
    return float(beta[0]), float(se), float(beta[0] / se), int(dof)


def main():
    rows = S.ladder_rows()
    x, y = _design(rows)
    out = {"n_runs": len(rows), "n_rungs": len(S.RUNGS)}

    q, se, t, dof = ols_t(rows)
    out["q_ols"], out["q_se"], out["q_t"], out["q_dof"] = q, se, t, dof

    # rung bootstrap: the honest interval
    bs = []
    for sub in S._rung_resamples(rows, n=N_BOOT):
        xx = np.log(np.array([r["params"] for r in sub], float))
        if len(set(xx)) < 3:              # need three distinct sizes for a quadratic
            continue
        xx = xx - xx.mean()
        yy = np.log(np.array([r["illegal_rate"] for r in sub], float))
        bs.append(np.polyfit(xx, yy, 2)[0])
    bs = np.array(bs)
    out["q_boot_lo"] = float(np.percentile(bs, 2.5))
    out["q_boot_hi"] = float(np.percentile(bs, 97.5))
    out["q_boot_frac_neg"] = float((bs < 0).mean())
    out["q_boot_n"] = int(len(bs))
    out["curvature_detected"] = bool(out["q_boot_lo"] > 0 or out["q_boot_hi"] < 0)

    # does adding the quadratic term actually buy anything in range?
    lin = np.polyfit(x, y, 1)
    quad = np.polyfit(x, y, 2)
    r_lin = y - np.polyval(lin, x)
    r_quad = y - np.polyval(quad, x)
    out["rmse_linear"] = float(np.sqrt((r_lin ** 2).mean()))
    out["rmse_quadratic"] = float(np.sqrt((r_quad ** 2).mean()))
    n, k1, k2 = len(y), 2, 3
    out["aic_linear"] = float(n * np.log((r_lin ** 2).mean()) + 2 * k1)
    out["aic_quadratic"] = float(n * np.log((r_quad ** 2).mean()) + 2 * k2)
    out["aic_prefers_quadratic"] = bool(out["aic_quadratic"] < out["aic_linear"])

    # for reference: the curvature the DERIVED total implies over the same range
    import admissibility as A
    fits = A.parts_fits(rows)
    lo, hi = min(r["params"] for r in rows), max(r["params"] for r in rows)
    out["implied_slope_change"] = float(A.effective_exponent(fits, hi)
                                        - A.effective_exponent(fits, lo))
    # a quadratic's slope changes by 2q*(log hi - log lo)
    out["implied_q"] = float(out["implied_slope_change"]
                             / (2 * (np.log(hi) - np.log(lo))))

    json.dump(out, open(os.path.join(S.RES, "curvature.json"), "w"), indent=1)

    print(f"{out['n_runs']} runs at {out['n_rungs']} sizes")
    print(f"OLS quadratic term q = {q:+.5f} (se {se:.5f}), t = {t:+.2f} on {dof} dof")
    print(f"rung bootstrap 95% [{out['q_boot_lo']:+.5f}, {out['q_boot_hi']:+.5f}], "
          f"{out['q_boot_frac_neg']:.3f} below zero, {out['q_boot_n']} resamples")
    print(f"curvature detected in range: {out['curvature_detected']}")
    print(f"RMSE linear {out['rmse_linear']:.5f}  quadratic {out['rmse_quadratic']:.5f}")
    print(f"AIC linear {out['aic_linear']:.2f}  quadratic {out['aic_quadratic']:.2f}  "
          f"prefers quadratic: {out['aic_prefers_quadratic']}")
    print(f"derived total implies slope change {out['implied_slope_change']:+.4f} "
          f"over the range, i.e. q = {out['implied_q']:+.5f}")


if __name__ == "__main__":
    main()
