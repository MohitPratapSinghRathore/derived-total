"""Compositional scaling: fit the failure classes on the simplex, not one by one.

Why this exists. The paper fits one power law per failure class and extrapolates
them. That is the standard move, and it is not closed: the classes partition the
illegal moves, so their absolute rates must sum to the illegal-move rate at every
scale, and nothing in six independent fits enforces it. Different exponents
guarantee the identity fails off the fitted range, and it fails in the direction
that flatters the argument, because the shallow classes are the ones the paper
says survive. At our largest rung the independently fitted shares already sum to
about 1.02; at the one billion parameters the paper extrapolates to they sum to
about 1.34; at a trillion they sum to about 3.9, with one share above 2.5. An
extrapolation that assigns a class 250% of the failures is not a weak estimate,
it is an arithmetic impossibility, and it invalidates the projection rather than
widening it.

The fix is to model the shares with a softmax linear in log N, which is closed by
construction: shares sum to one at every scale, for every parameter value, so the
absolute rate share(N) x illegal_rate(N) can never exceed the total it is part of.
The total keeps its own power law, which is legitimate because the total is not a
part of anything.

Two things had to be checked rather than assumed. The closed model could have
bought coherence with fit quality; it does not, and in fact fits slightly better
in range. And it could have overturned the paper's conclusion; it does not, but it
does change the story past our data in a way the independent fits conceal. Under
independent fits the check class rises without limit and appears to become nearly
all failures. Under the closed model it peaks and then declines, because it is
itself displaced by the class that never scales away: failures despite correctly
decoded local belief. The qualitative claim survives, the quantitative projection
does not, and the corrected projection is the more interesting one.
"""
import os, json, sys
import numpy as np
from scipy.optimize import minimize

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ds_stats as S

# The classes partition illegal moves. 'policy' is the residual the paper calls
# failure despite correctly decoded local belief; it belongs in the simplex
# because it is one of the outcomes, not a separate measurement.
K = S.CLASSES + ["policy"]
PIVOT = 1e7            # centring scale for conditioning, not a fitted quantity
N_BOOT = 1000


def _design(rows):
    x = np.log(np.array([q["params"] for q in rows], float)) - np.log(PIVOT)
    Y = np.zeros((len(rows), len(K)))
    for i, q in enumerate(rows):
        for j, k in enumerate(S.CLASSES):
            Y[i, j] = q["share_" + k]
        Y[i, -1] = q["share_policy"]
    # Renormalise away the small unattributed residual so the response is a
    # proper composition; its size is reported separately as a quality check.
    tot = Y.sum(1, keepdims=True)
    return x, Y / tot, tot.ravel()


def _shares(p, xs):
    a = np.concatenate([[0.0], p[:len(K) - 1]])
    b = np.concatenate([[0.0], p[len(K) - 1:]])
    z = a[None, :] + b[None, :] * np.asarray(xs, float)[:, None]
    z -= z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def fit_simplex(rows):
    x, Y, _ = _design(rows)

    def nll(p):
        P = np.clip(_shares(p, x), 1e-12, 1.0)
        return -(Y * np.log(P)).sum()

    r = minimize(nll, np.zeros(2 * (len(K) - 1)), method="L-BFGS-B")
    return r.x


def shares_at(p, N):
    """Predicted composition at parameter count N."""
    return _shares(p, [np.log(N) - np.log(PIVOT)])[0]


def _independent(rows, key_index):
    """Log-linear fit of one share on log N, the uncoupled baseline."""
    x, Y, _ = _design(rows)
    m, c = np.polyfit(x, np.log(np.maximum(Y[:, key_index], 1e-9)), 1)
    return m, c


def independent_shares_at(rows, N):
    xs = np.log(N) - np.log(PIVOT)
    return np.array([np.exp(m * xs + c) for m, c in
                     (_independent(rows, j) for j in range(len(K)))])


def rmse(rows):
    """In-range fit quality, closed model against the uncoupled baseline."""
    x, Y, _ = _design(rows)
    P = _shares(fit_simplex(rows), x)
    I = np.column_stack([np.exp(m * x + c) for m, c in
                         (_independent(rows, j) for j in range(len(K)))])
    return float(np.sqrt(((P - Y) ** 2).mean())), float(np.sqrt(((I - Y) ** 2).mean()))


def peak_scale(p, lo=1e6, hi=1e15, key="leaves_check"):
    """Scale at which a class's share is maximal under the closed model.

    Golden-section on a unimodal curve; returns nan if the maximum sits at an
    endpoint, which is the honest answer when the data do not imply a turn.
    """
    j = K.index(key)
    f = lambda N: shares_at(p, N)[j]
    gr = (np.sqrt(5) - 1) / 2
    a, b = np.log(lo), np.log(hi)
    c_, d_ = b - gr * (b - a), a + gr * (b - a)
    for _ in range(200):
        if f(np.exp(c_)) > f(np.exp(d_)):
            b, d_ = d_, c_
            c_ = b - gr * (b - a)
        else:
            a, c_ = c_, d_
            d_ = a + gr * (b - a)
    N = float(np.exp((a + b) / 2))
    if N < lo * 1.01 or N > hi * 0.99:
        return float("nan")
    return N


def main():
    rows = S.ladder_rows()
    p = fit_simplex(rows)
    out = {"n_runs": len(rows), "classes": K, "pivot": PIVOT}

    r_simplex, r_indep = rmse(rows)
    out["rmse_simplex"] = r_simplex
    out["rmse_independent"] = r_indep
    out["simplex_fits_better"] = bool(r_simplex < r_indep)

    # How badly does the uncoupled baseline violate the identity it ignores?
    out["share_sum"] = {}
    for tag, N in (("ours_small", 3.44e6), ("ours_large", 3.99e7),
                   ("proj_1e9", 1e9), ("proj_1e12", 1e12)):
        s = independent_shares_at(rows, N)
        out["share_sum"][tag] = {"N": N, "sum": float(s.sum()),
                                 "max_share": float(s.max()),
                                 "argmax": K[int(s.argmax())]}

    # Closed-model composition at the same scales
    out["composition"] = {}
    for tag, N in (("ours_small", 3.44e6), ("ours_large", 3.99e7),
                   ("proj_1e9", 1e9), ("proj_1e12", 1e12)):
        s = shares_at(p, N)
        out["composition"][tag] = {"N": N,
                                   **{k: float(v) for k, v in zip(K, s)}}

    # The new claim: the global class peaks rather than rising without limit.
    pk = peak_scale(p)
    out["peak_leaves_check"] = pk
    out["peak_share"] = float(shares_at(p, pk)[K.index("leaves_check")]) \
        if np.isfinite(pk) else None

    boots = []
    for sub in S._rung_resamples(rows, n=N_BOOT):
        try:
            v = peak_scale(fit_simplex(sub))
        except Exception:
            v = float("nan")
        if np.isfinite(v):
            boots.append(v)
    boots = np.array(boots)
    out["peak_lo"] = float(np.percentile(boots, 2.5)) if len(boots) else None
    out["peak_hi"] = float(np.percentile(boots, 97.5)) if len(boots) else None
    out["peak_n_boot"] = int(len(boots))
    out["peak_frac_above_our_max"] = float((boots > 3.99e7).mean()) if len(boots) else None

    # Does the policy class overtake the check class, and where?
    def gap(N):
        s = shares_at(p, N)
        return s[K.index("policy")] - s[K.index("leaves_check")]
    lo, hi = 1e6, 1e18
    cross = float("nan")
    if gap(lo) < 0 < gap(hi):
        for _ in range(200):
            mid = np.exp((np.log(lo) + np.log(hi)) / 2)
            if gap(mid) < 0:
                lo = mid
            else:
                hi = mid
        cross = float(np.exp((np.log(lo) + np.log(hi)) / 2))
    out["policy_overtakes_check"] = cross

    json.dump(out, open(os.path.join(S.RES, "compositional.json"), "w"), indent=1)

    print(f"runs {out['n_runs']}  RMSE simplex {r_simplex:.5f} "
          f"independent {r_indep:.5f}  better={out['simplex_fits_better']}")
    print("\nuncoupled baseline, sum of extrapolated shares (must be 1):")
    for tag, d in out["share_sum"].items():
        print(f"  {tag:12s} N={d['N']:9.3g}  sum={d['sum']:6.3f} "
              f"largest={d['argmax']} at {d['max_share']:.3f}")
    print("\nclosed model composition:")
    for tag, d in out["composition"].items():
        print(f"  {tag:12s} " + " ".join(f"{k}={d[k]:.3f}" for k in K))
    print(f"\ncheck class peaks at {pk:.3g} params "
          f"[{out['peak_lo']:.3g}, {out['peak_hi']:.3g}] "
          f"({out['peak_n_boot']} boots, {out['peak_frac_above_our_max']:.3f} above our ladder)")
    print(f"policy class overtakes check at {cross:.3g} params")


if __name__ == "__main__":
    main()
