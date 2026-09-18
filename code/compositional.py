"""Compositional scaling: fit the failure classes on the simplex, not one by one.

Why this exists. The paper fits one power law per failure class and extrapolates
them. That is the standard move, and it is not closed: the classes partition the
illegal moves, so their absolute rates must sum to the illegal-move rate at every
scale, and nothing in five independent fits enforces it. The violation is forced
rather than accidental. The total's fitted exponent b is a rate-weighted average of
the parts' exponents in range, so the largest part exponent a* exceeds b whenever
the exponents differ at all, and the ratio of parts to whole then grows like
N^(a*-b) without bound. There is therefore always a finite scale past which a
per-category fit describes an impossibility, and it can be computed from the
reported exponents alone.

On our ladder the parts sum to 1.02 of the whole at our largest rung, which is fit
noise, but 1.27 at the one billion parameters the paper used to extrapolate to, and
2.82 at one trillion. The same computation on Karvonen's independent ladder, which
we did not train, breaks down likewise. A category assigned more than all the
failures is not a wide estimate, it is an arithmetic impossibility.

The fix is to model the shares with a softmax linear in log N, closed by
construction: shares sum to one at every scale, so absolute rate = share(N) x
illegal_rate(N) can never exceed the total it is part of. The total keeps its own
power law, which is legitimate because the total is not a part of anything.
Coherence is not bought with fit quality; the closed model fits slightly better in
range.

A note on what belongs in the simplex. The five classes plus the unattributed
residual sum to exactly one. The policy share, the fraction of illegal moves
occurring despite correctly decoded local belief, does NOT: it cuts across the
classes and is a conditional probability, not a part of the partition. An earlier
version of this file put it in the simplex, which is wrong, and the error
manufactured an interior peak in the check class that does not exist. Modelled
correctly the check share is monotone. The policy share gets its own bounded model
on the logit, for the same reason the classes get a simplex: a quantity confined to
[0,1] must be fitted with a link that respects the confinement.
"""
import os, json, sys
import numpy as np
from scipy.optimize import minimize

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ds_stats as S

# The true partition: these sum to exactly 1 in every structure file.
PART = S.CLASSES + ["other"]
PIVOT = 1e7            # centring scale for conditioning, not a fitted quantity
N_BOOT = 1000


def _composition(rws):
    x = np.log(np.array([q["params"] for q in rws], float)) - np.log(PIVOT)
    Y = np.zeros((len(rws), len(PART)))
    for i, q in enumerate(rws):
        for j, k in enumerate(S.CLASSES):
            Y[i, j] = q["share_" + k]
        Y[i, -1] = q["share_other"]
    return x, Y / Y.sum(1, keepdims=True)


def _shares(p, xs):
    a = np.concatenate([[0.0], p[:len(PART) - 1]])
    b = np.concatenate([[0.0], p[len(PART) - 1:]])
    z = a[None, :] + b[None, :] * np.asarray(xs, float)[:, None]
    z -= z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def fit_simplex(rws):
    x, Y = _composition(rws)

    def nll(p):
        P = np.clip(_shares(p, x), 1e-12, 1.0)
        return -(Y * np.log(P)).sum()

    return minimize(nll, np.zeros(2 * (len(PART) - 1)), method="L-BFGS-B").x


def shares_at(p, N):
    return _shares(p, [np.log(N) - np.log(PIVOT)])[0]


# ------------------------------------------------- the uncoupled baseline
def _powerlaw(rws, key):
    x = np.log([q["params"] for q in rws])
    y = np.log([max(q[key], 1e-12) for q in rws])
    return np.polyfit(x, y, 1)


def overspend(rws, N, keys=None):
    """Sum of independently extrapolated parts, over the extrapolated whole."""
    keys = keys or S.CLASSES
    s = sum(np.exp(np.polyval(_powerlaw(rws, k), np.log(N))) for k in keys)
    return float(s / np.exp(np.polyval(_powerlaw(rws, "illegal_rate"), np.log(N))))


def divergence_rate(rws, keys=None):
    """a* - b: the exponent at which parts outgrow the whole. Positive means the
    decomposition must become impossible at some finite scale."""
    keys = keys or S.CLASSES
    a = max(_powerlaw(rws, k)[0] for k in keys)
    b = _powerlaw(rws, "illegal_rate")[0]
    return float(a - b), float(a), float(b)


def weighted_mean_exponent(rws, keys=None):
    """The rate-weighted mean of the part exponents, which b tracks in range.
    a* >= this mean always, with equality only if every exponent is equal."""
    keys = keys or S.CLASSES
    w = {k: np.mean([q[k] for q in rws]) for k in keys}
    tot = sum(w.values())
    return float(sum(w[k] * _powerlaw(rws, k)[0] for k in keys) / tot)


def breakdown_scale(rws, tol=1.10, keys=None):
    """Smallest N at which the parts exceed the whole by more than tol."""
    lo, hi = 1e6, 1e30
    if overspend(rws, hi, keys) < tol:
        return float("nan")
    if overspend(rws, lo, keys) >= tol:
        return lo
    for _ in range(300):
        mid = np.exp((np.log(lo) + np.log(hi)) / 2)
        if overspend(rws, mid, keys) < tol:
            lo = mid
        else:
            hi = mid
    return float(np.exp((np.log(lo) + np.log(hi)) / 2))


def rmse(rws):
    x, Y = _composition(rws)
    P = _shares(fit_simplex(rws), x)
    I = np.column_stack([
        np.exp(np.polyval(np.polyfit(x, np.log(np.maximum(Y[:, j], 1e-9)), 1), x))
        for j in range(len(PART))])
    return (float(np.sqrt(((P - Y) ** 2).mean())),
            float(np.sqrt(((I - Y) ** 2).mean())))


# ------------------------------------------- the cross-cutting policy share
def policy_logit(rws):
    """Policy share on the logit, so the fitted value stays inside [0,1]."""
    x = np.log(np.array([q["params"] for q in rws], float)) - np.log(PIVOT)
    ps = np.clip(np.array([q["share_policy"] for q in rws]), 1e-6, 1 - 1e-6)
    m, c = np.polyfit(x, np.log(ps / (1 - ps)), 1)
    return float(m), float(c)


def policy_at(mc, N):
    m, c = mc
    return float(1 / (1 + np.exp(-(m * (np.log(N) - np.log(PIVOT)) + c))))


# ----------------------------------------------------- external replication
def external_partition_rows():
    """Karvonen's ladder, keeping only classes with a nonzero rate at every
    checkpoint. 'malformed' is zero at two of three, so an exponent for it is
    an artefact of the floor, not a measurement."""
    ext = S.external_rows()
    cand = ["unreachable", "leaves_check", "to_own", "ambiguous", "malformed"]
    keep = [k for k in cand
            if all(r["absolute"].get(k, 0.0) > 0 for r in ext)]
    rows = [{"params": r["params"], "illegal_rate": r["illegal_rate"],
             **{k: r["absolute"][k] for k in keep}} for r in ext]
    return rows, keep


def main():
    rows = S.ladder_rows()
    p = fit_simplex(rows)
    out = {"n_runs": len(rows), "partition": PART, "pivot": PIVOT}

    r_s, r_i = rmse(rows)
    out["rmse_simplex"], out["rmse_independent"] = r_s, r_i
    out["simplex_fits_better"] = bool(r_s < r_i)

    d, a_star, b = divergence_rate(rows)
    out["divergence_rate"], out["a_star"], out["b_total"] = d, a_star, b
    out["weighted_mean_exponent"] = weighted_mean_exponent(rows)
    out["breakdown_scale"] = breakdown_scale(rows)

    out["overspend"] = {}
    for tag, N in (("ours_large", 3.99e7), ("proj_1e9", 1e9), ("proj_1e12", 1e12)):
        out["overspend"][tag] = {"N": N, "ratio": overspend(rows, N)}

    out["composition"] = {}
    for tag, N in (("ours_small", 3.44e6), ("ours_large", 3.99e7),
                   ("proj_1e9", 1e9), ("proj_1e12", 1e12)):
        s = shares_at(p, N)
        out["composition"][tag] = {"N": N, **{k: float(v) for k, v in zip(PART, s)}}

    # Is the check share monotone under the closed model, or does it turn?
    grid = np.logspace(6, 15, 400)
    j = PART.index("leaves_check")
    vals = np.array([shares_at(p, N)[j] for N in grid])
    k = int(vals.argmax())
    out["check_monotone"] = bool(k == len(grid) - 1)
    out["check_max_share"] = float(vals[k])

    mc = policy_logit(rows)
    out["policy_logit_slope"] = mc[0]
    out["policy_at"] = {t: policy_at(mc, N) for t, N in
                        (("ours_large", 3.99e7), ("proj_1e9", 1e9),
                         ("proj_1e12", 1e12))}
    out["policy_obs_min"] = float(min(q["share_policy"] for q in rows))
    out["policy_obs_max"] = float(max(q["share_policy"] for q in rows))

    # Independent replication of the methodological failure
    erows, ekeys = external_partition_rows()
    ed, ea, eb = divergence_rate(erows, ekeys)
    out["ext_classes"] = ekeys
    out["ext_divergence_rate"], out["ext_a_star"], out["ext_b"] = ed, ea, eb
    out["ext_overspend_1e9"] = overspend(erows, 1e9, ekeys)
    out["ext_breakdown"] = breakdown_scale(erows, keys=ekeys)

    bs = []
    for sub in S._rung_resamples(rows, n=N_BOOT):
        try:
            bs.append(overspend(sub, 1e9))
        except Exception:
            pass
    bs = np.array([v for v in bs if np.isfinite(v)])
    out["overspend_1e9_lo"] = float(np.percentile(bs, 2.5))
    out["overspend_1e9_hi"] = float(np.percentile(bs, 97.5))
    out["overspend_1e9_frac_above_one"] = float((bs > 1).mean())

    json.dump(out, open(os.path.join(S.RES, "compositional.json"), "w"), indent=1)

    print(f"runs {out['n_runs']}   RMSE simplex {r_s:.5f} independent {r_i:.5f} "
          f"better={out['simplex_fits_better']}")
    print(f"a* {a_star:+.4f}  b {b:+.4f}  divergence {d:+.4f}  "
          f"weighted-mean {out['weighted_mean_exponent']:+.4f}")
    print(f"breakdown (>10% over) at N={out['breakdown_scale']:.3g}")
    for t, v in out["overspend"].items():
        print(f"  overspend {t:11s} N={v['N']:8.3g}  {v['ratio']:.3f}")
    print(f"  1e9 overspend 95% [{out['overspend_1e9_lo']:.3f}, "
          f"{out['overspend_1e9_hi']:.3f}], "
          f"{out['overspend_1e9_frac_above_one']:.3f} above 1")
    print("\nclosed composition:")
    for t, dd in out["composition"].items():
        print(f"  {t:11s} " + " ".join(f"{k[:9]}={dd[k]:.3f}" for k in PART))
    print(f"check share monotone: {out['check_monotone']}  "
          f"max {out['check_max_share']:.3f}")
    print(f"\npolicy logit slope {mc[0]:+.4f}; "
          + " ".join(f"{t}={v:.3f}" for t, v in out["policy_at"].items()))
    print(f"\nexternal ({','.join(ekeys)}): a* {ea:+.4f} b {eb:+.4f} "
          f"divergence {ed:+.4f}  overspend@1e9 {out['ext_overspend_1e9']:.3f} "
          f"breakdown {out['ext_breakdown']:.3g}")


if __name__ == "__main__":
    main()
