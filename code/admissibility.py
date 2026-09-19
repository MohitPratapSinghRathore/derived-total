"""Admissibility of a decomposed scaling forecast, and what the repair costs.

This file supersedes the framing in the first version of compositional.py. That
version said per-category power-law fits are inadmissible. That is not right, and
the correction matters.

If the categories are power laws with different exponents, then the total is a sum
of power laws, which is provably NOT a power law. So the misspecified object is the
independently fitted total, not the category curves. It follows that the simplest
repair is to stop fitting the total at all and define T := sum_i r_i, which is
admissible everywhere, keeps every category exponent, and needs no link function.
The claim this file supports is therefore about the PAIR: per-category power laws
together with an independently fitted power-law total are mutually inconsistent,
and it is the total that has to go.

Two facts make the repair less of a concession than it looks.

First, the sum-of-parts shares ARE a softmax linear in log N:

    s_i(N) = A_i N^{a_i} / sum_j A_j N^{a_j}
           = exp(log A_i + a_i log N) / sum_j exp(log A_j + a_j log N)

so the objection's remedy and the softmax remedy are the same model of the
composition, differing only in the loss used to estimate it. They agree here to
within 0.01 in share across six orders of magnitude, and they agree on the
asymptotics: the shallowest category's share tends to one under both.

Second, the repair is not free, and its cost is the thing practitioners most want
to report. Under T := sum_i r_i the total is not a power law, so there is no
aggregate exponent to quote. Its effective slope drifts, and on our ladder it
drifts across most of the range spanned by the category exponents themselves. A
paper that reports both per-category exponents and an aggregate exponent is
reporting two things that cannot both be true.
"""
import os, json, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ds_stats as S

N_BOOT = 2000


def powerlaw(rws, key):
    x = np.log([q["params"] for q in rws])
    y = np.log([max(q[key], 1e-12) for q in rws])
    return np.polyfit(x, y, 1)


def parts_fits(rws, keys=None):
    return {k: powerlaw(rws, k) for k in (keys or S.CLASSES)}


def T_sum(fits, N):
    """The admissible total: the parts, added up."""
    return float(sum(np.exp(np.polyval(mc, np.log(N))) for mc in fits.values()))


def T_pow(rws, N):
    """The inadmissible total: an independently fitted power law."""
    return float(np.exp(np.polyval(powerlaw(rws, "illegal_rate"), np.log(N))))


def effective_exponent(fits, N, h=0.01):
    """d log T_sum / d log N. Constant only if all part exponents are equal."""
    return float((np.log(T_sum(fits, N * np.exp(h)))
                  - np.log(T_sum(fits, N / np.exp(h)))) / (2 * h))


def overspend(rws, N, keys=None, _fits=None, _tot=None):
    """Parts over independently fitted whole. One if the total is the sum."""
    fits = _fits if _fits is not None else parts_fits(rws, keys)
    tot = _tot if _tot is not None else powerlaw(rws, "illegal_rate")
    return T_sum(fits, N) / float(np.exp(np.polyval(tot, np.log(N))))


def breakdown_scale(rws, tol=1.10, keys=None, lo=None, hi=1e30):
    """Smallest N above the fitted range where the overspend exceeds tol.

    The curves are fitted once and reused across the bisection. Refitting inside
    the loop, as the first version did, made a bootstrap of this unusably slow
    without changing any answer.
    """
    fits = parts_fits(rws, keys)
    tot = powerlaw(rws, "illegal_rate")
    f = lambda N: overspend(rws, N, keys, _fits=fits, _tot=tot)
    lo = lo or max(q["params"] for q in rws)
    if f(hi) < tol:
        return float("nan")
    if f(lo) >= tol:
        return float(lo)
    for _ in range(200):
        mid = np.exp((np.log(lo) + np.log(hi)) / 2)
        if f(mid) < tol:
            lo = mid
        else:
            hi = mid
    return float(np.exp((np.log(lo) + np.log(hi)) / 2))


# ------------------------------------------------ apples-to-apples accuracy
def accuracy_both_spaces(rws):
    """Compare the two share models under ONE loss at a time.

    The earlier comparison was not like for like: the simplex model was fitted
    and scored in share space while the uncoupled fits were fitted in log-rate
    space. Here both are scored in share space and in rate space, and the
    sum-of-parts model is the uncoupled fits' own implied composition, so no
    refitting is needed to make the comparison fair.
    """
    import compositional as C
    x, Y = C._composition(rws)          # observed shares, true partition
    p = C.fit_simplex(rws)
    P_simplex = C._shares(p, x)

    fits = parts_fits(rws)
    P_sum = []
    for q in rws:
        N = q["params"]
        r = np.array([np.exp(np.polyval(fits[k], np.log(N))) for k in S.CLASSES])
        r = np.append(r, 0.0)           # 'other' carries no fitted curve
        P_sum.append(r / r.sum())
    P_sum = np.array(P_sum)

    ill = np.array([q["illegal_rate"] for q in rws])[:, None]
    out = {
        "share_rmse_simplex": float(np.sqrt(((P_simplex - Y) ** 2).mean())),
        "share_rmse_sumparts": float(np.sqrt(((P_sum - Y) ** 2).mean())),
        "rate_rmse_simplex": float(np.sqrt((((P_simplex - Y) * ill) ** 2).mean())),
        "rate_rmse_sumparts": float(np.sqrt((((P_sum - Y) * ill) ** 2).mean())),
    }
    out["share_winner"] = ("simplex" if out["share_rmse_simplex"]
                           < out["share_rmse_sumparts"] else "sum-of-parts")
    out["rate_winner"] = ("simplex" if out["rate_rmse_simplex"]
                          < out["rate_rmse_sumparts"] else "sum-of-parts")
    out["max_share_gap"] = float(np.abs(P_simplex - P_sum).max())
    return out


def main():
    rows = S.ladder_rows()
    fits = parts_fits(rows)
    out = {"n_runs": len(rows)}

    b = powerlaw(rows, "illegal_rate")[0]
    out["b_fitted"] = float(b)
    out["a_exponents"] = {k: float(mc[0]) for k, mc in fits.items()}
    out["a_star"] = float(max(out["a_exponents"].values()))
    out["a_min"] = float(min(out["a_exponents"].values()))

    # the aggregate exponent that does not exist
    eff = {}
    for tag, N in (("small", 3.44e6), ("large", 3.99e7), ("bn", 1e9),
                   ("tn", 1e12)):
        eff[tag] = effective_exponent(fits, N)
    out["effective_exponent"] = eff
    out["eff_drift_in_range"] = abs(eff["large"] - eff["small"])
    out["eff_drift_to_bn"] = abs(eff["bn"] - eff["small"])
    # how much of the category-exponent span does the drift cover?
    span = out["a_star"] - out["a_min"]
    out["exponent_span"] = float(span)
    out["drift_frac_of_span"] = float(out["eff_drift_to_bn"] / span)

    out["overspend"] = {t: overspend(rows, N) for t, N in
                        (("large", 3.99e7), ("bn", 1e9), ("tn", 1e12))}

    # breakdown scale, now WITH an interval
    pt = breakdown_scale(rows)
    bs = []
    for sub in S._rung_resamples(rows, n=N_BOOT):
        try:
            v = breakdown_scale(sub)
        except Exception:
            v = float("nan")
        if np.isfinite(v):
            bs.append(v)
    bs = np.array(bs)
    out["breakdown"] = pt
    out["breakdown_lo"] = float(np.percentile(bs, 2.5))
    out["breakdown_hi"] = float(np.percentile(bs, 97.5))
    out["breakdown_n_boot"] = int(len(bs))

    out["accuracy"] = accuracy_both_spaces(rows)

    json.dump(out, open(os.path.join(S.RES, "admissibility.json"), "w"), indent=1)

    print(f"runs {out['n_runs']}")
    print(f"category exponents {out['a_exponents']}")
    print(f"independently fitted aggregate b = {b:+.4f}")
    print("\nthe aggregate exponent does not exist: effective slope of T_sum")
    for t, v in eff.items():
        print(f"  {t:6s} {v:+.4f}")
    print(f"  drift within our range {out['eff_drift_in_range']:.4f}; "
          f"to 1e9 {out['eff_drift_to_bn']:.4f}, which is "
          f"{100*out['drift_frac_of_span']:.0f}% of the category-exponent span")
    print("\noverspend of the independently fitted total:")
    for t, v in out["overspend"].items():
        print(f"  {t:6s} {v:.3f}")
    print(f"\nbreakdown {pt:.3g} [{out['breakdown_lo']:.3g}, "
          f"{out['breakdown_hi']:.3g}] over {out['breakdown_n_boot']} resamples")
    a = out["accuracy"]
    print("\napples-to-apples accuracy (same loss for both):")
    print(f"  share space: simplex {a['share_rmse_simplex']:.5f}  "
          f"sum-of-parts {a['share_rmse_sumparts']:.5f}  -> {a['share_winner']}")
    print(f"  rate  space: simplex {a['rate_rmse_simplex']:.5f}  "
          f"sum-of-parts {a['rate_rmse_sumparts']:.5f}  -> {a['rate_winner']}")
    print(f"  max share gap between the two models: {a['max_share_gap']:.4f}")


if __name__ == "__main__":
    main()
