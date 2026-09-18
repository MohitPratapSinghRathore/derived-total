"""Check whether a published per-category scaling decomposition is admissible.

Standalone on purpose. It imports numpy and nothing from this project, so it can
be pointed at any paper that reports per-category scaling exponents, including
ones with no connection to chess. The whole diagnostic needs only what such papers
already print.

The check. Categories partitioning a total must satisfy sum_i r_i(N) = T(N) at
every N. Fitting r_i(N) = A_i N^{a_i} and T(N) = B N^{b} independently does not
enforce this, and cannot: the parts sum asymptotically to N^{a*} where a* = max_i
a_i, the whole goes as N^{b}, so their ratio grows as N^{a*-b}. Fitted in range, b
tracks the rate-weighted mean of the a_i, and a maximum is at least a weighted
mean. So a* > b whenever the exponents differ at all, which is the only reason to
report them separately, and every such decomposition describes an impossibility
beyond a finite scale.

What to do with a positive result. It does not mean the fitted exponents are
wrong; in range they may be excellent. It means the extrapolation is not entitled
to the numbers it reports past the breakdown scale, and that a closed
parameterisation (softmax over the categories, times a separate law for the total)
should be used instead. It also means any NEGATIVE conclusion drawn from the
extrapolation is void: a model that predicts impossibilities cannot be used to
reject a hypothesis on the grounds that its predictions are wrong.

Usage
-----
    from closure_check import check, from_points

    # from reported exponents and intercepts (log-log intercepts, natural log)
    check(parts={'a': (-0.57, -1.2), 'b': (-0.11, -2.0)}, total=(-0.27, -0.8))

    # or straight from measurements
    check(**from_points(N=[...], parts={'a': [...]}, total=[...]))

    python closure_check.py          # runs the self-test on this project's data
"""
import numpy as np

__all__ = ["check", "from_points", "Report"]


class Report(dict):
    """Plain dict with a readable printout."""

    def __str__(self):
        w = "ADMISSIBLE" if not self["diverges"] else "NOT ADMISSIBLE"
        L = [f"closure check: {w}",
             f"  largest part exponent a*   {self['a_star']:+.4f}  ({self['a_star_name']})",
             f"  total exponent b           {self['b']:+.4f}",
             f"  divergence rate a* - b     {self['divergence']:+.4f}"]
        if self["diverges"]:
            where = ("already within the fitted range"
                     if self.get("breakdown_within_range") else
                     f"N = {self['breakdown']:.3g}")
            L.append(f"  parts exceed whole by >{self['tol']:.0%} at {where}")
            for N, r in self["overspend"].items():
                L.append(f"    overspend at N={N:<10.3g} {r:.3f}")
        else:
            L.append("  parts do not outgrow the whole; extrapolation is coherent")
        return "\n".join(L)


def from_points(N, parts, total):
    """Fit log-log lines to measurements. N is shared by every series.

    Also returns fitted_max, the top of the range actually observed. The
    breakdown search starts there: the parts-over-whole ratio is U-shaped, large
    at small N because the steep categories dominate and large at big N because
    the shallow one does, so searching from zero finds the wrong arm and reports
    a breakdown below the data.
    """
    x = np.log(np.asarray(N, float))
    if len(x) < 2:
        raise ValueError("need at least two scales to fit a slope")

    def f(y):
        y = np.asarray(y, float)
        if y.shape != x.shape:
            raise ValueError("every series must have one value per scale in N")
        if (y <= 0).any():
            raise ValueError(
                "non-positive rate: a log-log fit is undefined there. Drop the "
                "category (a class that is zero at some scale has no exponent) "
                "rather than flooring it, which invents a slope.")
        return tuple(np.polyfit(x, np.log(y), 1))

    return {"parts": {k: f(v) for k, v in parts.items()}, "total": f(total),
            "fitted_max": float(np.exp(x.max()))}


def _val(mc, N):
    m, c = mc
    return np.exp(m * np.log(N) + c)


def check(parts, total, tol=0.10, report_at=(1e9, 1e12), fitted_max=None,
          atol=1e-9):
    """parts: {name: (exponent, intercept)}; total: (exponent, intercept).

    tol is the fractional overspend treated as the breakdown threshold, so the
    default asks where the parts exceed the whole by more than ten percent,
    which keeps in-range fit noise from counting as a violation.

    fitted_max is the largest scale actually observed; the breakdown search
    starts there, since only extrapolation above the data is at issue. Pass the
    value from_points returns, or omit it to search from the smallest scale the
    reported figures mention.

    atol guards the comparison a* > b. Identical exponents differ by rounding,
    and a decomposition whose categories genuinely share one exponent is
    admissible however the arithmetic lands.
    """
    if not parts:
        raise ValueError("no categories given")
    name, (a_star, _) = max(parts.items(), key=lambda kv: kv[1][0])
    b = total[0]
    div = a_star - b

    def overspend(N):
        return float(sum(_val(mc, N) for mc in parts.values()) / _val(total, N))

    out = Report(a_star=float(a_star), a_star_name=name, b=float(b),
                 divergence=float(div), diverges=bool(div > atol), tol=tol,
                 n_parts=len(parts), fitted_max=fitted_max,
                 overspend={float(N): overspend(N) for N in report_at})

    if not out["diverges"]:
        out["breakdown"] = float("nan")
        return out

    lo = float(fitted_max) if fitted_max else min(report_at) / 1e6
    hi = max(lo * 1e30, 1e40)
    thr = 1.0 + tol
    if overspend(lo) >= thr:
        out["breakdown"] = lo
        out["breakdown_within_range"] = True
        return out
    for _ in range(400):
        mid = np.exp((np.log(lo) + np.log(hi)) / 2)
        if overspend(mid) < thr:
            lo = mid
        else:
            hi = mid
    out["breakdown"] = float(np.exp((np.log(lo) + np.log(hi)) / 2))
    return out


def _selftest():
    import os, sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import ds_stats as S

    print("=" * 62)
    print("our ladder")
    rows = S.ladder_rows()
    N = [q["params"] for q in rows]
    r = check(**from_points(N, {c: [q[c] for q in rows] for c in S.CLASSES},
                            [q["illegal_rate"] for q in rows]))
    print(r)

    print("=" * 62)
    print("Karvonen's ladder, classes nonzero at every checkpoint")
    ext = S.external_rows()
    cand = ["unreachable", "leaves_check", "to_own", "ambiguous", "malformed"]
    keep = [k for k in cand if all(e["absolute"].get(k, 0) > 0 for e in ext)]
    print(f"  kept {keep}, dropped {[k for k in cand if k not in keep]}")
    r2 = check(**from_points([e["params"] for e in ext],
                             {k: [e["absolute"][k] for e in ext] for k in keep},
                             [e["illegal_rate"] for e in ext]))
    print(r2)

    print("=" * 62)
    print("control: a decomposition that is admissible by construction")
    Ns = np.array([1e6, 1e7, 1e8], float)
    tot = 0.3 * Ns ** -0.25
    r3 = check(**from_points(Ns, {"x": 0.6 * tot, "y": 0.4 * tot}, tot))
    print(r3)
    assert not r3["diverges"], "equal exponents must not diverge"
    assert r["diverges"] and r2["diverges"]
    print("\nself-test passed")


if __name__ == "__main__":
    _selftest()
