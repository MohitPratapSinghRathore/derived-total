"""Interpolated horizon.

The bucketed horizon returns the first bucket where the illegal-move rate crosses
a threshold, which on these models is the same bucket for every condition and so
discriminates nothing. Interpolating the crossing point between bucket midpoints
gives a continuous quantity that does. The underlying curve is unchanged; only
the read-off is finer.
"""
import numpy as np


def interp_horizon(rates, buckets, thr=0.05):
    """Ply at which a rising curve first crosses thr, linearly interpolated."""
    mids = [(lo + hi) / 2 for lo, hi in buckets]
    r = np.asarray(rates, dtype=float)
    for i in range(len(r)):
        if r[i] > thr:
            if i == 0:
                return float(mids[0])
            x0, x1 = mids[i - 1], mids[i]
            y0, y1 = r[i - 1], r[i]
            if y1 == y0:
                return float(x1)
            return float(x0 + (thr - y0) * (x1 - x0) / (y1 - y0))
    return float(mids[-1])
