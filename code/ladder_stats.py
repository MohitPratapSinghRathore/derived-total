"""Seeded ladder statistics, shared by macros, tables and figures.

Every rung now has three seeds, so the scale trend is reported as a mean and a
spread rather than a single run. Kept in one place so the three consumers cannot
drift apart.
"""
import os, json
import numpy as np
from horizon import interp_horizon

RES = os.path.join(os.path.dirname(__file__), "..", "results")

RUNGS = [("attn_6L192", "6L/192"), ("attn_8L256", "8L/256"),
         ("attn_12L256", "12L/256"), ("attn_8L384", "8L/384"),
         ("attn_12L384", "12L/384"), ("attn_12L512", "12L/512")]


def _load(n):
    p = os.path.join(RES, f"{n}.json")
    return json.load(open(p)) if os.path.exists(p) else None


def rung_stats(base):
    """Per-rung seeded statistics, or None if the rung has no runs."""
    out = {"params": None, "n": 0, "h_ill": [], "h_fid": [], "occ40": [],
           "ill40": [], "bel": [], "bel_ratio": [], "bel_mis": [], "act_first": []}
    for s in (0, 1, 2):
        r = _load(f"{base}_s{s}")
        if not r:
            continue
        out["params"] = r["params"]
        out["n"] += 1
        out["h_ill"].append(interp_horizon(r["illegal"], r["buckets"], 0.05))
        out["occ40"].append(np.array(r["fidelity_occ"])[r["best_layer"]][4])
        out["ill40"].append(r["illegal"][4])
        a = _load(f"{base}_s{s}_auc")
        if a:
            ks = list(a["per_bucket"])
            err = [a["per_bucket"][k]["action_relevant_error"] for k in ks]
            bk = [tuple(int(x) for x in k.split("-")) for k in ks]
            out["h_fid"].append(interp_horizon(err, bk, 0.5))
            out["act_first"].append(err[0])
        b = _load(f"{base}_s{s}_belief")
        if b:
            o = b["overall"]
            out["bel"].append(o["illegal_legal_in_belief"])
            out["bel_mis"].append(o.get("mismatched_belief_legal"))
            out["bel_ratio"].append(o["illegal_legal_in_belief"] /
                                    max(o["legal_legal_in_belief"], 1e-9))
    return out if out["n"] else None


def ms(v):
    """mean, sd (sd is 0.0 when a single value)."""
    v = [x for x in v if x is not None]
    if not v:
        return None, None
    a = np.array(v, float)
    return float(a.mean()), (float(a.std(ddof=1)) if len(a) > 1 else 0.0)


def fmt(v, dp=1):
    m, sd = ms(v)
    if m is None:
        return "n/a"
    if len(v) > 1:
        return f"{m:.{dp}f}\\,$\\pm$\\,{sd:.{dp}f}"
    return f"{m:.{dp}f}"


if __name__ == "__main__":
    for base, label in RUNGS:
        st = rung_stats(base)
        if st:
            print(f"{label:9s} n={st['n']} params={st['params']:>9,} "
                  f"H_ill={ms(st['h_ill'])[0]:.1f} H_fid={ms(st['h_fid'])[0]:.1f} "
                  f"bel/ceil={ms(st['bel_ratio'])[0]:.3f}")
