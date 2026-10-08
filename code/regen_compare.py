"""What the corrected pass changed, separated into its two causes.

Two different things could move a number between the original pass and this one:
the classifier and taxonomy changed, and the arithmetic changed device and
precision. Conflating them would make the correction unauditable, so each is
measured on its own.

The classifier effect is exact. Both labels come from the same recorded move, so
the difference between them is caused by the relabelling and nothing else.

The numerical effect is estimated by comparing the *old* labels from this pass
against the shares stored by the original run. Those should agree up to
float16-versus-float32 differences in which move is top-1, and any disagreement
is the drift that porting introduced. Reporting it is the point: the alternative
is to assume a CPU rerun reproduces a GPU run, which is exactly the kind of
unchecked assumption this project keeps finding in its own history.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ds_stats as S  # noqa: E402
from structure_regen import NEW_CLASSES, OLD_CLASSES  # noqa: E402


def load_regen(name):
    p = os.path.join(S.RES, f"{name}_regen.json")
    if not os.path.exists(p):
        return None
    return json.load(open(p))


def shares(recs, key, classes):
    n = len(recs)
    return {c: sum(1 for r in recs if r[key] == c) / n for c in classes}


def main() -> int:
    names = [f"attn_{r}_s{s}" for r in S.RUNGS for s in S.SEEDS]
    rows, drift, flow = [], [], {}
    n_legal = n_tot = 0
    for nm in names:
        d = load_regen(nm)
        if not d:
            continue
        recs = d["records"]
        n_tot += len(recs)
        n_legal += d["n_truly_legal"]
        old = shares(recs, "old_label", OLD_CLASSES)
        new = shares(recs, "new_label", NEW_CLASSES)
        st = S.load(f"{nm}_structure.json")
        if st:
            for c in OLD_CLASSES:
                drift.append({"name": nm, "cls": c,
                              "stored": st["overall"][c]["mean"],
                              "regen_old": old[c],
                              "d": old[c] - st["overall"][c]["mean"]})
        for r in recs:
            flow.setdefault((r["old_label"], r["new_label"]), 0)
            flow[(r["old_label"], r["new_label"])] += 1
        rows.append({"name": nm, "old": old, "new": new,
                     "n": len(recs),
                     "policy_share": float(np.mean([r["is_policy"]
                                                    for r in recs]))})

    print(f"{n_tot:,} recorded failures across {len(rows)} runs; "
          f"{n_legal} were actually legal under the corrected gate")

    print("\nCLASS FLOW, old label -> new label (nonzero cells)")
    for (o, nw), c in sorted(flow.items(), key=lambda kv: -kv[1]):
        mark = "" if o == nw else "   <= changed"
        print(f"  {o:14s} -> {nw:20s} {c:6,d}{mark}")

    print("\nNUMERICAL DRIFT, old labels this pass vs stored shares")
    if drift:
        ad = [abs(q["d"]) for q in drift]
        print(f"  mean |delta| {np.mean(ad):.5f}, max {np.max(ad):.5f}")
        worst = max(drift, key=lambda q: abs(q["d"]))
        print(f"  worst: {worst['name']} {worst['cls']} "
              f"stored {worst['stored']:.4f} regen {worst['regen_old']:.4f}")
        bycls = {}
        for q in drift:
            bycls.setdefault(q["cls"], []).append(abs(q["d"]))
        for c, v in bycls.items():
            print(f"    {c:14s} mean |delta| {np.mean(v):.5f}")

    # The differential as the superseded pass reported it, recomputed from the
    # stored structure files rather than transcribed, so the comparison in the
    # manuscript cites a number with a provenance instead of a remembered one.
    old_diff = None
    try:
        import numpy as _np
        orows = S.ladder_rows()
        if orows:
            _x = _np.log([q["params"] for q in orows])
            _a = _np.polyfit(_x, _np.log([q["leaves_check"] for q in orows]), 1)[0]
            _b = _np.polyfit(_x, _np.log([q["from_empty"] for q in orows]), 1)[0]
            old_diff = float(_a - _b)
            print(f"superseded pass differential {old_diff:+.4f}")
    except Exception as exc:                               # noqa: BLE001
        print("could not recompute the superseded differential:", exc)

    out = {"n_failures": n_tot, "n_runs": len(rows),
           "superseded_differential": old_diff,
           "n_truly_legal": n_legal,
           "flow": {f"{o}->{nw}": c for (o, nw), c in flow.items()},
           "n_relabelled": sum(c for (o, nw), c in flow.items() if o != nw),
           "drift_mean_abs": float(np.mean([abs(q["d"]) for q in drift]))
           if drift else None,
           "drift_max_abs": float(np.max([abs(q["d"]) for q in drift]))
           if drift else None,
           "rows": rows}
    # The classes the headline rests on must be shown to be untouched, not
    # assumed to be: empty source and leaves check are the differential's two
    # endpoints, so any flow into or out of them would change the result.
    moved = {c: sum(v for (o, nw), v in flow.items()
                    if (o == c) != (nw == c))
             for c in ("from_empty", "from_opponent", "leaves_check", "to_own")}
    out["headline_class_moves"] = moved
    print("\nMOVES INTO OR OUT OF THE HEADLINE CLASSES")
    for c, v in moved.items():
        print(f"  {c:14s} {v}")
    json.dump(out, open(os.path.join(S.RES, "regen_compare.json"), "w"),
              indent=1)
    print("\nwrote results/regen_compare.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
