"""Two robustness checks the reviewer asked for, and one audit.

**Leave one rung out.** Six sizes is not many for a slope, and a resampling
interval over rungs does not show whether one rung carries the result. Refitting
with each rung removed in turn does. A differential that survives every deletion
is a different object from one that depends on the top or bottom point.

**Fit uncertainty on the external ladder.** Three checkpoints admit a slope but
barely constrain it, and the paper previously said no uncertainty could be
estimated at all. Leave one out there too, which with three points means three
two-point fits, and report the spread as what it is: poorly determined.

**The own-destination audit.** The locality measurement reported a maximum
dependency count of 48 for the own-destination class, which should be decidable
from one square. A maximum that large means the counterfactual is doing something
other than what the definition says, and the reviewer was right to ask. This finds
the cases and explains them.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ds_stats as S  # noqa: E402


def slope(rows, key):
    x = np.log([q["params"] for q in rows])
    y = np.log([max(q[key], 1e-12) for q in rows])
    return float(np.polyfit(x, y, 1)[0]) if len(set(x)) > 1 else float("nan")


def differential(rows, a, b):
    return slope(rows, a) - slope(rows, b)


def leave_one_rung_out(rows):
    """Refit with each size removed. The headline is the paired differential."""
    out = {"full": {"diff": differential(rows, "leaves_check", "from_empty"),
                    "check": slope(rows, "leaves_check"),
                    "empty": slope(rows, "from_empty")},
           "dropped": {}}
    for r in S.RUNGS:
        sub = [q for q in rows if q["rung"] != r]
        if len({q["rung"] for q in sub}) < 3:
            continue
        out["dropped"][r] = {
            "diff": differential(sub, "leaves_check", "from_empty"),
            "check": slope(sub, "leaves_check"),
            "empty": slope(sub, "from_empty"),
            "n_rungs": len({q["rung"] for q in sub})}
    d = [v["diff"] for v in out["dropped"].values()]
    out["diff_min"], out["diff_max"] = float(min(d)), float(max(d))
    out["sign_stable"] = bool(all(x > 0 for x in d) or all(x < 0 for x in d))
    out["max_shift"] = float(max(abs(x - out["full"]["diff"]) for x in d))
    return out


def external_leave_one_out():
    """Three checkpoints, so three two-point fits. Spread, not a confidence interval."""
    ext = S.external_rows()
    if len(ext) < 3:
        return {}
    keys = ("unreachable", "leaves_check")
    out = {"n_points": len(ext)}
    full = {}
    x = np.log([r["params"] for r in ext])
    for k in keys:
        y = np.log([max(r["absolute"][k], 1e-12) for r in ext])
        full[k] = float(np.polyfit(x, y, 1)[0])
    out["full_diff"] = full["leaves_check"] - full["unreachable"]
    diffs = []
    for i in range(len(ext)):
        sub = [r for j, r in enumerate(ext) if j != i]
        xs = np.log([r["params"] for r in sub])
        sl = {}
        for k in keys:
            ys = np.log([max(r["absolute"][k], 1e-12) for r in sub])
            sl[k] = float(np.polyfit(xs, ys, 1)[0])
        diffs.append(sl["leaves_check"] - sl["unreachable"])
    out["loo_diffs"] = diffs
    out["loo_min"], out["loo_max"] = float(min(diffs)), float(max(diffs))
    out["sign_stable"] = bool(all(d > 0 for d in diffs) or all(d < 0 for d in diffs))
    return out


def own_destination_audit(n_games=40):
    """Why does a one-square class ever report 48 dependencies?

    Hypothesis worth testing rather than asserting: the counterfactual places a
    piece on an arbitrary square, and some placements are illegal positions, for
    instance leaving a side without a king or putting the side to move in an
    impossible check. python-chess will still answer is_legal on such a board, so
    a nonsense position can flip the verdict and be counted as a dependency.
    """
    import random

    import chess
    import chess.pgn
    sys.path.insert(0, HERE)
    from locality_index import dependency_size, illegal_moves_of_each_class

    pgn = os.path.join(HERE, "..", "data", "_sample.pgn")
    if not os.path.exists(pgn):
        return {"skipped": "no sample pgn"}
    rng = random.Random(11)
    big, total, kingless = [], 0, 0
    with open(pgn, encoding="utf-8", errors="ignore") as fh:
        g_i = 0
        while g_i < n_games:
            g = chess.pgn.read_game(fh)
            if g is None:
                break
            pl = list(g.mainline_moves())
            if len(pl) < 30:
                continue
            g_i += 1
            board = g.board()
            for i, mv in enumerate(pl):
                if i in (24, 34):
                    found = illegal_moves_of_each_class(board, rng, 2)
                    for m in found.get("to_own", []):
                        k = dependency_size(board.copy(), m)
                        total += 1
                        if k > 2:
                            big.append(k)
                            # does the flip require an illegal board?
                            b = board.copy()
                            b.remove_piece_at(m.to_square)
                            if b.king(chess.WHITE) is None or \
                                    b.king(chess.BLACK) is None:
                                kingless += 1
                board.push(mv)
    return {"n_checked": total, "n_with_k_gt_2": len(big),
            "share_gt_2": (len(big) / total) if total else 0.0,
            "max_k": max(big) if big else 0,
            "flips_needing_kingless_board": kingless}


def main() -> int:
    rows = S.ladder_rows()
    out = {}
    print("LEAVE ONE RUNG OUT, paired differential check minus empty")
    loro = leave_one_rung_out(rows)
    out["leave_one_rung_out"] = loro
    print(f"  full ladder            {loro['full']['diff']:+.4f}")
    for r, v in loro["dropped"].items():
        print(f"  without {r:8s} ({v['n_rungs']} rungs)  {v['diff']:+.4f}")
    print(f"  range [{loro['diff_min']:+.4f}, {loro['diff_max']:+.4f}], "
          f"largest shift {loro['max_shift']:.4f}, sign stable "
          f"{loro['sign_stable']}")

    print("\nEXTERNAL LADDER, leave one of three out")
    ext = external_leave_one_out()
    out["external_loo"] = ext
    if ext:
        print(f"  full {ext['full_diff']:+.4f}; two-point fits "
              f"{[round(d, 3) for d in ext['loo_diffs']]}")
        print(f"  range [{ext['loo_min']:+.4f}, {ext['loo_max']:+.4f}], "
              f"sign stable {ext['sign_stable']}")

    print("\nOWN-DESTINATION DEPENDENCY AUDIT")
    aud = own_destination_audit()
    out["own_destination_audit"] = aud
    print(" ", aud)

    json.dump(out, open(os.path.join(S.RES, "robustness_checks.json"), "w"),
              indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
