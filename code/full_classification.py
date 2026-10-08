"""Classify every failure on the whole evaluation set, not the first 900.

The corrected pass recorded a fixed-size sample: the first 900 failures each model
makes while walking the evaluation games in corpus order. Multiplying those shares
by the full-set illegal rate gives an estimate of each category's full-set rate,
and that estimate rests on the first 900 failures being representative of all of
them. Corpus order is not random, so the assumption is not free: games are read in
file order, and a model that fails more often early in the corpus reaches its 900th
failure sooner and is then described by a narrower slice of games.

Documenting the assumption is weaker than removing it, and removing it is cheap
here. The forward passes are needed either way; only the per-failure classification
is extra, and it is pure rules-engine work. So this classifies every failure at
every scored position.

What it stores is aggregate rather than per-move, because the per-move records for
the whole set would run to hundreds of megabytes. The counts kept are the ones the
manuscript uses: class totals, the old-to-new label crosstab, the split by whether
local belief was correctly decoded, and the split by depth band. The 900-failure
records stay on disk as the audit trail for the relabelling, since that is what
they were for.
"""
from __future__ import annotations

import argparse
import json
import os

import chess
import numpy as np
import torch

from probes import build, load_split
from structure_regen import NEW_CLASSES, OLD_CLASSES, classify_new, classify_old

DATA = os.path.join(os.path.dirname(__file__), "..", "data", "proc")
RUNS = os.path.join(os.path.dirname(__file__), "..", "runs")
RES = os.path.join(os.path.dirname(__file__), "..", "results")

RUNGS = ("6L192", "8L256", "12L256", "8L384", "12L384", "12L512")
SEEDS = (0, 1, 2)
BANDS = (("20-40", 20, 40), ("40-60", 40, 60), ("60+", 60, 10 ** 6))


def one(name, toks, lens, occ, itos, parsed, dev, minply):
    model, arch, _ = build(os.path.join(RUNS, f"{name}.pt"), device=dev)
    bl = json.load(open(os.path.join(RES, f"{name}.json")))["best_layer"]
    pls = torch.load(os.path.join(RES, f"{name}_probes.pt"),
                     map_location=dev, weights_only=False)
    probe = torch.nn.Linear(arch["width"], 64 * 13).to(dev)
    probe.load_state_dict(pls[bl])
    probe.eval()

    n_pos = 0
    counts = {c: 0 for c in NEW_CLASSES}
    old_counts = {c: 0 for c in OLD_CLASSES}
    flow: dict[str, int] = {}
    pol = {c: 0 for c in NEW_CLASSES}
    n_pol = 0
    band = {b: {c: 0 for c in NEW_CLASSES} for b, _, _ in BANDS}
    band_n = {b: 0 for b, _, _ in BANDS}
    n_legal_in_set = 0

    with torch.no_grad():
        for gi in range(len(toks)):
            T = int(lens[gi])
            if T < minply + 4:
                continue
            x = torch.from_numpy(toks[gi:gi + 1, :T].astype(np.int64)).to(dev)
            lg, hs, _ = model(x, return_hidden=True)
            top1 = lg[0].argmax(-1).cpu().numpy()
            bel_all = None
            board = chess.Board()
            for t in range(T - 1):
                if t >= minply:
                    n_pos += 1
                    mv = parsed.get(int(top1[t]))
                    if mv is not None and not board.is_legal(mv):
                        if bel_all is None:
                            bel_all = probe(hs[bl][0].float()) \
                                .reshape(T, 64, 13).argmax(-1).cpu().numpy()
                        c = classify_new(board, mv)
                        o = classify_old(board, mv)
                        counts[c] += 1
                        old_counts[o] += 1
                        k = o + "->" + c
                        flow[k] = flow.get(k, 0) + 1
                        if mv in board.legal_moves:
                            n_legal_in_set += 1
                        truth = np.asarray(occ[gi, t]).astype(int)
                        bel = bel_all[t]
                        if all(bel[sq] == truth[sq]
                               for sq in (mv.from_square, mv.to_square)):
                            pol[c] += 1
                            n_pol += 1
                        for b, lo, hi in BANDS:
                            if lo <= t < hi:
                                band[b][c] += 1
                                band_n[b] += 1
                                break
                try:
                    board.push_uci(itos[int(toks[gi, t + 1])])
                except Exception:                          # noqa: BLE001
                    break

    n_fail = sum(counts.values())
    return {"name": name, "n_positions": n_pos, "n_failures": n_fail,
            "illegal_rate": n_fail / max(n_pos, 1),
            "counts": counts, "old_counts": old_counts, "flow": flow,
            "policy_counts": pol, "n_policy": n_pol,
            "policy_share": n_pol / max(n_fail, 1),
            "band_counts": band, "band_n": band_n,
            "n_membership_disagreements": n_legal_in_set}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=600)
    ap.add_argument("--minply", type=int, default=20)
    ap.add_argument("--device", default=None)
    a = ap.parse_args()
    dev = a.device or ("cuda" if torch.cuda.is_available() else "cpu")

    stoi = json.load(open(os.path.join(DATA, "vocab.json")))
    itos = {v: k for k, v in stoi.items()}
    toks, lens, occ = load_split("eval", a.games)
    parsed = {}
    for i, u in itos.items():
        try:
            parsed[i] = chess.Move.from_uci(u)
        except Exception:                                  # noqa: BLE001
            parsed[i] = None

    out = {"games": a.games, "minply": a.minply, "device": dev, "rows": []}
    for r in RUNGS:
        for sd in SEEDS:
            name = f"attn_{r}_s{sd}"
            if not os.path.exists(os.path.join(RUNS, f"{name}.pt")):
                continue
            rec = one(name, toks, lens, occ, itos, parsed, dev, a.minply)
            rec["rung"] = r
            rec["seed"] = sd
            out["rows"].append(rec)
            print(f"{name:16s} pos {rec['n_positions']:7,d}  "
                  f"failures {rec['n_failures']:6,d}  "
                  f"rate {rec['illegal_rate']:.5f}  "
                  f"check {rec['counts']['leaves_check'] / rec['n_failures']:.3f}",
                  flush=True)

    if out["rows"]:
        out["total_failures"] = sum(r["n_failures"] for r in out["rows"])
        out["total_positions"] = sum(r["n_positions"] for r in out["rows"])
        out["total_membership_disagreements"] = sum(
            r["n_membership_disagreements"] for r in out["rows"])
        print(f"\n{out['total_failures']:,d} failures classified over "
              f"{out['total_positions']:,d} positions")
        print(f"legal moves inside the failure set: "
              f"{out['total_membership_disagreements']}")
    json.dump(out, open(os.path.join(RES, "full_classification.json"), "w"),
              indent=1)
    print("wrote results/full_classification.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
