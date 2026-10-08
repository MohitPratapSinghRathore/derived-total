"""The temperature-only control, which the paper previously owed the reader.

An earlier version explained the metric inversion by saying that illegal-move
rate rewards edits which flatten the output distribution. That explanation cannot
be right as stated, and the argument is short enough to settle analytically:
dividing logits by a temperature is monotone, so it cannot reorder them, so it
cannot change which move is top-1. Top-1 legality is therefore exactly invariant
to temperature, and no amount of flattening produces the effect attributed to it.

This runs the control anyway, for two reasons. The invariance should be confirmed
numerically rather than asserted, because a claim of exact invariance is a claim
about the implementation as well as the mathematics. And the quantities that are
not invariant are the interesting part: total probability mass on legal moves
moves a great deal with temperature, and the forced-choice preference moves too.
So the three measures the paper uses respond differently to the same manipulation,
which is the honest version of the point the hedging story was trying to make.

The candidate pair is fixed before any manipulation, as it must be: choosing the
best legal and the top illegal move separately at each temperature would let the
pair drift and would measure the selection rather than the preference.
"""
from __future__ import annotations

import argparse
import json
import os

import chess
import numpy as np
import torch

from probes import build, load_split

DATA = os.path.join(os.path.dirname(__file__), "..", "data", "proc")
RUNS = os.path.join(os.path.dirname(__file__), "..", "runs")
RES = os.path.join(os.path.dirname(__file__), "..", "results")

TEMPS = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="attn_12L512_s0")
    ap.add_argument("--games", type=int, default=120)
    ap.add_argument("--minply", type=int, default=20)
    ap.add_argument("--device", default=None)
    a = ap.parse_args()
    dev = a.device or ("cuda" if torch.cuda.is_available() else "cpu")

    model, _, _ = build(os.path.join(RUNS, f"{a.name}.pt"), device=dev)
    stoi = json.load(open(os.path.join(DATA, "vocab.json")))
    itos = {v: k for k, v in stoi.items()}
    toks, lens, _ = load_split("eval", a.games)

    # Pre-resolve every vocabulary entry to a move once.
    uci = [itos.get(i, "") for i in range(len(stoi))]
    parsed = []
    for u in uci:
        try:
            parsed.append(chess.Move.from_uci(u))
        except Exception:                                  # noqa: BLE001
            parsed.append(None)

    agg = {t: {"top1_legal": 0, "legal_mass": [], "pref": [],
               "rank_legal": [], "rank_illegal": []} for t in TEMPS}
    n_pos = 0
    n_pairs = 0
    with torch.no_grad():
        for gi in range(len(toks)):
            T = int(lens[gi])
            if T < a.minply + 4:
                continue
            x = torch.from_numpy(toks[gi:gi + 1, :T].astype(np.int64)).to(dev)
            lg, _, _ = model(x)
            Z = lg[0].float().cpu().numpy()

            board = chess.Board()
            for t in range(T - 1):
                if t >= a.minply:
                    z = Z[t]
                    legal = set()
                    for mv in board.legal_moves:
                        j = stoi.get(mv.uci())
                        if j is not None:
                            legal.add(j)
                    if legal:
                        n_pos += 1
                        legal_idx = np.fromiter(legal, dtype=np.int64)
                        # The pair is chosen once, from the unmodified logits at
                        # temperature one, and then held fixed across all
                        # temperatures.
                        best_legal = int(legal_idx[np.argmax(z[legal_idx])])
                        mask = np.ones(len(z), dtype=bool)
                        mask[legal_idx] = False
                        ill_pool = np.where(mask)[0]
                        top_illegal = int(ill_pool[np.argmax(z[ill_pool])]) \
                            if len(ill_pool) else None
                        for temp in TEMPS:
                            zz = z / temp
                            zz = zz - zz.max()
                            p = np.exp(zz)
                            p /= p.sum()
                            if int(np.argmax(p)) in legal:
                                agg[temp]["top1_legal"] += 1
                            agg[temp]["legal_mass"].append(
                                float(p[legal_idx].sum()))
                            if top_illegal is not None:
                                d = (z[best_legal] - z[top_illegal]) / temp
                                agg[temp]["pref"].append(
                                    float(1.0 / (1.0 + np.exp(-d))))
                                order = np.argsort(-p)
                                rk = np.empty(len(p), dtype=np.int64)
                                rk[order] = np.arange(len(p))
                                agg[temp]["rank_legal"].append(
                                    int(rk[best_legal]) + 1)
                                agg[temp]["rank_illegal"].append(
                                    int(rk[top_illegal]) + 1)
                        if top_illegal is not None:
                            n_pairs += 1
                try:
                    board.push_uci(itos[int(toks[gi, t + 1])])
                except Exception:                          # noqa: BLE001
                    break

    out = {"name": a.name, "device": dev, "n_positions": n_pos,
           "n_pairs": n_pairs, "temperatures": TEMPS, "by_temp": {}}
    print(f"[{a.name}] {n_pos:,} scored positions, {n_pairs:,} with a fixed pair\n")
    print(f"{'T':>6} {'top-1 legal':>12} {'legal mass':>11} "
          f"{'pref(fixed pair)':>17} {'rank legal':>11} {'rank illegal':>13}")
    for temp in TEMPS:
        d = agg[temp]
        rec = {"top1_legal_rate": d["top1_legal"] / max(n_pos, 1),
               "legal_mass": float(np.mean(d["legal_mass"])),
               "pref": float(np.mean(d["pref"])) if d["pref"] else None,
               "rank_legal": float(np.mean(d["rank_legal"]))
               if d["rank_legal"] else None,
               "rank_illegal": float(np.mean(d["rank_illegal"]))
               if d["rank_illegal"] else None}
        out["by_temp"][str(temp)] = rec
        print(f"{temp:6.2f} {rec['top1_legal_rate']:12.6f} "
              f"{rec['legal_mass']:11.4f} {rec['pref']:17.4f} "
              f"{rec['rank_legal']:11.2f} {rec['rank_illegal']:13.2f}")

    t1 = out["by_temp"]["1.0"]["top1_legal_rate"]
    rates = [out["by_temp"][str(t)]["top1_legal_rate"] for t in TEMPS]
    out["top1_invariant"] = bool(max(abs(r - t1) for r in rates) == 0.0)
    out["top1_max_deviation"] = float(max(abs(r - t1) for r in rates))
    masses = [out["by_temp"][str(t)]["legal_mass"] for t in TEMPS]
    out["legal_mass_range"] = [float(min(masses)), float(max(masses))]
    prefs = [out["by_temp"][str(t)]["pref"] for t in TEMPS]
    out["pref_range"] = [float(min(prefs)), float(max(prefs))]
    ranks = [out["by_temp"][str(t)]["rank_legal"] for t in TEMPS]
    out["rank_legal_invariant"] = bool(max(ranks) - min(ranks) == 0.0)
    print(f"\ntop-1 legality exactly invariant: {out['top1_invariant']} "
          f"(max deviation {out['top1_max_deviation']:.2e})")
    print(f"ranks of the fixed pair invariant: {out['rank_legal_invariant']}")
    print(f"legal probability mass ranges over {out['legal_mass_range']}")
    print(f"forced-choice preference ranges over {out['pref_range']}")
    json.dump(out, open(os.path.join(RES, f"{a.name}_temperature.json"), "w"),
              indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
