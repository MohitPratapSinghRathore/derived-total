"""Is the aggregate illegal-move rate affected by the membership-gate defect?

The regenerated pass established that none of the 900 recorded failures per model
was actually legal. That is a statement about a fixed-size sample of failures, not
about the whole evaluation set, and the aggregate rate is computed over every
scored position. So the question the review raises cannot be settled by inference
from the sample: it has to be measured on the full set.

This scores every position of the evaluation split under both gates, the
membership test the original pass used and the direct legality test, and reports
the two rates. No probe is loaded, because only the move matters here.

If the two rates agree, the aggregate rate and everything built on it are
unaffected, and the paper can say so about the whole set rather than a sample.
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=600)
    ap.add_argument("--minply", type=int, default=20)
    ap.add_argument("--device", default=None)
    a = ap.parse_args()
    dev = a.device or ("cuda" if torch.cuda.is_available() else "cpu")

    stoi = json.load(open(os.path.join(DATA, "vocab.json")))
    itos = {v: k for k, v in stoi.items()}
    toks, lens, _ = load_split("eval", a.games)
    parsed = {}
    for i, u in itos.items():
        try:
            parsed[i] = chess.Move.from_uci(u)
        except Exception:                                  # noqa: BLE001
            parsed[i] = None

    out = {"games": a.games, "minply": a.minply, "device": dev, "rows": []}
    for rung in ("6L192", "8L256", "12L256", "8L384", "12L384", "12L512"):
        for sd in (0, 1, 2):
            name = f"attn_{rung}_s{sd}"
            p = os.path.join(RUNS, f"{name}.pt")
            if not os.path.exists(p):
                continue
            model, _, _ = build(p, device=dev)
            n_pos = n_memb = n_legal = n_disagree = 0
            with torch.no_grad():
                for gi in range(len(toks)):
                    T = int(lens[gi])
                    if T < a.minply + 4:
                        continue
                    x = torch.from_numpy(
                        toks[gi:gi + 1, :T].astype(np.int64)).to(dev)
                    lg, _, _ = model(x)
                    top1 = lg[0].argmax(-1).cpu().numpy()
                    board = chess.Board()
                    for t in range(T - 1):
                        if t >= a.minply:
                            mv = parsed.get(int(top1[t]))
                            n_pos += 1
                            if mv is None:
                                n_memb += 1
                                n_legal += 1
                                continue
                            in_gen = mv in board.legal_moves
                            try:
                                truly = board.is_legal(mv)
                            except Exception:              # noqa: BLE001
                                truly = False
                            n_memb += int(not in_gen)
                            n_legal += int(not truly)
                            n_disagree += int(in_gen != truly)
                        try:
                            board.push_uci(itos[int(toks[gi, t + 1])])
                        except Exception:                  # noqa: BLE001
                            break
            r = {"name": name, "n_positions": n_pos,
                 "rate_membership_gate": n_memb / max(n_pos, 1),
                 "rate_legality_gate": n_legal / max(n_pos, 1),
                 "n_gates_disagree": n_disagree}
            r["delta"] = r["rate_membership_gate"] - r["rate_legality_gate"]
            out["rows"].append(r)
            print(f"{name:16s} n={n_pos:7,d}  membership {r['rate_membership_gate']:.6f}"
                  f"  legality {r['rate_legality_gate']:.6f}"
                  f"  disagree {n_disagree}", flush=True)

    if out["rows"]:
        out["max_abs_delta"] = max(abs(r["delta"]) for r in out["rows"])
        out["total_disagreements"] = sum(r["n_gates_disagree"]
                                         for r in out["rows"])
        out["aggregate_unaffected"] = bool(out["total_disagreements"] == 0)
        print(f"\ntotal positions where the two gates disagree: "
              f"{out['total_disagreements']}")
        print(f"largest absolute difference in rate: {out['max_abs_delta']:.2e}")
    json.dump(out, open(os.path.join(RES, "aggregate_gate_check.json"), "w"),
              indent=1)
    print("wrote results/aggregate_gate_check.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
