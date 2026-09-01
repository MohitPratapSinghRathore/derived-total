"""Planning error as a function of how strictly we condition on state.

The decomposition has a measurement tension we should not paper over. Requiring
the probe to recover the EXACT board isolates judgement cleanly but yields no
samples past the opening. Requiring only the squares the move touches yields
samples everywhere but leaves the rest of the board unconstrained, so the
resulting "planning error" still contains errors caused by state loss elsewhere.

This measures the trend between those poles. For each tolerance k we score only
moves that are legal, whose touched squares are correct, and where at most k of
the remaining squares are wrong. If centipawn loss falls as k tightens, the
residual error is state-driven rather than a failure of judgement, and the size
of the fall says how much of the apparent planning error is really memory.
"""
import os, json, argparse
import numpy as np
import torch
import chess, chess.engine
from probes import load_split, build

DATA = os.path.join(os.path.dirname(__file__), "..", "data", "proc")
RUNS = os.path.join(os.path.dirname(__file__), "..", "runs")
RES = os.path.join(os.path.dirname(__file__), "..", "results")
SF = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "sf",
                                  "stockfish", "stockfish-windows-x86-64-avx2.exe"))
TOLS = [0, 2, 4, 8, 16, 64]          # max wrong squares away from the move
BUCKETS = [(20, 30), (30, 40), (40, 50)]
DEPTH = 10
BLUNDER_CP = 100


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="attn_8L256_s0")
    ap.add_argument("--per_cell", type=int, default=90)
    ap.add_argument("--games", type=int, default=4000)
    args = ap.parse_args()

    model, a, c = build(os.path.join(RUNS, f"{args.name}.pt"))
    res = json.load(open(os.path.join(RES, f"{args.name}.json")))
    best = res["best_layer"]
    pls = torch.load(os.path.join(RES, f"{args.name}_probes.pt"), weights_only=False)
    probe = torch.nn.Linear(a["width"], 64 * 13).cuda()
    probe.load_state_dict(pls[best]); probe.eval()

    toks, lens, occ = load_split("eval", args.games)
    stoi = json.load(open(os.path.join(DATA, "vocab.json")))
    itos = {v: k for k, v in stoi.items()}

    cells = {(bi, t): [] for bi in range(len(BUCKETS)) for t in TOLS}
    with torch.no_grad():
        for i in range(0, len(toks), 32):
            if all(len(v) >= args.per_cell for v in cells.values()):
                break
            idx = np.arange(i, min(i + 32, len(toks)))
            x = torch.from_numpy(toks[idx]).cuda()
            s = torch.from_numpy(np.asarray(occ[idx]).astype(np.int64)).cuda()
            with torch.amp.autocast("cuda", dtype=torch.float16):
                logits, hs, _ = model(x[:, :-1], return_hidden=True)
            pred = logits.float().argmax(-1).cpu().numpy()
            pstate = probe(hs[best].float()).reshape(len(idx), -1, 64, 13).argmax(-1)
            ok = (pstate == s[:, :pstate.shape[1]]).cpu().numpy()
            for r, gi in enumerate(idx):
                board = chess.Board()
                T = int(lens[gi])
                for t in range(T - 1):
                    bi = next((k for k, (lo, hi) in enumerate(BUCKETS) if lo <= t < hi), None)
                    if bi is not None:
                        mv = itos.get(int(pred[r, t]), "")
                        try:
                            m_ = chess.Move.from_uci(mv)
                            legal = m_ in board.legal_moves
                        except Exception:
                            m_, legal = None, False
                        if legal and ok[r, t, m_.from_square] and ok[r, t, m_.to_square]:
                            elsewhere = int((~ok[r, t]).sum())
                            elsewhere -= int(not ok[r, t, m_.from_square])
                            elsewhere -= int(not ok[r, t, m_.to_square])
                            for tol in TOLS:
                                if elsewhere <= tol and len(cells[(bi, tol)]) < args.per_cell:
                                    cells[(bi, tol)].append((board.fen(), mv))
                    board.push_uci(itos[int(toks[gi, t + 1])])

    eng = chess.engine.SimpleEngine.popen_uci(SF)
    lim = chess.engine.Limit(depth=DEPTH)
    out = {}
    for (bi, tol), items in sorted(cells.items()):
        losses = []
        for fen, mv in items:
            b = chess.Board(fen)
            try:
                bs = eng.analyse(b, lim)["score"].pov(b.turn).score(mate_score=10000)
                b.push_uci(mv)
                ms = eng.analyse(b, lim)["score"].pov(not b.turn).score(mate_score=10000)
                losses.append(max(0, bs - ms))
            except Exception:
                continue
        lo, hi = BUCKETS[bi]
        key = f"{lo}-{hi}|tol{tol}"
        L = np.array(losses, float)
        out[key] = {"n": len(L),
                    "mean_cp": float(L.mean()) if len(L) else None,
                    "median_cp": float(np.median(L)) if len(L) else None,
                    "blunder_rate": float((L >= BLUNDER_CP).mean()) if len(L) else None}
    eng.quit()
    json.dump({"name": args.name, "tolerances": TOLS, "buckets": BUCKETS,
               "cells": out}, open(os.path.join(RES, f"{args.name}_eplanstrict.json"),
                                   "w"), indent=1)

    print(f"[{args.name}] planning error against conditioning strictness")
    print(f"{'ply':>8} " + " ".join(f"tol{t:<2d}(n)   " for t in TOLS))
    for bi, (lo, hi) in enumerate(BUCKETS):
        row = f"{str(lo)+'-'+str(hi):>8} "
        for t in TOLS:
            cell = out[f"{lo}-{hi}|tol{t}"]
            v = "n/a" if cell["mean_cp"] is None else f"{cell['mean_cp']:.0f}"
            row += f"{v:>5}({cell['n']:>3})  "
        print(row)


if __name__ == "__main__":
    main()
