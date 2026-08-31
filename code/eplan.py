"""E_plan: engine centipawn loss, CONDITIONED on the state being intact.

A move is scored only when (a) the model's top-1 move is legal and (b) the
layer-best probe recovers the exact position.  This is the conditioning that
separates H3 (memory improves) from H4 (judgement improves) -- without it the
two are confounded, which is the paper's central methodological claim.
"""
import os, json, argparse
import numpy as np
import torch
import chess, chess.engine
from probes import load_split, build
from model import ChessLM

DATA = os.path.join(os.path.dirname(__file__), "..", "data", "proc")
RUNS = os.path.join(os.path.dirname(__file__), "..", "runs")
RES = os.path.join(os.path.dirname(__file__), "..", "results")
SF = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "sf",
                                  "stockfish", "stockfish-windows-x86-64-avx2.exe"))
BUCKETS = [(10, 20), (20, 30), (30, 40), (40, 50), (50, 60), (60, 80)]
BLUNDER_CP = 100          # fixed in advance, not tuned
DEPTH = 10


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--per_bucket", type=int, default=150)
    ap.add_argument("--games", type=int, default=1500)
    args = ap.parse_args()

    model, a, c = build(os.path.join(RUNS, f"{args.name}.pt"))
    res = json.load(open(os.path.join(RES, f"{args.name}.json")))
    best = res["best_layer"]
    probes = torch.load(os.path.join(RES, f"{args.name}_probes.pt"), weights_only=False)
    probe = torch.nn.Linear(a["width"], 64 * 13).cuda()
    probe.load_state_dict(probes[best])
    probe.eval()

    toks, lens, occ = load_split("eval", args.games)
    stoi = json.load(open(os.path.join(DATA, "vocab.json")))
    itos = {v: k for k, v in stoi.items()}

    # collect candidate (fen, model_move, bucket) with state-intact conditioning
    cand = {bi: [] for bi in range(len(BUCKETS))}
    filled = 0
    with torch.no_grad():
        for i in range(0, len(toks), 32):
            if filled == len(BUCKETS):
                break
            idx = np.arange(i, min(i + 32, len(toks)))
            x = torch.from_numpy(toks[idx]).cuda()
            s = torch.from_numpy(np.asarray(occ[idx]).astype(np.int64)).cuda()
            with torch.amp.autocast("cuda", dtype=torch.float16):
                logits, hs, _ = model(x[:, :-1], return_hidden=True)
            pred = logits.float().argmax(-1).cpu().numpy()
            h = hs[best].float()
            pstate = probe(h).reshape(len(idx), -1, 64, 13).argmax(-1)
            intact = (pstate == s[:, :pstate.shape[1]]).all(-1).cpu().numpy()
            for r, gi in enumerate(idx):
                board = chess.Board()
                T = int(lens[gi])
                for t in range(T - 1):
                    bi = next((k for k, (lo, hi) in enumerate(BUCKETS) if lo <= t < hi), None)
                    if bi is not None and len(cand[bi]) < args.per_bucket and intact[r, t]:
                        mv = itos.get(int(pred[r, t]), "")
                        try:
                            m_ = chess.Move.from_uci(mv)
                            if m_ in board.legal_moves:
                                cand[bi].append((board.fen(), mv))
                        except Exception:
                            pass
                    board.push_uci(itos[int(toks[gi, t + 1])])
            filled = sum(len(v) >= args.per_bucket for v in cand.values())

    eng = chess.engine.SimpleEngine.popen_uci(SF)
    lim = chess.engine.Limit(depth=DEPTH)
    out = {}
    for bi, (lo, hi) in enumerate(BUCKETS):
        losses = []
        for fen, mv in cand[bi]:
            b = chess.Board(fen)
            try:
                best_sc = eng.analyse(b, lim)["score"].pov(b.turn).score(mate_score=10000)
                b.push_uci(mv)
                mv_sc = eng.analyse(b, lim)["score"].pov(not b.turn).score(mate_score=10000)
                losses.append(max(0, best_sc - mv_sc))
            except Exception:
                continue
        losses = np.array(losses, dtype=float)
        out[f"{lo}-{hi}"] = {
            "n": len(losses),
            "mean_cp_loss": float(losses.mean()) if len(losses) else None,
            "median_cp_loss": float(np.median(losses)) if len(losses) else None,
            "blunder_rate": float((losses >= BLUNDER_CP).mean()) if len(losses) else None,
        }
        print(f"  {lo}-{hi}: n={len(losses)} mean_cp={out[f'{lo}-{hi}']['mean_cp_loss']} "
              f"blunder={out[f'{lo}-{hi}']['blunder_rate']}", flush=True)
    eng.quit()
    json.dump({"name": args.name, "depth": DEPTH, "blunder_cp": BLUNDER_CP,
               "conditioned_on": "top1 legal AND probe-exact state", "eplan": out},
              open(os.path.join(RES, f"{args.name}_eplan.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
