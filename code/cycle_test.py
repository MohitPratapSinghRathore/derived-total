"""Is it transcript length, or is it position difficulty?

The paper reads the decay curves as state loss caused by longer composition. A
position at ply 100 differs from one at ply 10 in material, phase, legal-move
count and familiarity, so depth is confounded with position difficulty and the
causal reading is not licensed by the curves alone.

This removes the confound. Take a position, then insert a reversible cycle of
legal moves that returns to exactly the same position: a knight out and back for
each side leaves piece placement, side to move and castling rights unchanged
while adding four plies of transcript. Probe fidelity is then measured on the
identical state at two different transcript lengths.

If fidelity falls across the cycle, the loss is attributable to transcript length
rather than to the position being harder. Repeating the cycle gives a dose
response.

The one caveat we cannot remove: the inserted moves are legal but unlike training
data, so part of any drop may be distribution shift rather than length. The
random-play evaluation bounds that concern separately, and we report both.
"""
import os, json, argparse
import numpy as np
import torch
import chess
from probes import load_split, build

DATA = os.path.join(os.path.dirname(__file__), "..", "data", "proc")
RUNS = os.path.join(os.path.dirname(__file__), "..", "runs")
RES = os.path.join(os.path.dirname(__file__), "..", "results")


def find_cycle(board):
    """A four-ply sequence returning to the same position: knight out and back
    for each side. Returns move ucis, or None if unavailable."""
    for w in board.legal_moves:
        if board.piece_at(w.from_square).piece_type != chess.KNIGHT:
            continue
        if board.is_capture(w) or w.promotion:
            continue
        b1 = board.copy(); b1.push(w)
        for bl in b1.legal_moves:
            if b1.piece_at(bl.from_square).piece_type != chess.KNIGHT:
                continue
            if b1.is_capture(bl) or bl.promotion:
                continue
            b2 = b1.copy(); b2.push(bl)
            back_w = chess.Move(w.to_square, w.from_square)
            if back_w not in b2.legal_moves:
                continue
            b3 = b2.copy(); b3.push(back_w)
            back_b = chess.Move(bl.to_square, bl.from_square)
            if back_b not in b3.legal_moves:
                continue
            b4 = b3.copy(); b4.push(back_b)
            if b4.board_fen() == board.board_fen() and b4.turn == board.turn:
                return [w.uci(), bl.uci(), back_w.uci(), back_b.uci()]
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="attn_8L256_s0")
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--cycles", type=int, default=3)
    args = ap.parse_args()

    model, a, c = build(os.path.join(RUNS, f"{args.name}.pt"))
    res = json.load(open(os.path.join(RES, f"{args.name}.json")))
    best = res["best_layer"]
    pls = torch.load(os.path.join(RES, f"{args.name}_probes.pt"), weights_only=False)
    probe = torch.nn.Linear(a["width"], 64 * 13).cuda()
    probe.load_state_dict(pls[best]); probe.eval()

    toks, lens, occ = load_split("eval", 3000)
    stoi = json.load(open(os.path.join(DATA, "vocab.json")))
    itos = {v: k for k, v in stoi.items()}
    rng = np.random.default_rng(0)
    maxlen = toks.shape[1]

    occ_acc = [[] for _ in range(args.cycles + 1)]
    ill = [[] for _ in range(args.cycles + 1)]
    games = []
    tot = 0

    with torch.no_grad():
        for gi in range(len(toks)):
            if tot >= args.n:
                break
            T = int(lens[gi])
            if T < 30:
                continue
            t = int(rng.integers(15, min(T - 2, 45)))
            board = chess.Board()
            for u in range(1, t + 1):
                board.push_uci(itos[int(toks[gi, u])])
            cyc = find_cycle(board)
            if cyc is None or any(m not in stoi for m in cyc):
                continue
            truth = np.asarray(occ[gi, t]).astype(np.int64)
            base = [int(v) for v in toks[gi, :t + 1]]

            ok_this, ill_this = [], []
            for k in range(args.cycles + 1):
                seq = base + [stoi[m] for m in cyc] * k
                if len(seq) >= maxlen:
                    break
                x = torch.tensor(seq, dtype=torch.long).unsqueeze(0).cuda()
                with torch.amp.autocast("cuda", dtype=torch.float16):
                    lg, hs, _ = model(x, return_hidden=True)
                dec = probe(hs[best][0, -1].float()).reshape(64, 13).argmax(-1)
                dec = dec.cpu().numpy()
                occupied = truth > 0
                ok_this.append(float((dec[occupied] == truth[occupied]).mean()))
                mv = itos.get(int(lg.float()[0, -1].argmax()), "")
                try:
                    legal = chess.Move.from_uci(mv) in board.legal_moves
                except Exception:
                    legal = False
                ill_this.append(0.0 if legal else 1.0)
            if len(ok_this) != args.cycles + 1:
                continue
            for k in range(args.cycles + 1):
                occ_acc[k].append(ok_this[k]); ill[k].append(ill_this[k])
            games.append(gi)
            tot += 1

    def ci(v, g, n_boot=400):
        v = np.asarray(v, float); g = np.asarray(g)
        uniq = np.unique(g); by = {u: np.where(g == u)[0] for u in uniq}
        rng2 = np.random.default_rng(0); out = []
        for _ in range(n_boot):
            pick = rng2.choice(uniq, len(uniq), replace=True)
            sel = np.concatenate([by[u] for u in pick])
            out.append(v[sel].mean())
        return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))

    out = {"name": args.name, "n": tot, "cycles": args.cycles, "per_cycle": {}}
    print(f"[{args.name}] reversible-cycle test, n={tot} positions")
    print(f"{'added plies':>12} {'occ acc':>9} {'95% CI':>18} {'illegal':>9}")
    for k in range(args.cycles + 1):
        lo, hi = ci(occ_acc[k], games)
        m = float(np.mean(occ_acc[k])); mi = float(np.mean(ill[k]))
        out["per_cycle"][str(4 * k)] = {"occ_acc": m, "ci_lo": lo, "ci_hi": hi,
                                        "illegal": mi, "n": tot}
        print(f"{4*k:12d} {m:9.3f}   [{lo:.3f}, {hi:.3f}] {mi:9.3f}")
    json.dump(out, open(os.path.join(RES, f"{args.name}_cycles.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
