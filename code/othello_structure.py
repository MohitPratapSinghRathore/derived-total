"""What survives in a near-perfect model? The Othello boundary test.

The differential scaling claim cannot be tested here as it was in chess. There is
no published Othello ladder -- one trained model and a random-init control -- so
there are no exponents to fit. That limit is reported rather than worked around.

What can be asked is the question the claim implies. Li's synthetic OthelloGPT is
close to perfectly legal, so its failures are the residual after training has
removed everything easy. Our chess result says the residual should be the
non-local part. Othello is the hard case for that, because its legality is
inherently non-local: a move is legal only if it flanks a line, so nothing is
decidable from the target square alone.

  occupied   the square already holds a disc. Decidable from that square alone.
  no_flank   the square is empty but flanks nothing in any of eight directions.
  pass_token the model emits the pass token where a legal move exists.

If the residual is essentially all no_flank, the picture from chess carries over
to a game with different rules: what remains after scale and training is the part
that requires integrating the whole board. If occupied is a meaningful share,
that is a boundary and it is worth reporting as one.

Batched: sequences are at most 59 tokens, so many games run in one forward.
"""
import os, json, argparse
import numpy as np
import torch
from othello import (load, random_game, legal_moves, apply_move, start_board,
                     SQ2TOK, TOK2SQ, VALID, BLACK)

RES = os.path.join(os.path.dirname(__file__), "..", "results")
CLASSES = ["occupied", "no_flank", "pass_token", "other"]


def classify(board, colour, sq, tok):
    if tok == 0:
        return "pass_token"
    if sq is None:
        return "other"
    if board[sq] != 0:
        return "occupied"
    from othello import flips
    if not flips(board, sq, colour):
        return "no_flank"
    return "other"


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=4000)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--minmove", type=int, default=5)
    ap.add_argument("--model", default="synthetic_model.pth")
    args = ap.parse_args()

    model = load(args.model)
    rng = np.random.default_rng(0)
    labels, plies, gids = [], [], []
    n_pos = 0
    buf = []

    def flush(buf, base):
        nonlocal n_pos
        if not buf:
            return
        L = max(len(b[0]) for b in buf)
        x = torch.zeros(len(buf), L, dtype=torch.long, device="cuda")
        for i, (toks, _, _) in enumerate(buf):
            x[i, :len(toks)] = torch.tensor(toks, device="cuda")
        lg = model(x)
        pred = lg.argmax(-1).cpu().numpy()
        for i, (toks, boards, colours) in enumerate(buf):
            for t in range(args.minmove, len(toks) - 1):
                b, c = boards[t + 1], colours[t + 1]
                if not legal_moves(b, c):
                    continue
                n_pos += 1
                tok = int(pred[i, t])
                sq = TOK2SQ.get(tok)
                if sq is not None and sq in legal_moves(b, c):
                    continue
                labels.append(classify(b, c, sq, tok))
                plies.append(t + 1)
                gids.append(base + i)

    for gi in range(args.games):
        moves, boards, colours = random_game(rng)
        if len(moves) < args.minmove + 3:
            continue
        buf.append(([SQ2TOK[m] for m in moves], boards, colours))
        if len(buf) >= args.batch:
            flush(buf, gi - len(buf) + 1)
            buf = []
            if gi % 512 == 0:
                print(f"  game {gi}/{args.games}  positions {n_pos:,}  "
                      f"failures {len(labels)}", flush=True)
    flush(buf, args.games - len(buf))

    labels = np.array(labels)
    plies = np.array(plies)
    rate = len(labels) / max(n_pos, 1)
    out = {"model": args.model, "n_positions": int(n_pos),
           "n_failures": int(len(labels)), "illegal_rate": float(rate),
           "counts": {c: int((labels == c).sum()) for c in CLASSES},
           "shares": {c: (float((labels == c).mean()) if len(labels) else None)
                      for c in CLASSES}}

    print(f"\n[OthelloGPT {args.model}] {n_pos:,} scored positions")
    print(f"  illegal-move rate: {rate:.6f}  ({len(labels)} failures)\n")
    print(f"{'class':>12} {'count':>7} {'share':>8} {'absolute':>11}")
    for c in CLASSES:
        n = int((labels == c).sum())
        sh = n / max(len(labels), 1)
        print(f"{c:>12} {n:7d} {sh:8.4f} {sh*rate:11.6f}")

    if len(labels):
        print(f"\n  by depth (move index):")
        for lo, hi in [(5, 20), (20, 40), (40, 59)]:
            m = (plies >= lo) & (plies < hi)
            if m.sum():
                print(f"    {lo:>2}-{hi:<2} n={int(m.sum()):4d}  "
                      + "  ".join(f"{c} {float((labels[m]==c).mean()):.3f}"
                                  for c in CLASSES[:2]))
        out["strata"] = {f"{lo}-{hi}": {
            "n": int(((plies >= lo) & (plies < hi)).sum()),
            **{c: float((labels[(plies >= lo) & (plies < hi)] == c).mean())
               for c in CLASSES}}
            for lo, hi in [(5, 20), (20, 40), (40, 59)]
            if ((plies >= lo) & (plies < hi)).sum()}

    tag = args.model.replace(".pth", "")
    json.dump(out, open(os.path.join(RES, f"othello_structure_{tag}.json"), "w"),
              indent=1)


if __name__ == "__main__":
    main()
