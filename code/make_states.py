"""Stage 2: exact per-ply board state from a rules engine.

For every prefix of every game we store 64 squares x 13 classes (empty, 6 piece
types x 2 colours), plus side-to-move and castling rights.  These are the
ground-truth labels s_t = Phi(x_1:t) of the paper's Sec. 3.
"""
import os, sys, json
import numpy as np
import chess
from multiprocessing import Pool

DATA = os.path.join(os.path.dirname(__file__), "..", "data", "proc")
PIECE2CLS = {}
for ci, col in enumerate([chess.WHITE, chess.BLACK]):
    for pi, pt in enumerate([chess.PAWN, chess.KNIGHT, chess.BISHOP,
                             chess.ROOK, chess.QUEEN, chess.KING]):
        PIECE2CLS[(pt, col)] = 1 + ci * 6 + pi     # 0 = empty

ITOS = None


def game_states(args):
    toks, ln = args
    T = int(ln)
    occ = np.zeros((T, 64), dtype=np.uint8)
    extra = np.zeros((T, 2), dtype=np.uint8)   # side-to-move, castling bits
    board = chess.Board()
    legal_mask_ok = np.ones(T, dtype=np.uint8)
    for t in range(T):
        # state AFTER consuming token t (t=0 is <bos> -> start position)
        for sq, pc in board.piece_map().items():
            occ[t, sq] = PIECE2CLS[(pc.piece_type, pc.color)]
        extra[t, 0] = int(board.turn)
        cr = (board.has_kingside_castling_rights(chess.WHITE) |
              board.has_queenside_castling_rights(chess.WHITE) << 1 |
              board.has_kingside_castling_rights(chess.BLACK) << 2 |
              board.has_queenside_castling_rights(chess.BLACK) << 3)
        extra[t, 1] = cr
        if t + 1 < T:
            uci = ITOS[toks[t + 1]]
            try:
                board.push_uci(uci)
            except Exception:
                legal_mask_ok[t + 1:] = 0
                break
    return occ, extra, legal_mask_ok


def init(itos):
    global ITOS
    ITOS = itos


def main(split):
    stoi = json.load(open(os.path.join(DATA, "vocab.json")))
    itos = {v: k for k, v in stoi.items()}
    d = np.load(os.path.join(DATA, f"{split}.npz"))
    toks, lens = d["toks"], d["lens"]
    N, L = toks.shape
    occ = np.zeros((N, L, 64), dtype=np.uint8)
    extra = np.zeros((N, L, 2), dtype=np.uint8)
    ok = np.zeros((N, L), dtype=np.uint8)
    with Pool(initializer=init, initargs=(itos,)) as p:
        for i, (o, e, m) in enumerate(
                p.imap(game_states, [(toks[j], lens[j]) for j in range(N)], chunksize=64)):
            T = len(m)
            occ[i, :T] = o
            extra[i, :T] = e
            ok[i, :T] = m
            if i % 5000 == 0:
                print(f"  {split} {i}/{N}", flush=True)
    np.save(os.path.join(DATA, f"{split}_occ.npy"), occ)
    np.save(os.path.join(DATA, f"{split}_extra.npy"), extra)
    np.save(os.path.join(DATA, f"{split}_ok.npy"), ok)
    print(split, "states", occ.shape, "replay-clean frac", ok.mean())


if __name__ == "__main__":
    for s in sys.argv[1:]:
        main(s)
