"""Align board state to character positions in Chess-GPT's PGN stream.

Our UCI models give one token per move, so position t carries the board after t
moves and the model predicts move t+1 there. Chess-GPT spreads a move over
several characters, so the equivalent site has to be found rather than indexed.

The site is the last character of the prefix that precedes a move: the "." after
a move number for White, or the space after White's move for Black. At that
character the model is about to emit the first character of the move, and the
board it should hold is the position that move is played from. That reproduces
the UCI convention exactly, which matters because every downstream comparison
against the ladder assumes the two are measuring the same thing.

An off-by-one here would silently corrupt every number built on it and would
still look plausible, so verify() checks each recorded site three ways: the
recorded board must equal an independent replay, the prefix text must parse to
the same board, and the character at the site must be one of the two expected
boundary characters.
"""
import os, json
import numpy as np
import chess

ROOT = os.path.join(os.path.dirname(__file__), "..")
DATA = os.path.join(ROOT, "data", "proc")

PIECE2CLS = {}
for ci, col in enumerate([chess.WHITE, chess.BLACK]):
    for pi, pt in enumerate([chess.PAWN, chess.KNIGHT, chess.BISHOP,
                             chess.ROOK, chess.QUEEN, chess.KING]):
        PIECE2CLS[(pt, col)] = 1 + ci * 6 + pi      # 0 = empty


def occupancy(board):
    occ = np.zeros(64, dtype=np.int64)
    for sq, pc in board.piece_map().items():
        occ[sq] = PIECE2CLS[(pc.piece_type, pc.color)]
    return occ


def build(uci_moves, max_plies=None):
    """Return (text, sites) where each site is a probe position.

    site = dict(char_index, ply, san, occ, fen, turn)
      char_index  index of the LAST character before the move begins
      occ         board the move is played FROM, as 64 class ids
    """
    b = chess.Board()
    text = ";"
    sites = []
    for i, u in enumerate(uci_moves):
        if max_plies is not None and i >= max_plies:
            break
        try:
            mv = chess.Move.from_uci(u)
            if mv not in b.legal_moves:
                break
            san = b.san(mv)
        except Exception:
            break
        if i % 2 == 0:
            text += f"{i // 2 + 1}."
        # the model is about to emit this move's first character
        sites.append({"char_index": len(text) - 1, "ply": i, "san": san,
                      "occ": occupancy(b), "fen": b.fen(), "turn": int(b.turn)})
        text += san + " "
        b.push(mv)
    return text, sites


def verify(uci_moves, text, sites):
    """Three independent checks that the alignment is right."""
    problems = []
    for s in sites:
        # 1. recorded board equals an independent replay of the first `ply` moves
        b = chess.Board()
        for u in uci_moves[:s["ply"]]:
            b.push(chess.Move.from_uci(u))
        if not np.array_equal(occupancy(b), s["occ"]):
            problems.append((s["ply"], "occ mismatch vs replay"))
            continue
        if b.fen() != s["fen"]:
            problems.append((s["ply"], "fen mismatch vs replay"))
            continue
        # 2. the prefix text itself must parse to the same board
        prefix = text[:s["char_index"] + 1]
        bb = chess.Board()
        try:
            for tok in prefix[1:].replace(".", " ").split():
                if tok.isdigit():
                    continue
                bb.push_san(tok)
        except Exception as e:
            problems.append((s["ply"], f"prefix unparseable: {e}"))
            continue
        if bb.fen() != s["fen"]:
            problems.append((s["ply"], "prefix parses to a different board"))
            continue
        # 3. the site character must be a move boundary
        ch = text[s["char_index"]]
        if ch not in (".", " "):
            problems.append((s["ply"], f"site char is {ch!r}, not a boundary"))
        expect = "." if s["ply"] % 2 == 0 else " "
        if ch != expect:
            problems.append((s["ply"], f"site char {ch!r} but expected {expect!r}"))
    return problems


def load_games(split="eval", n=200):
    d = np.load(os.path.join(DATA, f"{split}.npz"))
    toks, lens = d["toks"].astype(np.int64), d["lens"].astype(np.int64)
    stoi = json.load(open(os.path.join(DATA, "vocab.json")))
    itos = {v: k for k, v in stoi.items()}
    out = []
    for gi in range(min(n, len(toks))):
        T = int(lens[gi])
        out.append([itos[int(t)] for t in toks[gi, 1:T]])
    return out


def random_legal_games(n, rng, max_plies=120, min_plies=30):
    """Uniformly-random legal play.

    Chess-GPT plays human games almost perfectly -- roughly one illegal move in
    a thousand positions -- so an illegal-move population large enough to measure
    belief consistency on does not fall out of held-out human games at any
    affordable scale. Random legal play walks the model into positions unlike its
    training distribution and lifts the illegal rate by orders of magnitude.

    This mirrors the uniformly-random-legal-play reference used in the main
    paper, where the coherence rate was shown not to shift between the human and
    random distributions. Both distributions are reported here for the same
    reason: the shift in illegal RATE is the point, and the claim is that the
    conditional structure given a failure is what stays put.
    """
    out = []
    for _ in range(n):
        b = chess.Board()
        moves = []
        while len(moves) < max_plies:
            lm = list(b.legal_moves)
            if not lm:
                break
            mv = lm[int(rng.integers(len(lm)))]
            moves.append(mv.uci())
            b.push(mv)
        if len(moves) >= min_plies:
            out.append(moves)
    return out


def main():
    games = load_games("eval", 60)
    print(f"checking alignment on {len(games)} games\n")
    tot_sites = 0
    tot_problems = 0
    for gi, g in enumerate(games):
        text, sites = build(g, max_plies=80)
        probs = verify(g, text, sites)
        tot_sites += len(sites)
        tot_problems += len(probs)
        if probs and gi < 3:
            print(f"  game {gi}: {len(probs)} problems, first: {probs[0]}")
    print(f"  sites checked : {tot_sites:,}")
    print(f"  problems      : {tot_problems}")
    print("  " + ("ALIGNMENT VERIFIED" if tot_problems == 0
                  else "ALIGNMENT BROKEN -- do not proceed"))

    text, sites = build(games[0], max_plies=12)
    print(f"\nexample stream:\n  {text!r}")
    print(f"\n{'ply':>4} {'idx':>5} {'char':>6} {'san':>7}  {'to move':>8}")
    for s in sites[:8]:
        print(f"{s['ply']:>4} {s['char_index']:>5} "
              f"{text[s['char_index']]!r:>6} {s['san']:>7}  "
              f"{'white' if s['turn'] else 'black':>8}")


if __name__ == "__main__":
    main()
