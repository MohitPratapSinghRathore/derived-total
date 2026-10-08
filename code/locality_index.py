"""Measure how much of the board each illegality class actually depends on.

The paper orders its failure classes by locality and shows their scaling
exponents follow that order. The ordering is currently an argument from the
rules: an empty source square is decidable from one square, a king left in check
is not. That is persuasive but it is not a measurement, and the exponent is.
Relating a measured exponent to an asserted ordering is weaker than relating it
to a measured quantity, and it leaves the choice of ordering as a researcher
degree of freedom. Fitting the exponents against guessed values of k gives an
R-squared anywhere between 0.67 and 0.87 depending on the guess, which is reason
enough not to publish the guess.

So k is measured here, counterfactually and without any model.

**Definition.** For a position B and an illegal move m, square s belongs to the
dependency set D(B, m) if the legality of m can be changed by altering the
contents of s alone. Formally, s is in D if there exists a piece p (or the empty
square) such that placing p on s flips `is_legal(m)`. k is |D|.

This is a causal definition, not a syntactic one: it asks what the verdict
actually depends on, rather than what the rule text mentions. It is computed by
brute force over all 64 squares and all 12 piece types, which is affordable
because the engine is cheap and the sample is thousands of moves rather than
millions.

**No model is involved.** Illegal moves are enumerated from real positions by the
rules engine, so k is a property of the error class and the position, not of
which network happened to produce the error. That matters for the claim: the law
being tested relates a quantity derived from the rules to a quantity derived from
the models, and the two must not share a source.

The squares named by the move are excluded from D. Every class depends on them by
construction, so counting them would add a constant to every k and compress the
relationship being tested.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import defaultdict

import chess
import chess.pgn

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
RES = os.path.join(HERE, "..", "results")

PIECES = [chess.Piece(pt, c) for pt in range(1, 7) for c in (True, False)]


def classify(board: chess.Board, mv: chess.Move) -> str:
    """The paper's classifier, copied so this file stands alone."""
    pc = board.piece_at(mv.from_square)
    if pc is None:
        return "from_empty"
    if pc.color != board.turn:
        return "from_opponent"
    dst = board.piece_at(mv.to_square)
    if dst is not None and dst.color == board.turn:
        # King onto one's own rook is how UCI and Chess960 encode castling, so a
        # naive destination-occupancy test captures castling attempts here. They
        # are not destination errors: whether the move is legal turns on rights,
        # the intervening squares and three check conditions, none of which live
        # on the destination square. Auditing the own-destination class is what
        # surfaced this; the paper's model-side classifier shares the defect.
        if pc.piece_type == chess.KING and dst.piece_type == chess.ROOK:
            return "castling"
        return "to_own"
    if not board.is_pseudo_legal(mv):
        return "geometry"
    if not board.is_legal(mv):
        return "leaves_check"
    return "other"


def dependency_size(board: chess.Board, mv: chess.Move) -> int:
    """|D(B, m)|: squares whose contents can flip the verdict, move squares aside.

    A square counts once however many pieces flip it. Emptying an occupied square
    is tested as well as filling an empty one, because dependence runs both ways:
    a blocking piece and a missing defender are both dependencies.
    """
    base = board.is_legal(mv)
    skip = {mv.from_square, mv.to_square}
    n = 0
    for sq in chess.SQUARES:
        if sq in skip:
            continue
        original = board.piece_at(sq)
        flipped = False
        # Emptying the square, when something is on it.
        if original is not None:
            board.remove_piece_at(sq)
            try:
                flipped = board.is_legal(mv) != base
            except Exception:                              # noqa: BLE001
                flipped = False
            board.set_piece_at(sq, original)
        if not flipped:
            for p in PIECES:
                if original is not None and p == original:
                    continue
                board.set_piece_at(sq, p)
                try:
                    if board.is_legal(mv) != base:
                        flipped = True
                except Exception:                          # noqa: BLE001
                    pass
                if original is None:
                    board.remove_piece_at(sq)
                else:
                    board.set_piece_at(sq, original)
                if flipped:
                    break
        n += int(flipped)
    return n


def illegal_moves_of_each_class(board: chess.Board, rng, per_class=2):
    """Sample illegal moves from this position, grouped by what made them illegal.

    Enumerating every illegal move is wasteful: the move space is ~4096 and most
    are from_empty. Sampling per class keeps the classes balanced so that a rare
    class is not represented by a handful of positions.
    """
    found = defaultdict(list)
    squares = list(chess.SQUARES)
    rng.shuffle(squares)
    legal = set(board.legal_moves)
    for a in squares:
        for b in squares:
            if a == b:
                continue
            mv = chess.Move(a, b)
            if mv in legal:
                continue
            # Membership is not sufficient. python-chess enumerates castling as
            # e1g1 yet is_legal also accepts the e1h1 rook-square encoding, so a
            # legal castling move passes the membership test. Ask directly.
            try:
                if board.is_legal(mv):
                    continue
            except Exception:                              # noqa: BLE001
                continue
            try:
                c = classify(board, mv)
            except Exception:                              # noqa: BLE001
                continue
            if c == "other" or len(found[c]) >= per_class:
                continue
            found[c].append(mv)
        if all(len(found[c]) >= per_class
               for c in ("from_empty", "from_opponent", "to_own",
                         "geometry", "leaves_check")):
            break
    return found


def geometry_kind(board: chess.Board, mv: chess.Move) -> str:
    """Is a geometry error a blocked path, or a move the piece cannot make at all?

    The paper describes geometry as depending on intervening squares. That is
    true of a blocked slide and false of a knight asked to move like a bishop,
    and the two are lumped under one label. Emptying the board except for the
    move's own squares separates them: if the move becomes pseudo-legal on an
    otherwise empty board it was blocked, and if it does not the geometry was
    never possible.
    """
    b = board.copy()
    for sq in chess.SQUARES:
        if sq not in (mv.from_square, mv.to_square) and b.piece_at(sq):
            b.remove_piece_at(sq)
    return "blocked" if b.is_pseudo_legal(mv) else "impossible"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=120)
    ap.add_argument("--positions-per-game", type=int, default=4)
    ap.add_argument("--per-class", type=int, default=2)
    ap.add_argument("--min-ply", type=int, default=20)
    ap.add_argument("--pgn", default=None)
    ap.add_argument("--seed", type=int, default=20260919)
    a = ap.parse_args()

    rng = random.Random(a.seed)
    pgn_path = a.pgn
    if pgn_path is None:
        import glob
        cand = sorted(glob.glob(os.path.join(HERE, "..", "data", "*.pgn")))
        if not cand:
            print("no plain .pgn found; pass --pgn. The .zst dumps need decompressing.")
            return 1
        pgn_path = cand[0]
    print(f"positions from {os.path.basename(pgn_path)}")

    sizes = defaultdict(list)
    geom = defaultdict(int)
    seen_games = 0
    with open(pgn_path, encoding="utf-8", errors="ignore") as fh:
        while seen_games < a.games:
            g = chess.pgn.read_game(fh)
            if g is None:
                break
            board = g.board()
            plies = list(g.mainline_moves())
            if len(plies) < a.min_ply + 5:
                continue
            seen_games += 1
            picks = sorted(rng.sample(range(a.min_ply, len(plies)),
                                      min(a.positions_per_game,
                                          len(plies) - a.min_ply)))
            nxt = 0
            for i, mv in enumerate(plies):
                if nxt < len(picks) and i == picks[nxt]:
                    nxt += 1
                    found = illegal_moves_of_each_class(board, rng, a.per_class)
                    for cls, moves in found.items():
                        for m in moves:
                            k = dependency_size(board.copy(), m)
                            sizes[cls].append(k)
                            if cls == "geometry":
                                kind = geometry_kind(board, m)
                                geom[kind] += 1
                                sizes["geometry_" + kind].append(k)
                board.push(mv)
            if seen_games % 20 == 0:
                print(f"  {seen_games} games, "
                      f"{sum(len(v) for v in sizes.values())} moves measured",
                      flush=True)

    import statistics as st
    out = {"games": seen_games, "seed": a.seed, "per_class": a.per_class,
           "definition": "squares whose contents can flip is_legal, move squares excluded",
           "classes": {}}
    print(f"\n{'class':16s} {'n':>6s} {'mean k':>8s} {'median':>7s} {'sd':>7s} {'min':>4s} {'max':>4s}")
    for cls in ("from_empty", "from_opponent", "to_own",
                "geometry", "geometry_impossible", "geometry_blocked",
                "castling", "leaves_check"):
        v = sizes.get(cls, [])
        if not v:
            print(f"{cls:20s} {'0':>6s}   (none sampled)")
            continue
        rec = {"n": len(v), "mean": st.mean(v), "median": st.median(v),
               "sd": st.pstdev(v) if len(v) > 1 else 0.0,
               "min": min(v), "max": max(v)}
        out["classes"][cls] = rec
        print(f"{cls:20s} {rec['n']:6d} {rec['mean']:8.2f} {rec['median']:7.1f} "
              f"{rec['sd']:7.2f} {rec['min']:4d} {rec['max']:4d}")

    tot = sum(geom.values())
    out["geometry_kinds"] = ({k: {"n": v, "share": v / tot}
                              for k, v in geom.items()} if tot else {})
    if tot:
        print(f"\ngeometry sub-kinds over {tot} moves:")
        for k, v in sorted(geom.items(), key=lambda kv: -kv[1]):
            print(f"  {k:11s} n={v:5d}  share={v / tot:.3f}")
    json.dump(out, open(os.path.join(RES, "locality_index.json"), "w"), indent=1)
    print("\nwrote results/locality_index.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
