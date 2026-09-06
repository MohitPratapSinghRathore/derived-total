"""Classify illegal SAN moves, preserving the local/global distinction.

board.parse_san raises for any illegal move, well-formed or not, so it cannot
separate "not a move string" from "a move string that is illegal here". An
earlier version of the Chess-GPT analysis used it directly and put every failure
in "unparseable", which was wrong: none of the observed failures were malformed.

The deeper issue is that SAN does not name a source square. UCI e2e4 always
parses to a Move whatever the position, so from_empty is directly checkable. SAN
names a piece type and a destination and leaves the source implicit, so the most
local failure class in our taxonomy has no SAN analogue at all. The model cannot
claim a piece stands on an empty square, because it never names the square.

That is worth stating rather than working around: the notation structurally
prevents the failure mode that scales away fastest in our ladder, which is one
reason a SAN model is not directly comparable to a UCI one.

The classes below keep the distinction the argument rests on:

  malformed      not SAN at all
  to_own         destination holds one of the mover's own pieces
  unreachable    no piece of the named type can pseudo-legally reach the
                 destination. Local: determined by the piece and the path.
  leaves_check   some piece of that type can reach it, but every such move
                 leaves the king in check. Global: determined by the whole board.
  ambiguous      several pieces could make the move and the SAN does not
                 disambiguate, so the string is under-specified rather than wrong
"""
import re
import chess

SAN_RE = re.compile(
    r"^(?P<piece>[KQRBN])?"
    r"(?P<ff>[a-h])?(?P<fr>[1-8])?"
    r"(?P<cap>x)?"
    r"(?P<dest>[a-h][1-8])"
    r"(?:=(?P<promo>[QRBN]))?"
    r"(?P<chk>[+#])?$")

CLASSES = ["malformed", "to_own", "unreachable", "leaves_check", "ambiguous",
           "other"]


def parse_loose(san):
    """Structure of a SAN string without asking whether it is legal."""
    s = san.strip()
    if s in ("O-O", "O-O-O", "0-0", "0-0-0"):
        return {"castle": s.replace("0", "O")}
    m = SAN_RE.match(s)
    if not m:
        return None
    d = m.groupdict()
    return {"castle": None,
            "piece": chess.PIECE_SYMBOLS.index((d["piece"] or "P").lower()),
            "dest": chess.parse_square(d["dest"]),
            "from_file": chess.FILE_NAMES.index(d["ff"]) if d["ff"] else None,
            "from_rank": int(d["fr"]) - 1 if d["fr"] else None,
            "promo": (chess.PIECE_SYMBOLS.index(d["promo"].lower())
                      if d["promo"] else None)}


def classify_san(board, san):
    p = parse_loose(san)
    if p is None:
        return "malformed"

    if p["castle"]:
        want = chess.Move.from_uci(
            ("e1g1" if board.turn else "e8g8") if p["castle"] == "O-O"
            else ("e1c1" if board.turn else "e8c8"))
        if board.is_pseudo_legal(want):
            return "leaves_check" if not board.is_legal(want) else "other"
        return "unreachable"

    dest = p["dest"]
    occ = board.piece_at(dest)
    if occ is not None and occ.color == board.turn:
        return "to_own"

    # candidate sources: pieces of the named type, filtered by any disambiguator
    cands = []
    for sq in board.pieces(p["piece"], board.turn):
        if p["from_file"] is not None and chess.square_file(sq) != p["from_file"]:
            continue
        if p["from_rank"] is not None and chess.square_rank(sq) != p["from_rank"]:
            continue
        cands.append(sq)
    if not cands:
        return "unreachable"

    pseudo = []
    for sq in cands:
        mv = chess.Move(sq, dest, promotion=p["promo"])
        if board.is_pseudo_legal(mv):
            pseudo.append(mv)
    if not pseudo:
        return "unreachable"

    legal = [m for m in pseudo if board.is_legal(m)]
    if not legal:
        return "leaves_check"
    if len(legal) > 1:
        return "ambiguous"
    return "other"


def selftest():
    """Each case names the class it must produce, so a regression is visible."""
    cases = [
        # position, san, expected
        (chess.Board(), "e5", "unreachable"),      # no pawn can reach e5
        # e2 holds White's own pawn, so this is to_own, not unreachable. The
        # first version of this case expected unreachable and the classifier was
        # right; the test was wrong.
        (chess.Board(), "Ke2", "to_own"),
        (chess.Board(), "Nf3", "other"),           # legal
        (chess.Board(), "Qd1", "to_own"),          # own queen already there
        (chess.Board(), "zz9", "malformed"),
        (chess.Board("4k3/8/8/8/8/8/4r3/4K3 w - - 0 1"), "Kd1", "other"),
        (chess.Board("4k3/8/8/8/8/8/4r3/4K3 w - - 0 1"), "Ke2", "other"),
        (chess.Board("rnbqkbnr/pppp1ppp/8/4p3/6P1/5P2/PPPPP2P/RNBQKBNR b KQkq - 0 2"),
         "Qh4", "other"),
    ]
    bad = 0
    for b, san, want in cases:
        got = classify_san(b, san)
        ok = got == want
        if not ok:
            bad += 1
        print(f"  {'ok ' if ok else 'BAD'} {san:<6} -> {got:<13} "
              f"(expected {want})")
    print(f"  {len(cases)-bad}/{len(cases)} correct")
    return bad == 0


if __name__ == "__main__":
    print("SAN classifier self-test")
    selftest()
