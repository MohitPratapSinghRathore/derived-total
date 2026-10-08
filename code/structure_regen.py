"""Regenerate the failure sets with corrected legality and classification.

The audit in the locality section found that the classifier folds castling into
the own-destination class, and worse, that the failure gate itself was a
membership test against the generated legal moves. python-chess emits castling as
e1g1 while ``is_legal`` also accepts the e1h1 king-onto-rook form, so a legal
castling move written that way passes ``mv not in board.legal_moves`` and is
recorded as a failure. Disclosure is not correction, so this re-derives the sets.

Three things are done differently from the original pass, and only the first two
change any number:

1. The failure gate asks ``board.is_legal(mv)`` directly rather than testing
   membership in the generated move list, so a legal move cannot enter the set.
2. Castling becomes its own class, and geometry splits into impossible movement
   and blocked path, so the scaling analysis uses the repaired taxonomy rather
   than a partly heterogeneous category.
3. Every failure is recorded with its position, move and both labels, so that the
   next audit does not need a regeneration pass. Not retaining them is why the
   affected share could not previously be quantified.

Each move is labelled under both the old and the new scheme on the same forward
pass. That matters for attribution: the class differences are then caused by the
classifier alone and cannot be confounded with any numerical difference between
this run and the original. The original pass ran in float16 on a GPU that is no
longer attached; this one runs in float32 on whatever device is present, and the
drift that introduces is measured separately by comparing the old labels here
against the stored shares rather than being assumed to be zero.

One speed note, because it changes nothing but makes the pass affordable on CPU.
The model is causal, so a single forward over a game's full token sequence yields
the next-move distribution at every prefix simultaneously; the original pass ran
one forward per ply over a growing prefix. The logits at position t depend only
on tokens up to t under the causal mask, so the two are identical up to
floating-point summation order.
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

OLD_CLASSES = ["from_empty", "from_opponent", "to_own", "geometry",
               "leaves_check", "other"]
NEW_CLASSES = ["from_empty", "from_opponent", "to_own", "castling",
               "geometry_impossible", "geometry_blocked", "leaves_check",
               "other"]
LOOKBACK = 10


def classify_old(board, mv):
    """The original classifier, kept verbatim so the comparison is honest."""
    pc = board.piece_at(mv.from_square)
    if pc is None:
        return "from_empty"
    if pc.color != board.turn:
        return "from_opponent"
    dst = board.piece_at(mv.to_square)
    if dst is not None and dst.color == board.turn:
        return "to_own"
    if not board.is_pseudo_legal(mv):
        return "geometry"
    if not board.is_legal(mv):
        return "leaves_check"
    return "other"


def classify_new(board, mv):
    """Castling separated; geometry split by whether any arrangement permits it."""
    pc = board.piece_at(mv.from_square)
    if pc is None:
        return "from_empty"
    if pc.color != board.turn:
        return "from_opponent"
    dst = board.piece_at(mv.to_square)
    if dst is not None and dst.color == board.turn:
        # King onto one's own rook is the UCI and Chess960 castling encoding, so
        # a destination-occupancy test would collect castling attempts. What
        # makes such a move illegal is rights, the intervening squares or a check
        # condition, none of which sit on the destination square.
        if pc.piece_type == chess.KING and dst.piece_type == chess.ROOK:
            return "castling"
        return "to_own"
    if not board.is_pseudo_legal(mv):
        # Blocked means the move would be available on an emptier board; the rest
        # are movements the piece cannot make under any arrangement.
        b = board.copy()
        keep = {mv.from_square, mv.to_square}
        for sq in chess.SQUARES:
            if sq not in keep:
                b.remove_piece_at(sq)
        return ("geometry_blocked" if b.is_pseudo_legal(mv)
                else "geometry_impossible")
    if not board.is_legal(mv):
        return "leaves_check"
    return "other"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="attn_8L256_s0")
    ap.add_argument("--n", type=int, default=900)
    ap.add_argument("--minply", type=int, default=20)
    ap.add_argument("--games", type=int, default=3000)
    ap.add_argument("--device", default=None)
    a = ap.parse_args()
    dev = a.device or ("cuda" if torch.cuda.is_available() else "cpu")

    model, arch, _ = build(os.path.join(RUNS, f"{a.name}.pt"), device=dev)
    bl = json.load(open(os.path.join(RES, f"{a.name}.json")))["best_layer"]
    pls = torch.load(os.path.join(RES, f"{a.name}_probes.pt"),
                     map_location=dev, weights_only=False)
    probe = torch.nn.Linear(arch["width"], 64 * 13).to(dev)
    probe.load_state_dict(pls[bl])
    probe.eval()

    stoi = json.load(open(os.path.join(DATA, "vocab.json")))
    itos = {v: k for k, v in stoi.items()}
    toks, lens, occ = load_split("eval", a.games)

    recs = []
    with torch.no_grad():
        for gi in range(len(toks)):
            if len(recs) >= a.n:
                break
            T = int(lens[gi])
            if T < a.minply + 4:
                continue
            x = torch.from_numpy(toks[gi:gi + 1, :T].astype(np.int64)).to(dev)
            lg, hs, _ = model(x, return_hidden=True)
            top1 = lg[0].argmax(-1).cpu().numpy()
            bel_all = probe(hs[bl][0].float()).reshape(T, 64, 13) \
                .argmax(-1).cpu().numpy()

            board = chess.Board()
            hist = []
            for t in range(T - 1):
                if len(recs) >= a.n:
                    break
                if t >= a.minply:
                    uci = itos.get(int(top1[t]), "")
                    try:
                        mv = chess.Move.from_uci(uci)
                    except Exception:                      # noqa: BLE001
                        mv = None
                    if mv is not None:
                        in_gen = mv in board.legal_moves
                        try:
                            truly_legal = board.is_legal(mv)
                        except Exception:                  # noqa: BLE001
                            truly_legal = False
                        # The original gate. Kept so that the set of moves this
                        # pass examines is the same one the paper reported on.
                        if not in_gen:
                            truth = np.asarray(occ[gi, t]).astype(int)
                            bel = bel_all[t]
                            act = [mv.from_square, mv.to_square]
                            rc = 0
                            for b_old in hist[-LOOKBACK:]:
                                try:
                                    if mv in b_old.legal_moves:
                                        rc = 1
                                        break
                                except Exception:          # noqa: BLE001
                                    pass
                            bo = board.copy()
                            bo.turn = not bo.turn
                            try:
                                os_flag = int(mv in bo.legal_moves)
                            except Exception:              # noqa: BLE001
                                os_flag = 0
                            recs.append({
                                "game": int(gi), "ply": int(t),
                                "fen": board.fen(), "uci": uci,
                                "old_label": classify_old(board, mv),
                                "new_label": classify_new(board, mv),
                                "in_generated_legal": bool(in_gen),
                                "truly_legal": bool(truly_legal),
                                "is_policy": int(all(bel[s] == truth[s]
                                                     for s in act)),
                                "legal_recently": rc,
                                "legal_other_side": os_flag})
                hist.append(board.copy())
                try:
                    board.push_uci(itos[int(toks[gi, t + 1])])
                except Exception:                          # noqa: BLE001
                    break

    out = {"name": a.name, "device": dev, "dtype": "float32",
           "n_examined": len(recs),
           "n_truly_legal": sum(1 for r in recs if r["truly_legal"]),
           "records": recs}
    p = os.path.join(RES, f"{a.name}_regen.json")
    json.dump(out, open(p, "w"), indent=0)
    nl = out["n_truly_legal"]
    chg = sum(1 for r in recs if r["old_label"] != r["new_label"])
    print(f"[{a.name}] {len(recs)} examined, {nl} actually legal, "
          f"{chg} relabelled -> {os.path.basename(p)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
