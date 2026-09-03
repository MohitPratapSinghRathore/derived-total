"""Is an illegal move a random error, or a correct move in a misremembered board?

The locality result says that when the model plays an illegal move, the probe is
wrong on precisely the squares that move touches. That is suggestive but stops
short of the claim worth making. This tests the claim directly.

We decode the model's believed board from the probe, reconstruct it as an actual
position, and ask whether the illegal move is LEGAL IN THAT BELIEVED POSITION.
If it is, the model is not hallucinating moves. It is playing coherently in a
board it has misremembered, which is a statement about memory rather than about
policy, and it is what the whole decomposition is trying to establish.

Controls, because a permissive decoded board would make anything look legal:
  * the same test on a RANDOM illegal move (how often does the believed board
    happen to license an arbitrary illegal move?),
  * the true-board rate, which is zero by construction,
  * the believed-board legality rate for moves the model played that were
    actually legal, which should be high if the decoding is faithful.
"""
import os, json, argparse
import numpy as np
import torch
import chess
from probes import load_split, build

DATA = os.path.join(os.path.dirname(__file__), "..", "data", "proc")
RUNS = os.path.join(os.path.dirname(__file__), "..", "runs")
RES = os.path.join(os.path.dirname(__file__), "..", "results")
BUCKETS = [(0, 10), (10, 20), (20, 30), (30, 40), (40, 50),
           (50, 60), (60, 80), (80, 120), (120, 161)]

CLS2PIECE = {}
for ci, col in enumerate([chess.WHITE, chess.BLACK]):
    for pi, pt in enumerate([chess.PAWN, chess.KNIGHT, chess.BISHOP,
                             chess.ROOK, chess.QUEEN, chess.KING]):
        CLS2PIECE[1 + ci * 6 + pi] = (pt, col)


def bucket_of(t):
    for i, (lo, hi) in enumerate(BUCKETS):
        if lo <= t < hi:
            return i
    return None


def believed_board(occ_row, turn_white):
    """Rebuild a position from decoded occupancy. Castling rights are cleared so
    that castling cannot be licensed by an assumption we did not decode."""
    b = chess.Board(None)
    for sq in range(64):
        c = int(occ_row[sq])
        if c:
            pt, col = CLS2PIECE[c]
            b.set_piece_at(sq, chess.Piece(pt, col))
    b.turn = chess.WHITE if turn_white else chess.BLACK
    b.castling_rights = 0
    return b


def pseudo_ok(board, mv):
    """Pseudo-legality: the piece exists, belongs to the mover, and the geometry
    is legal. We avoid full legality because a decoded board may lack a king,
    which makes check tests ill-defined."""
    try:
        return mv in board.pseudo_legal_moves
    except Exception:
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--games", type=int, default=1500)
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
    all_moves = [m for m in stoi if m not in ("<pad>", "<bos>")]
    rng = np.random.default_rng(0)

    nb = len(BUCKETS)
    ill_n = np.zeros(nb); ill_believed_ok = np.zeros(nb)
    leg_n = np.zeros(nb); leg_believed_ok = np.zeros(nb)
    rand_n = np.zeros(nb); rand_believed_ok = np.zeros(nb)
    # mismatched-belief control: score the played illegal move against a belief
    # board decoded at a DIFFERENT position of similar depth. This separates
    # "the model's own belief licenses it" from "any plausible board licenses it".
    mis_n = np.zeros(nb); mis_believed_ok = np.zeros(nb)
    belief_bank = {b: [] for b in range(nb)}
    per_case = []          # (bucket, is_illegal, believed_ok) for bootstrap CIs
    clus = {"game": [], "own": [], "mis": [], "rnd": []}   # game-clustered records

    bs = 32
    with torch.no_grad():
        for i in range(0, len(toks), bs):
            idx = np.arange(i, min(i + bs, len(toks)))
            x = torch.from_numpy(toks[idx]).cuda()
            with torch.amp.autocast("cuda", dtype=torch.float16):
                logits, hs, _ = model(x[:, :-1], return_hidden=True)
            pred_tok = logits.float().argmax(-1).cpu().numpy()
            dec = probe(hs[best].float()).reshape(len(idx), -1, 64, 13).argmax(-1).cpu().numpy()

            for r, gi in enumerate(idx):
                T = int(lens[gi])
                board = chess.Board()
                for t in range(min(T - 1, dec.shape[1])):
                    b = bucket_of(t)
                    if b is not None:
                        mv_s = itos.get(int(pred_tok[r, t]), "")
                        try:
                            mv = chess.Move.from_uci(mv_s)
                        except Exception:
                            mv = None
                        if mv is not None:
                            legal = mv in board.legal_moves
                            bb = believed_board(dec[r, t], board.turn == chess.WHITE)
                            ok = pseudo_ok(bb, mv)
                            per_case.append((b, 0 if legal else 1, int(ok)))
                            if len(belief_bank[b]) < 400:
                                belief_bank[b].append(dec[r, t].copy())
                            elif rng.random() < 0.02:
                                belief_bank[b][int(rng.integers(400))] = dec[r, t].copy()
                            if legal:
                                leg_n[b] += 1; leg_believed_ok[b] += ok
                            else:
                                ill_n[b] += 1; ill_believed_ok[b] += ok
                                # control: an arbitrary illegal move at this position
                                cand = None
                                for _ in range(6):
                                    _c = chess.Move.from_uci(
                                        all_moves[int(rng.integers(len(all_moves)))])
                                    if _c not in board.legal_moves:
                                        cand = _c
                                        rand_n[b] += 1
                                        rand_believed_ok[b] += pseudo_ok(bb, cand)
                                        break
                                if cand is None:
                                    cand = chess.Move.null()
                                # same move, someone else's belief at similar depth
                                if belief_bank[b]:
                                    other = belief_bank[b][int(rng.integers(len(belief_bank[b])))]
                                    ob = believed_board(other, board.turn == chess.WHITE)
                                    mis_n[b] += 1
                                    _mo = pseudo_ok(ob, mv)
                                    mis_believed_ok[b] += _mo
                                    clus["game"].append(int(gi))
                                    clus["own"].append(int(ok))
                                    clus["mis"].append(int(_mo))
                                    clus["rnd"].append(int(pseudo_ok(bb, cand)))
                    board.push_uci(itos[int(toks[gi, t + 1])])

    out = {
        "name": args.name, "buckets": BUCKETS,
        "illegal_legal_in_belief": (ill_believed_ok / np.maximum(ill_n, 1)).tolist(),
        "legal_legal_in_belief": (leg_believed_ok / np.maximum(leg_n, 1)).tolist(),
        "randomillegal_legal_in_belief": (rand_believed_ok / np.maximum(rand_n, 1)).tolist(),
        "mismatched_belief_legal": (mis_believed_ok / np.maximum(mis_n, 1)).tolist(),
        "n_illegal": ill_n.tolist(), "n_legal": leg_n.tolist(),
        "n_random": rand_n.tolist(), "n_mismatched": mis_n.tolist(),
        "overall": {
            "illegal_legal_in_belief": float(ill_believed_ok.sum() / max(ill_n.sum(), 1)),
            "legal_legal_in_belief": float(leg_believed_ok.sum() / max(leg_n.sum(), 1)),
            "randomillegal_legal_in_belief": float(rand_believed_ok.sum() / max(rand_n.sum(), 1)),
            "mismatched_belief_legal": float(mis_believed_ok.sum() / max(mis_n.sum(), 1)),
        },
    }
    # Game-clustered intervals on exactly the population the headline numbers
    # use, so the point estimates are unchanged and only the interval is added.
    if clus["game"]:
        _g = np.array(clus["game"]); _uniq = np.unique(_g)
        _by = {u: np.where(_g == u)[0] for u in _uniq}
        _rng = np.random.default_rng(0)
        _draws = {k: [] for k in ("own", "mis", "rnd")}
        for _ in range(400):
            _pick = _rng.choice(_uniq, len(_uniq), replace=True)
            _sel = np.concatenate([_by[u] for u in _pick])
            for k in _draws:
                _draws[k].append(float(np.array(clus[k])[_sel].mean()))
        out["clustered"] = {
            k: {"mean": float(np.mean(clus[k])),
                "ci_lo": float(np.percentile(_draws[k], 2.5)),
                "ci_hi": float(np.percentile(_draws[k], 97.5))}
            for k in _draws}
        out["clustered"]["n_games"] = int(len(_uniq))
        out["clustered"]["n"] = int(len(_g))

    # bootstrap CI over cases for the headline contrast
    pc = np.array(per_case)
    if len(pc):
        ill = pc[pc[:, 1] == 1][:, 2]
        boots = [ill[rng.integers(0, len(ill), len(ill))].mean() for _ in range(400)]
        out["overall"]["illegal_ci95"] = [float(np.percentile(boots, 2.5)),
                                          float(np.percentile(boots, 97.5))]
    json.dump(out, open(os.path.join(RES, f"{args.name}_belief.json"), "w"), indent=1)

    print(f"[{args.name}] belief-consistency of illegal moves")
    print(f"{'ply':>9} {'ill->believedOK':>16} {'leg->believedOK':>16} "
          f"{'rand->believedOK':>17} {'n_ill':>7}")
    for i, (lo, hi) in enumerate(BUCKETS):
        print(f"{str(lo)+'-'+str(hi):>9} {out['illegal_legal_in_belief'][i]:16.4f} "
              f"{out['legal_legal_in_belief'][i]:16.4f} "
              f"{out['randomillegal_legal_in_belief'][i]:17.4f} {ill_n[i]:7.0f}")
    o = out["overall"]
    print(f"\nOVERALL  illegal moves legal in believed board: {o['illegal_legal_in_belief']:.4f}")
    print(f"         legal moves   legal in believed board: {o['legal_legal_in_belief']:.4f}")
    print(f"         random illegal legal in believed board: {o['randomillegal_legal_in_belief']:.4f}")
    print(f"         SAME move, mismatched belief board:      {o['mismatched_belief_legal']:.4f}")
    if "illegal_ci95" in o:
        print(f"         illegal rate 95% CI: [{o['illegal_ci95'][0]:.4f}, {o['illegal_ci95'][1]:.4f}]")


if __name__ == "__main__":
    main()
