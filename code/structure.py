"""What kind of wrong are the failures repair cannot reach?

The whole-board repair arm was withdrawn (amendment A3), so the residual cannot
be probed by editing: natural failures carry a median of 14 divergent squares and
the channel fails above about two. Description is the only route left, and it is
worth taking, because a residual with structure is a named mechanism while a
residual without structure is a shrug.

Each illegal top-1 move is assigned to exactly one class, in priority order, by
asking what actually made it illegal on the true board:

  from_empty        no piece on the source square
  from_opponent     the source holds the other side's piece
  to_own            the destination holds a piece of the mover's own colour
  geometry          a piece of the right colour is there, but it cannot reach the
                    destination that way, or the path is blocked
  leaves_check      pseudo-legal and rejected only because it leaves the king in
                    check
  other             none of the above

Two orthogonal flags, since these cut across the classes and are the ones that
would name a mechanism:

  legal_recently    the move was legal at some ply within the last ten, so the
                    model is playing a position it has not updated
  legal_other_side  the move would be legal if it were the opponent's turn

Reported separately for failures where belief was already correct at both action
squares (the policy bucket) and where it was not, because those are different
populations and averaging them hides whichever is structured.

This is descriptive. It does not establish that the residual is genuine failure
rather than beyond the reach of the edit, and must not be written as if it does.
"""
import os, json, argparse
import numpy as np
import torch
import chess
from probes import load_split, build

DATA = os.path.join(os.path.dirname(__file__), "..", "data", "proc")
RUNS = os.path.join(os.path.dirname(__file__), "..", "runs")
RES = os.path.join(os.path.dirname(__file__), "..", "results")
CLASSES = ["from_empty", "from_opponent", "to_own", "geometry",
           "leaves_check", "other"]
LOOKBACK = 10


def boot_shares(labels, games, keys, n_boot=500, seed=0):
    rng = np.random.default_rng(seed)
    labels = np.asarray(labels)
    games = np.asarray(games)
    u = np.unique(games)
    by = {g: np.where(games == g)[0] for g in u}
    acc = {k: [] for k in keys}
    for _ in range(n_boot):
        sel = np.concatenate([by[g] for g in rng.choice(u, len(u), replace=True)])
        l = labels[sel]
        for k in keys:
            acc[k].append(float((l == k).mean()))
    return {k: (float(np.mean(acc[k])), float(np.percentile(acc[k], 2.5)),
                float(np.percentile(acc[k], 97.5))) for k in keys}


def classify(board, mv):
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="attn_8L256_s0")
    ap.add_argument("--n", type=int, default=1500)
    ap.add_argument("--minply", type=int, default=20)
    ap.add_argument("--games", type=int, default=3000)
    args = ap.parse_args()

    model, a, ck = build(os.path.join(RUNS, f"{args.name}.pt"))
    bl = json.load(open(os.path.join(RES, f"{args.name}.json")))["best_layer"]
    pls = torch.load(os.path.join(RES, f"{args.name}_probes.pt"),
                     weights_only=False)
    probe = torch.nn.Linear(a["width"], 64 * 13).cuda()
    probe.load_state_dict(pls[bl])
    probe.eval()
    stoi = json.load(open(os.path.join(DATA, "vocab.json")))
    itos = {v: k for k, v in stoi.items()}
    toks, lens, occ = load_split("eval", args.games)

    labels, games, is_policy, recent, other_side, plies = [], [], [], [], [], []
    n = 0
    with torch.no_grad():
        for gi in range(len(toks)):
            if n >= args.n:
                break
            T = int(lens[gi])
            if T < args.minply + 4:
                continue
            board = chess.Board()
            hist = []
            for t in range(T - 1):
                if n >= args.n:
                    break
                if t >= args.minply:
                    x = torch.from_numpy(
                        toks[gi:gi + 1, :t + 1].astype(np.int64)).cuda()
                    with torch.amp.autocast("cuda", dtype=torch.float16):
                        lg, hs, _ = model(x, return_hidden=True)
                    p0 = torch.softmax(lg.float()[0, -1], -1)
                    bel = probe(hs[bl][0, -1].float()).reshape(64, 13) \
                        .argmax(-1).cpu().numpy()
                    top1 = int(p0.argmax())
                    try:
                        mv = chess.Move.from_uci(itos.get(top1, ""))
                        ill = mv not in board.legal_moves
                    except Exception:
                        mv, ill = None, False
                    if mv is not None and ill:
                        truth = np.asarray(occ[gi, t]).astype(int)
                        act = [mv.from_square, mv.to_square]
                        labels.append(classify(board, mv))
                        is_policy.append(
                            int(all(bel[s] == truth[s] for s in act)))
                        rc = 0
                        for b_old in hist[-LOOKBACK:]:
                            try:
                                if mv in b_old.legal_moves:
                                    rc = 1
                                    break
                            except Exception:
                                pass
                        recent.append(rc)
                        bo = board.copy()
                        bo.turn = not bo.turn
                        try:
                            other_side.append(int(mv in bo.legal_moves))
                        except Exception:
                            other_side.append(0)
                        games.append(gi)
                        plies.append(t)
                        n += 1
                hist.append(board.copy())
                try:
                    board.push_uci(itos[int(toks[gi, t + 1])])
                except Exception:
                    break

    labels = np.array(labels)
    games = np.array(games)
    is_policy = np.array(is_policy)
    recent = np.array(recent)
    other_side = np.array(other_side)
    sh = boot_shares(labels, games, CLASSES)

    out = {"name": args.name, "n": int(n),
           "n_games": int(len(np.unique(games))),
           "overall": {k: {"mean": sh[k][0], "ci_lo": sh[k][1],
                           "ci_hi": sh[k][2]} for k in CLASSES},
           "legal_recently": float(recent.mean()),
           "legal_other_side": float(other_side.mean()),
           "policy_share": float(is_policy.mean())}

    print(f"\n[{args.name}] structure of {n} illegal top-1 moves from "
          f"{len(np.unique(games))} games\n")
    print(f"{'class':>16} {'share':>8}  {'95% CI':>20}  "
          f"{'belief OK at act':>17} {'belief wrong':>13}")
    for k in CLASSES:
        m, lo, hi = sh[k]
        mk = labels == k
        pa = float(mk[is_policy == 1].mean()) if (is_policy == 1).sum() else 0.0
        pb = float(mk[is_policy == 0].mean()) if (is_policy == 0).sum() else 0.0
        out["overall"][k]["share_when_belief_ok"] = pa
        out["overall"][k]["share_when_belief_wrong"] = pb
        print(f"{k:>16} {m:8.4f}  [{lo:.4f}, {hi:.4f}]  {pa:17.4f} {pb:13.4f}")

    print(f"\n  move was legal within the last {LOOKBACK} plies   "
          f"{recent.mean():.4f}")
    print(f"  move would be legal for the other side     "
          f"{other_side.mean():.4f}")
    print(f"  belief already correct at both act squares {is_policy.mean():.4f}")

    for nm, m in (("belief correct at action squares", is_policy == 1),
                  ("belief wrong at action squares", is_policy == 0)):
        if m.sum() < 30:
            continue
        out.setdefault("flags_by_bucket", {})[nm] = {
            "n": int(m.sum()),
            "legal_recently": float(recent[m].mean()),
            "legal_other_side": float(other_side[m].mean())}
        print(f"\n  [{nm}] n={m.sum()}")
        print(f"    legal within last {LOOKBACK} plies  "
              f"{recent[m].mean():.4f}")
        print(f"    legal for the other side      {other_side[m].mean():.4f}")

    print(f"\nby depth:")
    plies = np.array(plies)
    print(f"{'ply':>9} {'n':>6} " + " ".join(f"{k[:11]:>12}" for k in CLASSES)
          + f" {'recent':>8}")
    for lo_, hi_ in [(20, 40), (40, 60), (60, 120)]:
        m = (plies >= lo_) & (plies < hi_)
        if m.sum() < 25:
            continue
        out.setdefault("strata", {})[f"{lo_}-{hi_}"] = {
            "n": int(m.sum()),
            **{k: float((labels[m] == k).mean()) for k in CLASSES},
            "legal_recently": float(recent[m].mean())}
        print(f"{str(lo_)+'-'+str(hi_):>9} {m.sum():6,} "
              + " ".join(f"{float((labels[m]==k).mean()):12.4f}"
                         for k in CLASSES)
              + f" {float(recent[m].mean()):8.4f}")

    json.dump(out, open(os.path.join(RES, f"{args.name}_structure.json"), "w"),
              indent=1)


if __name__ == "__main__":
    main()
