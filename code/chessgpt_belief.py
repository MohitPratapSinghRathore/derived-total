"""Belief consistency and the move-touched AUC, on Chess-GPT.

Two of the three load-bearing measurements, ported to a model that saw roughly
16M games rather than our 96k. If they hold here, the objection that the
mechanism is an artefact of undertrained 40M-parameter models has a direct
empirical answer.

Belief consistency asks, among positions where the model's move is illegal on the
true board, how often that move is legal on the board its own probe decodes.
Controls follow the main paper:

  own          the move against this position's decoded board
  mismatched   the same move against a decoded board from a DIFFERENT position at
               similar depth, which keeps the statistics of a decoded board while
               destroying the correspondence
  random       a uniformly random move against this position's decoded board,
               which is the floor
  reference    the fraction of moves legal in the true position, so coherence is
               readable against how easy being legal is here

AUC asks whether per-square probe fidelity predicts an illegal next move, and
whether fidelity at the two squares the move touches predicts it better than
whole-board fidelity. Move-touched beating aggregate is the local-coupling claim.

Both distributions are reported. Chess-GPT plays human games almost perfectly, so
the illegal population there is thin; uniformly-random legal play walks it into
unfamiliar positions and lifts the rate by orders of magnitude. The claim is that
the conditional structure given a failure is stable across the two, not the rate.
"""
import os, json, argparse
import numpy as np
import torch
import torch.nn as nn
import chess
from chessgpt import load, encode
from chessgpt_nano import convert as nano_convert
from chessgpt_data import build, load_games, random_legal_games
from chessgpt_structure import decode_game
from san_classify import parse_loose

ROOT = os.path.join(os.path.dirname(__file__), "..")
RES = os.path.join(ROOT, "results")
EXT = os.path.join(ROOT, "data", "ext")
NCLS = 13


def boot_ci(v, g, n_boot=500, seed=0):
    rng = np.random.default_rng(seed)
    v = np.asarray(v, float)
    g = np.asarray(g)
    u = np.unique(g)
    by = {k: np.where(g == k)[0] for k in u}
    o = [v[np.concatenate([by[k] for k in
         rng.choice(u, len(u), replace=True)])].mean() for _ in range(n_boot)]
    return float(np.percentile(o, 2.5)), float(np.percentile(o, 97.5))


def auc(s, y):
    s, y = np.asarray(s, float), np.asarray(y, int)
    n1, n0 = int(y.sum()), int((1 - y).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    o = np.argsort(s, kind="mergesort")
    r = np.empty(len(s), float)
    r[o] = np.arange(1, len(s) + 1)
    u, inv, cnt = np.unique(s, return_inverse=True, return_counts=True)
    sm = np.zeros(len(u))
    np.add.at(sm, inv, r)
    r = (sm / cnt)[inv]
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def auc_ci(s, y, g, n_boot=400, seed=0):
    rng = np.random.default_rng(seed)
    s, y, g = np.asarray(s), np.asarray(y), np.asarray(g)
    u = np.unique(g)
    by = {k: np.where(g == k)[0] for k in u}
    out = []
    for _ in range(n_boot):
        sel = np.concatenate([by[k] for k in rng.choice(u, len(u), replace=True)])
        v = auc(s[sel], y[sel])
        if np.isfinite(v):
            out.append(v)
    return (float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))) \
        if out else (float("nan"), float("nan"))


def board_from_occ(occ, turn):
    b = chess.Board()
    b.clear()
    b.turn = turn
    for sq in range(64):
        c = int(occ[sq])
        if c > 0:
            b.set_piece_at(sq, chess.Piece((c - 1) % 6 + 1,
                                           chess.WHITE if c <= 6 else chess.BLACK))
    return b


def legal_on(board, san):
    try:
        return board.parse_san(san) in board.legal_moves
    except Exception:
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=800)
    ap.add_argument("--minply", type=int, default=20)
    ap.add_argument("--maxply", type=int, default=120)
    ap.add_argument("--dist", choices=["human", "random"], default="human")
    ap.add_argument("--nano", default=None)
    ap.add_argument("--tag", default="chessgpt2")
    ap.add_argument("--label", default=None)
    args = ap.parse_args()

    if args.nano:
        model, cfg, _ = nano_convert(os.path.join(EXT, args.nano))
    else:
        model, cfg, _ = load(args.tag)
    label = args.label or (args.nano or args.tag)
    safe = label.replace(".pt", "")
    pk = torch.load(os.path.join(RES, f"{safe}_probes.pt"), weights_only=False)
    bl = pk["best_layer"]
    probe = nn.Linear(pk["d"], 64 * NCLS).cuda()
    probe.load_state_dict(pk["probes"][bl])
    probe.eval()

    games = (random_legal_games(args.games, np.random.default_rng(0),
                                max_plies=args.maxply)
             if args.dist == "random" else load_games("eval", args.games))

    # one pass: record everything, compute controls afterwards
    rec = []
    with torch.no_grad():
        for gi, g in enumerate(games):
            text, sites = build(g, max_plies=args.maxply)
            ids = encode(text)[:1023]
            want = [s["char_index"] for s in sites
                    if s["ply"] >= args.minply and s["char_index"] < len(ids)]
            if not want:
                continue
            sans = decode_game(model, ids, want)
            x = torch.tensor([ids], dtype=torch.long, device="cuda")
            with torch.amp.autocast("cuda", dtype=torch.float16):
                hs = model(x, output_hidden_states=True).hidden_states
            h = hs[bl][0].float()
            for s in sites:
                ci = s["char_index"]
                if ci not in sans:
                    continue
                dec = probe(h[ci]).reshape(64, NCLS).argmax(-1).cpu().numpy()
                b = chess.Board(s["fen"])
                san = sans[ci]
                rec.append({"g": gi, "ply": s["ply"], "fen": s["fen"],
                            "turn": int(s["turn"]), "san": san,
                            "dec": dec, "truth": s["occ"],
                            "legal": legal_on(b, san)})
            if len(rec) > 60000:
                break

    rng = np.random.default_rng(0)
    by_bucket = {}
    for i, r in enumerate(rec):
        by_bucket.setdefault(r["ply"] // 10, []).append(i)

    own, mis, rnd, ref, gid = [], [], [], [], []
    fid_all, fid_touch, y, gy, touch_ok = [], [], [], [], []
    for i, r in enumerate(rec):
        truth, dec = r["truth"], r["dec"]
        b = chess.Board(r["fen"])
        fid_all.append(float((dec == truth).mean()))
        # Move-touched fidelity must be defined identically for legal and
        # illegal moves, or the measure is comparing different square sets
        # between the two classes. board.parse_san raises on every illegal move,
        # so an earlier version fell back to an arbitrary legal move and scored
        # the failure cases at unrelated squares, which reversed the result.
        #
        # SAN never names a source square, so the destination is the only
        # move-touched square the notation provides. It is extracted by the
        # loose parser, which works whether or not the move is legal, plus any
        # explicit disambiguation file/rank the string carries.
        pl = parse_loose(r["san"])
        tsq = None
        if pl and not pl.get("castle"):
            tsq = [pl["dest"]]
            if pl["from_file"] is not None and pl["from_rank"] is not None:
                tsq.append(chess.square(pl["from_file"], pl["from_rank"]))
        fid_touch.append(float(np.mean([dec[q] == truth[q] for q in tsq]))
                         if tsq else fid_all[-1])
        touch_ok.append(1 if tsq else 0)
        y.append(0 if r["legal"] else 1)
        gy.append(r["g"])
        if r["legal"]:
            continue
        bb = board_from_occ(dec, bool(r["turn"]))
        own.append(float(legal_on(bb, r["san"])))
        pool = by_bucket.get(r["ply"] // 10, [])
        j = i
        for _ in range(8):
            if len(pool) > 1:
                j = pool[int(rng.integers(len(pool)))]
                if j != i:
                    break
        b2 = board_from_occ(rec[j]["dec"], bool(rec[j]["turn"]))
        mis.append(float(legal_on(b2, r["san"])))
        # random control: a uniformly random move, against the decoded board
        f = int(rng.integers(64))
        t = int(rng.integers(64))
        rmv = chess.Move(f, t)
        rnd.append(float(rmv in bb.legal_moves))
        ref.append(len(list(b.legal_moves)) / 1968.0)
        gid.append(r["g"])

    out = {"model": label, "distribution": args.dist,
           "n_scored": len(rec), "n_illegal": len(own),
           "illegal_rate": float(np.mean(y)) if y else None,
           "n_games": int(len(set(gy)))}
    for nm, v in (("own", own), ("mismatched", mis), ("random", rnd),
                  ("legal_fraction_reference", ref)):
        if not v:
            continue
        lo, hi = boot_ci(v, gid)
        out[nm] = {"mean": float(np.mean(v)), "ci_lo": lo, "ci_hi": hi}

    a_all, a_tch = auc(-np.array(fid_all), y), auc(-np.array(fid_touch), y)
    lo_a, hi_a = auc_ci(-np.array(fid_all), y, gy)
    lo_t, hi_t = auc_ci(-np.array(fid_touch), y, gy)
    out["auc_whole_board"] = {"auc": a_all, "ci_lo": lo_a, "ci_hi": hi_a}
    out["auc_move_touched"] = {"auc": a_tch, "ci_lo": lo_t, "ci_hi": hi_t}
    out["auc_difference"] = a_tch - a_all
    out["frac_destination_parsed"] = float(np.mean(touch_ok))

    print(f"\n[{label}] dist={args.dist}  {len(rec):,} scored, "
          f"{len(own)} illegal (rate {out['illegal_rate']:.4f}) "
          f"from {out['n_games']} games")
    print(f"\nbelief consistency (illegal move legal on the decoded board)")
    for nm in ("own", "mismatched", "random", "legal_fraction_reference"):
        if nm in out:
            r = out[nm]
            print(f"  {nm:>26} {r['mean']:.4f} "
                  f"[{r['ci_lo']:.4f}, {r['ci_hi']:.4f}]")
    print(f"\nAUC predicting an illegal move from probe fidelity")
    print(f"  whole board   {a_all:.4f} [{lo_a:.4f}, {hi_a:.4f}]")
    print(f"  move-touched  {a_tch:.4f} [{lo_t:.4f}, {hi_t:.4f}]")
    print(f"  difference    {a_tch - a_all:+.4f}")
    print(f"  (destination square recovered for "
          f"{100*np.mean(touch_ok):.1f}% of sites; move-touched here means the "
          f"destination, since SAN does not name a source)")

    json.dump(out, open(os.path.join(RES, f"{safe}_belief_{args.dist}.json"),
                        "w"), indent=1)


if __name__ == "__main__":
    main()
