"""Week-0 go/no-go for the decomposition instrument.

The decomposition's failure mode is not that repair works weakly. It is that
naturally occurring errors come from diffusely wrong beliefs, in which case the
"unresolved" bucket swallows everything and the instrument measures nothing. That
is decidable now, from probe and oracle alone, with no edits and no training.

Three measurements, in increasing order of how much they decide:

  sparsity     how many of the 64 squares the probe gets wrong when the model's
               top-1 move is illegal, against the same distribution when it is
               legal. Repair is only tractable if this is small.

  consistency  whether the illegal move is legal on the probe-decoded board,
               which is Paper 1's belief-consistency rate recomputed on this
               population as a cross-check that the two agree.

  policy share among illegal moves, the fraction whose belief is already correct
               at the two squares the move touches. Those are failures repair
               cannot fix by construction, so this previews S2's policy bucket
               and bounds what S1 can possibly claim.

Paper 1's belief consistency of 0.355 gives the memory-ish share a predicted
value, so this is a test rather than a description. A policy share near one means
the instrument has nothing to separate and the reconciliation paper is the honest
outcome.
"""
import os, json, argparse
import numpy as np
import torch
import chess
from probes import load_split, build, fit_probes, hiddens

DATA = os.path.join(os.path.dirname(__file__), "..", "data", "proc")
RUNS = os.path.join(os.path.dirname(__file__), "..", "runs")
RES = os.path.join(os.path.dirname(__file__), "..", "results")
NCLS = 13
STRATA = [(20, 40), (40, 60), (60, 120)]


def cluster_ci(vals_by_game, n_boot=500, seed=0):
    rng = np.random.default_rng(seed)
    g = sorted(vals_by_game)
    if not g:
        return (float("nan"),) * 3
    b = [float(np.mean(np.concatenate([vals_by_game[k] for k in
         rng.choice(g, len(g), replace=True)]))) for _ in range(n_boot)]
    return float(np.mean(b)), float(np.percentile(b, 2.5)), \
        float(np.percentile(b, 97.5))


@torch.no_grad()
def run(model, a, probe, layer, toks, lens, occ, itos, minply, bs=24):
    rows = []
    N, L = toks.shape
    for i in range(0, N, bs):
        idx = np.arange(i, min(i + bs, N))
        x = torch.from_numpy(toks[idx].astype(np.int64)).cuda()
        logits, hs = hiddens(model, x)
        pred_board = probe(hs[layer]).reshape(len(idx), L, 64, NCLS
                                              ).argmax(-1).cpu().numpy()
        top1 = logits.argmax(-1).cpu().numpy()
        st = np.asarray(occ[idx]).astype(np.int64)
        for r, gi in enumerate(idx):
            T = int(lens[gi])
            board = chess.Board()
            for t in range(T - 1):
                if t >= minply:
                    truth = st[r, t]
                    bel = pred_board[r, t]
                    uci = itos.get(int(top1[r, t]), "")
                    try:
                        mv = chess.Move.from_uci(uci)
                        legal = mv in board.legal_moves
                    except Exception:
                        mv, legal = None, False
                    if mv is not None:
                        ndiv = int((bel != truth).sum())
                        act = [mv.from_square, mv.to_square]
                        act_ok = bool((bel[act] == truth[act]).all())
                        bel_legal = -1
                        if not legal:
                            bb = chess.Board()
                            bb.clear()
                            bb.turn = board.turn
                            for sq in range(64):
                                c = int(bel[sq])
                                if c > 0:
                                    pt = (c - 1) % 6 + 1
                                    col = chess.WHITE if c <= 6 else chess.BLACK
                                    bb.set_piece_at(sq, chess.Piece(pt, col))
                            try:
                                bel_legal = int(mv in bb.legal_moves)
                            except Exception:
                                bel_legal = 0
                        rows.append((gi, t, int(legal), ndiv, int(act_ok),
                                     bel_legal))
                try:
                    board.push_uci(itos[int(toks[gi, t + 1])])
                except Exception:
                    break
    return np.array(rows, dtype=np.int64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="attn_8L256_s0")
    ap.add_argument("--fit_games", type=int, default=2500)
    ap.add_argument("--eval_games", type=int, default=1200)
    ap.add_argument("--minply", type=int, default=20)
    args = ap.parse_args()

    stoi = json.load(open(os.path.join(DATA, "vocab.json")))
    itos = {v: k for k, v in stoi.items()}
    model, a, ck = build(os.path.join(RUNS, f"{args.name}.pt"))
    layer = json.load(open(os.path.join(RES, f"{args.name}.json")))["best_layer"]
    ftk, flen, focc = load_split("train_probe", args.fit_games)
    probes = fit_probes(model, a, ftk, flen, focc, epochs=3)
    etk, elen, eocc = load_split("eval", args.eval_games)
    R = run(model, a, probes[layer], layer, etk, elen, eocc, itos, args.minply)

    gi, t, legal, ndiv, act_ok, bel_legal = (R[:, k] for k in range(6))
    ill = legal == 0
    out = {"name": args.name, "layer": layer, "n": int(len(R)),
           "n_games": int(len(np.unique(gi))),
           "illegal_rate": float(ill.mean()), "strata": {}}

    print(f"\n[{args.name}] {len(R):,} positions from "
          f"{len(np.unique(gi)):,} games, illegal top-1 rate {ill.mean():.4f}")

    print("\ndivergent squares of 64, when top-1 is:")
    for nm, m in (("legal", ~ill), ("illegal", ill)):
        d = ndiv[m]
        print(f"  {nm:8s} n={m.sum():7,}  median {np.median(d):5.1f}  "
              f"mean {d.mean():6.2f}  p90 {np.percentile(d,90):5.1f}   "
              f"<=2 sq {float((d<=2).mean()):.3f}  <=5 sq {float((d<=5).mean()):.3f}")
        out[f"ndiv_{nm}"] = {"median": float(np.median(d)),
                             "mean": float(d.mean()),
                             "p90": float(np.percentile(d, 90)),
                             "frac_le2": float((d <= 2).mean()),
                             "frac_le5": float((d <= 5).mean()),
                             "n": int(m.sum())}

    # The coupling test. Global divergence barely separates legal from illegal
    # moves, which is Balogh's weak coupling. But repair does not act globally;
    # it acts at the two squares the move touches. So the question that decides
    # whether the instrument has anything to grip is whether belief at THOSE
    # squares separates the two populations, when whole-board divergence does
    # not. Coupling may be local even where it is globally absent.
    print("\nbelief correct at both move squares, by outcome:")
    for nm, m in (("legal", ~ill), ("illegal", ill)):
        g_ = gi[m]
        d = {int(g): act_ok[m][g_ == g].astype(float) for g in np.unique(g_)}
        mm, l_, h_ = cluster_ci(d)
        out[f"act_ok_{nm}"] = {"mean": mm, "ci_lo": l_, "ci_hi": h_}
        print(f"  {nm:8s} {mm:.4f} [{l_:.4f}, {h_:.4f}]")
    lift = out["act_ok_legal"]["mean"] - out["act_ok_illegal"]["mean"]
    out["act_ok_lift"] = float(lift)
    dg = out["ndiv_illegal"]["mean"] - out["ndiv_legal"]["mean"]
    out["global_div_gap"] = float(dg)
    print(f"  local separation  {lift:+.4f}   "
          f"global divergence gap {dg:+.3f} squares of 64")

    gi_i = gi[ill]
    by = {int(g): act_ok[ill][gi_i == g].astype(float) for g in np.unique(gi_i)}
    m_, lo, hi = cluster_ci(by)
    out["policy_share"] = {"mean": m_, "ci_lo": lo, "ci_hi": hi}
    bl = bel_legal[ill]
    by2 = {int(g): (bl[gi_i == g] == 1).astype(float) for g in np.unique(gi_i)}
    mb, blo, bhi = cluster_ci(by2)
    out["belief_consistency"] = {"mean": mb, "ci_lo": blo, "ci_hi": bhi}

    print("\namong illegal top-1 moves:")
    print(f"  belief correct at both move squares (policy bucket)  "
          f"{m_:.4f} [{lo:.4f}, {hi:.4f}]")
    print(f"  move is legal on the probe-decoded board             "
          f"{mb:.4f} [{blo:.4f}, {bhi:.4f}]   (Paper 1 reported 0.355)")

    print("\nby depth, among illegal top-1:")
    print(f"{'ply':>9} {'n':>8} {'med div':>9} {'<=5 sq':>8} "
          f"{'policy':>9} {'bel-legal':>10}")
    for lo_, hi_ in STRATA:
        m = ill & (t >= lo_) & (t < hi_)
        if m.sum() < 30:
            continue
        out["strata"][f"{lo_}-{hi_}"] = {
            "n": int(m.sum()), "median_div": float(np.median(ndiv[m])),
            "frac_le5": float((ndiv[m] <= 5).mean()),
            "policy_share": float(act_ok[m].mean()),
            "belief_consistency": float((bel_legal[m] == 1).mean())}
        print(f"{str(lo_)+'-'+str(hi_):>9} {m.sum():8,} "
              f"{np.median(ndiv[m]):9.1f} {float((ndiv[m]<=5).mean()):8.3f} "
              f"{float(act_ok[m].mean()):9.4f} "
              f"{float((bel_legal[m]==1).mean()):10.4f}")

    json.dump(out, open(os.path.join(RES, f"{args.name}_natdiv.json"), "w"),
              indent=1)
    print("\nGO/NO-GO: repair is tractable only if divergence is sparse and "
          "the policy share is well below 1.")


if __name__ == "__main__":
    main()
