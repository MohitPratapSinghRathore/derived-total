"""Does a capacity-limited model triage its board state?

Two results point the same way. Width buys seven times the state fidelity that
depth does, and a nonlinear decoder recovers only a flat few points over a linear
one at every depth. Together they say the state is not entangled but genuinely
absent, and that what binds is room to hold it rather than steps to compute it.

If room is what binds, a model under capacity pressure should not forget the
board uniformly. It should spend what it has on the squares the next move needs
and let the rest go. We measure that directly, as the difference in probe
accuracy between the two squares the played move touches and the other occupied
squares, and call it selectivity.

The prediction that makes this a test rather than a description is comparative
and runs against the obvious intuition that better models are better at
everything. Holding depth fixed, a narrower model has less room and must triage
harder, so selectivity should be LARGER in the narrow model even though its
overall fidelity is lower. Holding width fixed, adding depth adds no room, so
selectivity should barely move. If instead selectivity is flat across the ladder,
or rises with width, the capacity account is wrong.
"""
import os, json, argparse
import numpy as np
import torch
import chess
from probes import load_split, build, fit_probes, hiddens

DATA = os.path.join(os.path.dirname(__file__), "..", "data", "proc")
RUNS = os.path.join(os.path.dirname(__file__), "..", "runs")
RES = os.path.join(os.path.dirname(__file__), "..", "results")
BUCKETS = [(0, 10), (10, 20), (20, 30), (30, 40), (40, 50),
           (50, 60), (60, 80), (80, 120)]
NCLS = 13


@torch.no_grad()
def score(model, a, probe, layer, toks, lens, occ, itos, bs=32):
    nb = len(BUCKETS)
    bidx = np.full(toks.shape[1], -1)
    for bi, (lo, hi) in enumerate(BUCKETS):
        bidx[lo:hi] = bi
    rel_h = np.zeros(nb); rel_n = np.zeros(nb)
    irr_h = np.zeros(nb); irr_n = np.zeros(nb)
    per_game = {}
    N, L = toks.shape
    for i in range(0, N, bs):
        idx = np.arange(i, min(i + bs, N))
        x = torch.from_numpy(toks[idx].astype(np.int64)).cuda()
        _, hs = hiddens(model, x)
        pred = probe(hs[layer]).reshape(len(idx), L, 64, NCLS
                                        ).argmax(-1).cpu().numpy()
        st = np.asarray(occ[idx]).astype(np.int64)
        for r, gi in enumerate(idx):
            T = int(lens[gi])
            for t in range(T - 1):
                bi = bidx[t]
                if bi < 0:
                    continue
                try:
                    mv = chess.Move.from_uci(itos[int(toks[gi, t + 1])])
                except Exception:
                    continue
                board = st[r, t]
                occm = board > 0
                if occm.sum() < 4:
                    continue
                rel = np.zeros(64, bool)
                rel[mv.from_square] = True
                rel[mv.to_square] = True
                ok = pred[r, t] == board
                a_m = occm & rel
                b_m = occm & ~rel
                if a_m.sum() and b_m.sum():
                    ra, rb = float(ok[a_m].mean()), float(ok[b_m].mean())
                    rel_h[bi] += ra; rel_n[bi] += 1
                    irr_h[bi] += rb; irr_n[bi] += 1
                    per_game.setdefault(int(gi), []).append(ra - rb)
    return (rel_h / np.maximum(rel_n, 1), irr_h / np.maximum(irr_n, 1),
            rel_n, per_game)


def boot(per_game, n_boot=500, seed=0):
    rng = np.random.default_rng(seed)
    g = sorted(per_game)
    v = [np.mean(np.concatenate([per_game[k] for k in
         rng.choice(g, len(g), replace=True)])) for _ in range(n_boot)]
    return float(np.mean(v)), float(np.percentile(v, 2.5)), \
        float(np.percentile(v, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--names", nargs="+",
                    default=["attn_8L256_s0", "attn_8L384_s0", "attn_12L256_s0"])
    ap.add_argument("--fit_games", type=int, default=2500)
    ap.add_argument("--eval_games", type=int, default=1200)
    args = ap.parse_args()

    stoi = json.load(open(os.path.join(DATA, "vocab.json")))
    itos = {v: k for k, v in stoi.items()}
    ftk, flen, focc = load_split("train_probe", args.fit_games)
    etk, elen, eocc = load_split("eval", args.eval_games)

    allout = {}
    for name in args.names:
        print(f"\n=== {name}", flush=True)
        model, a, ck = build(os.path.join(RUNS, f"{name}.pt"))
        res = json.load(open(os.path.join(RES, f"{name}.json")))
        layer = res["best_layer"]
        probes = fit_probes(model, a, ftk, flen, focc, epochs=3)
        rel, irr, n, pg = score(model, a, probes[layer], layer,
                                etk, elen, eocc, itos)
        m, lo, hi = boot(pg)
        allout[name] = {"width": a["width"], "layers": a["layers"],
                        "layer": layer, "buckets": BUCKETS,
                        "rel": rel.tolist(), "irr": irr.tolist(),
                        "n": n.tolist(),
                        "selectivity": {"mean": m, "ci_lo": lo, "ci_hi": hi}}
        print(f"  {'ply':>9} {'move-touched':>13} {'other occupied':>15} "
              f"{'selectivity':>12}")
        for i, (a_, b_) in enumerate(BUCKETS):
            if n[i] < 50:
                continue
            print(f"  {str(a_)+'-'+str(b_):>9} {rel[i]:13.4f} {irr[i]:15.4f} "
                  f"{rel[i]-irr[i]:+12.4f}")
        print(f"  overall selectivity {m:+.4f} [{lo:+.4f}, {hi:+.4f}]")

    json.dump(allout, open(os.path.join(RES, "selectivity.json"), "w"), indent=1)
    print("\n=== capacity prediction")
    for k, v in allout.items():
        s = v["selectivity"]
        print(f"  {k:18s} d={v['width']:4d} L={v['layers']:3d}  "
              f"selectivity {s['mean']:+.4f} [{s['ci_lo']:+.4f}, {s['ci_hi']:+.4f}]")


if __name__ == "__main__":
    main()
