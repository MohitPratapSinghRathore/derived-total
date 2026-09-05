"""How correct is the board we claim to have installed?

The induced control showed repair restores the decode at only 0.68 of the squares
it targets, so the sentence "we handed the model the correct board" cannot be
written literally. This measures the whole board rather than the targeted squares,
before and after installing true occupancy at every divergent square, on the
natural failures S2 classified.

Reported on all 64 squares and on occupied squares alone, because empty squares
dominate the first and inflate it.
"""
import os, json, argparse
import numpy as np
import torch
import chess
from probes import load_split, build

DATA = os.path.join(os.path.dirname(__file__), "..", "data", "proc")
RUNS = os.path.join(os.path.dirname(__file__), "..", "runs")
RES = os.path.join(os.path.dirname(__file__), "..", "results")


def boot_ci(v, g, n_boot=500, seed=0):
    rng = np.random.default_rng(seed)
    v = np.asarray(v, float)
    g = np.asarray(g)
    u = np.unique(g)
    by = {k: np.where(g == k)[0] for k in u}
    o = [v[np.concatenate([by[k] for k in
         rng.choice(u, len(u), replace=True)])].mean() for _ in range(n_boot)]
    return float(np.percentile(o, 2.5)), float(np.percentile(o, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="attn_8L256_s0")
    ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--minply", type=int, default=20)
    ap.add_argument("--games", type=int, default=3000)
    ap.add_argument("--mode", default="sum_renorm",
                    choices=["sum_renorm", "per_square", "per_square_sqrt"],
                    help="sum_renorm sums unit directions and renormalises the "
                         "total to one calibrated magnitude, so each of k "
                         "squares receives about 1/k of the strength the "
                         "validated two-square edit delivers. per_square gives "
                         "every square the full calibrated magnitude. "
                         "per_square_sqrt splits the budget as 1/sqrt(k).")
    ap.add_argument("--max_squares", type=int, default=64)
    args = ap.parse_args()

    model, a, ck = build(os.path.join(RUNS, f"{args.name}.pt"))
    bl = json.load(open(os.path.join(RES, f"{args.name}.json")))["best_layer"]
    pls = torch.load(os.path.join(RES, f"{args.name}_probes.pt"),
                     weights_only=False)
    probe = torch.nn.Linear(a["width"], 64 * 13).cuda()
    probe.load_state_dict(pls[bl])
    probe.eval()
    W = probe.weight.reshape(64, 13, a["width"]).float()
    stoi = json.load(open(os.path.join(DATA, "vocab.json")))
    itos = {v: k for k, v in stoi.items()}
    toks, lens, occ = load_split("eval", args.games)

    def fwd(x, edit):
        st = {}

        def hook(m, i, o):
            oo = o
            if edit is not None:
                oo = o.clone()
                oo[:, -1] = oo[:, -1] + edit
            st["h"] = oo
            return oo

        hd = model.blocks[bl].register_forward_hook(hook)
        with torch.amp.autocast("cuda", dtype=torch.float16):
            lg, _, _ = model(x)
        hd.remove()
        return torch.softmax(lg.float()[0, -1], -1), \
            probe(st["h"][0, -1].float()).reshape(64, 13).argmax(-1)

    pre_all, post_all, pre_occ, post_occ, tgt, games = [], [], [], [], [], []
    n = 0
    with torch.no_grad():
        for gi in range(len(toks)):
            if n >= args.n:
                break
            T = int(lens[gi])
            if T < args.minply + 4:
                continue
            board = chess.Board()
            for t in range(T - 1):
                if n >= args.n:
                    break
                if t >= args.minply:
                    x = torch.from_numpy(
                        toks[gi:gi + 1, :t + 1].astype(np.int64)).cuda()
                    p0, d0 = fwd(x, None)
                    top1 = int(p0.argmax())
                    try:
                        mv = chess.Move.from_uci(itos.get(top1, ""))
                        ill = mv not in board.legal_moves
                    except Exception:
                        mv, ill = None, False
                    if mv is not None and ill:
                        truth = np.asarray(occ[gi, t]).astype(int)
                        bel = d0.cpu().numpy()
                        div = [s for s in range(64) if bel[s] != truth[s]]
                        div = div[:args.max_squares]
                        if div:
                            with torch.amp.autocast("cuda",
                                                    dtype=torch.float16):
                                _, hs0, _ = model(x, return_hidden=True)
                            rms = float(hs0[bl][0, -1].float()
                                        .pow(2).mean().sqrt())
                            sc = args.alpha * rms * np.sqrt(a["width"])
                            k = max(len(div), 1)
                            per = {"sum_renorm": None,
                                   "per_square": sc,
                                   "per_square_sqrt": sc / np.sqrt(k)}[args.mode]
                            d = torch.zeros(a["width"], device="cuda")
                            for s in div:
                                vv = W[s, int(truth[s])] - W[s, int(bel[s])]
                                if float(vv.norm()) > 1e-6:
                                    u = vv / vv.norm()
                                    d = d + (u if per is None else u * per)
                            if per is None and float(d.norm()) > 1e-6:
                                d = d / d.norm() * sc
                            _, d1 = fwd(x, d)
                            b1 = d1.cpu().numpy()
                            om = truth > 0
                            pre_all.append(float((bel == truth).mean()))
                            post_all.append(float((b1 == truth).mean()))
                            pre_occ.append(float((bel[om] == truth[om]).mean()))
                            post_occ.append(float((b1[om] == truth[om]).mean()))
                            tgt.append(float(np.mean([b1[s] == truth[s]
                                                      for s in div])))
                            games.append(gi)
                            n += 1
                try:
                    board.push_uci(itos[int(toks[gi, t + 1])])
                except Exception:
                    break

    out = {"name": args.name, "mode": args.mode,
           "max_squares": args.max_squares,
           "n": n, "n_games": int(len(np.unique(games)))}
    print(f"\n[{args.name}] decode fidelity before and after installing true "
          f"occupancy at EVERY divergent square")
    print(f"n={n} natural illegal positions from "
          f"{len(np.unique(games))} games\n")
    print(f"{'measure':>28} {'before':>8} {'after':>8}  {'95% CI after':>22}")
    for lab, pre, post in (("all 64 squares", pre_all, post_all),
                           ("occupied squares only", pre_occ, post_occ)):
        lo, hi = boot_ci(post, games)
        out[lab] = {"before": float(np.mean(pre)),
                    "after": float(np.mean(post)), "ci_lo": lo, "ci_hi": hi}
        print(f"{lab:>28} {np.mean(pre):8.4f} {np.mean(post):8.4f}  "
              f"[{lo:.4f}, {hi:.4f}]")
    lo, hi = boot_ci(tgt, games)
    out["repaired_squares"] = {"after": float(np.mean(tgt)),
                               "ci_lo": lo, "ci_hi": hi}
    print(f"{'at the repaired squares':>28} {0.0:8.4f} {np.mean(tgt):8.4f}  "
          f"[{lo:.4f}, {hi:.4f}]")
    tag = f"{args.mode}_{args.max_squares}"
    json.dump(out, open(os.path.join(RES,
              f"{args.name}_repair_fidelity_{tag}.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
