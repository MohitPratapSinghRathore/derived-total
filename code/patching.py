"""Causal check: is the probe reading a representation the model USES?

Take the layer-best probe's decoder directions for a square, and push the
residual stream along (target_class - current_class) at one position.  If the
model's move distribution shifts consistently with the counterfactual position
-- e.g. moves originating from a square we emptied lose mass -- the state
representation is causally load-bearing, not a decorative correlate.
"""
import os, json, argparse
import numpy as np
import torch
import chess
from probes import load_split, build

DATA = os.path.join(os.path.dirname(__file__), "..", "data", "proc")
RUNS = os.path.join(os.path.dirname(__file__), "..", "runs")
RES = os.path.join(os.path.dirname(__file__), "..", "results")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--alpha", type=float, default=4.0)
    args = ap.parse_args()

    model, a, c = build(os.path.join(RUNS, f"{args.name}.pt"))
    res = json.load(open(os.path.join(RES, f"{args.name}.json")))
    best = res["best_layer"]
    pl = torch.load(os.path.join(RES, f"{args.name}_probes.pt"), weights_only=False)
    probe = torch.nn.Linear(a["width"], 64 * 13).cuda()
    probe.load_state_dict(pl[best])
    W = probe.weight.reshape(64, 13, a["width"])     # class directions per square

    toks, lens, occ = load_split("eval", 600)
    stoi = json.load(open(os.path.join(DATA, "vocab.json")))
    itos = {v: k for k, v in stoi.items()}

    hooked = {}

    def hook(mod, inp, out):
        hooked["h"] = out
        return out

    hits = tot = 0
    drops = []
    rng = np.random.default_rng(0)
    with torch.no_grad():
        for gi in range(len(toks)):
            if tot >= args.n:
                break
            T = int(lens[gi])
            t = int(rng.integers(15, max(16, min(T - 2, 60))))
            x = torch.from_numpy(toks[gi:gi + 1, :t + 1]).cuda()

            # replay to the true position at t
            board = chess.Board()
            for u in range(1, t + 1):
                board.push_uci(itos[int(toks[gi, u])])
            # pick an occupied square of the side to move that has legal moves
            srcs = sorted({m.from_square for m in board.legal_moves})
            if not srcs:
                continue
            sq = int(rng.choice(srcs))

            def run(edit=None):
                acts = {}

                def h_(mod, i_, o_):
                    if edit is not None:
                        o_ = o_.clone()
                        o_[:, -1] += edit
                    acts["o"] = o_
                    return o_
                hd = model.blocks[best].register_forward_hook(h_)
                with torch.amp.autocast("cuda", dtype=torch.float16):
                    lg, _, _ = model(x)
                hd.remove()
                return torch.softmax(lg.float()[0, -1], -1)

            p0 = run()
            cur = int(occ[gi, t, sq])
            # counterfactual: empty that square (class 0)
            d = (W[sq, 0] - W[sq, cur]).float()
            d = d / d.norm() * args.alpha * torch.tensor(
                float(np.linalg.norm(np.ones(a["width"]))), device=d.device) / np.sqrt(a["width"])
            p1 = run(edit=d)

            # mass on moves originating from the emptied square
            ids = [stoi[m.uci()] for m in board.legal_moves
                   if m.from_square == sq and m.uci() in stoi]
            if not ids:
                continue
            m0 = float(p0[ids].sum()); m1 = float(p1[ids].sum())
            drops.append(m0 - m1)
            hits += (m1 < m0)
            tot += 1

    drops = np.array(drops)
    out = {"name": args.name, "layer": best, "alpha": args.alpha, "n": int(tot),
           "frac_mass_decreased": float(hits / max(tot, 1)),
           "mean_mass_drop": float(drops.mean()),
           "median_mass_drop": float(np.median(drops))}
    print(out, flush=True)
    json.dump(out, open(os.path.join(RES, f"{args.name}_patch.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
