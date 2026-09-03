"""Constructive causal edit: move a piece in the model's belief from A to B.

The suppression edit shows that removing a square's occupancy direction removes
the mass on moves from it. That is consistent with a causal belief state, and it
is also consistent with generic damage to source-square features, which is the
obvious sceptical reading.

A transfer edit is much harder to explain that way. We take a square A that the
side to move genuinely occupies, and an empty square B, and edit the residual
stream toward "A is empty" AND "B holds the piece that was on A". If the decoded
state is a unified belief the policy reads, mass should leave moves from A and
arrive at moves from B. Generic damage predicts the first and not the second.

Reported, all at the calibrated strength used elsewhere:
  manipulation check   did the decode actually flip at both A and B
  mass out of A        should fall
  mass out of B        should RISE, and this is the constructive part
  norm-matched control a random direction of the same magnitude
"""
import os, json, argparse
import numpy as np
import torch
import chess
from probes import load_split, build

DATA = os.path.join(os.path.dirname(__file__), "..", "data", "proc")
RUNS = os.path.join(os.path.dirname(__file__), "..", "runs")
RES = os.path.join(os.path.dirname(__file__), "..", "results")


def boot_ci(vals, games, n_boot=400, seed=0):
    """Bootstrap over games, since positions within a game are dependent."""
    rng = np.random.default_rng(seed)
    vals = np.asarray(vals, float); games = np.asarray(games)
    uniq = np.unique(games)
    by = {g: np.where(games == g)[0] for g in uniq}
    out = []
    for _ in range(n_boot):
        pick = rng.choice(uniq, len(uniq), replace=True)
        sel = np.concatenate([by[g] for g in pick])
        out.append(vals[sel].mean())
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="attn_8L256_s0")
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--alpha", type=float, default=0.5)
    args = ap.parse_args()

    model, a, c = build(os.path.join(RUNS, f"{args.name}.pt"))
    res = json.load(open(os.path.join(RES, f"{args.name}.json")))
    best = res["best_layer"]
    pls = torch.load(os.path.join(RES, f"{args.name}_probes.pt"), weights_only=False)
    probe = torch.nn.Linear(a["width"], 64 * 13).cuda()
    probe.load_state_dict(pls[best]); probe.eval()
    W = probe.weight.reshape(64, 13, a["width"])

    toks, lens, occ = load_split("eval", 1500)
    stoi = json.load(open(os.path.join(DATA, "vocab.json")))
    itos = {v: k for k, v in stoi.items()}
    rng = np.random.default_rng(0)

    rec = {"outA": [], "outB": [], "outA_rand": [], "outB_rand": [],
           "flipA": [], "flipB": [], "game": []}
    tot = 0

    with torch.no_grad():
        for gi in range(len(toks)):
            if tot >= args.n:
                break
            T = int(lens[gi])
            if T < 25:
                continue
            t = int(rng.integers(12, min(T - 2, 70)))
            x = torch.from_numpy(toks[gi:gi + 1, :t + 1]).cuda()

            board = chess.Board()
            for u in range(1, t + 1):
                board.push_uci(itos[int(toks[gi, u])])

            # A: a square the mover occupies with legal moves.
            srcs = sorted({m.from_square for m in board.legal_moves})
            if not srcs:
                continue
            A = int(rng.choice(srcs))
            piece = int(occ[gi, t, A])
            if piece == 0:
                continue
            # B: an empty square from which that same piece type would have moves
            empties = [q for q in range(64) if int(occ[gi, t, q]) == 0]
            if not empties:
                continue
            B = int(rng.choice(empties))

            def run(edit):
                store = {}

                def hook(mod, inp, out):
                    o = out
                    if edit is not None:
                        o = o.clone()
                        o[:, -1] = o[:, -1] + edit
                    store["h"] = o
                    return o
                hd = model.blocks[best].register_forward_hook(hook)
                with torch.amp.autocast("cuda", dtype=torch.float16):
                    lg, _, _ = model(x)
                hd.remove()
                h = store["h"][0, -1].float()
                dec = probe(h).reshape(64, 13).argmax(-1)
                return torch.softmax(lg.float()[0, -1], -1), dec

            with torch.amp.autocast("cuda", dtype=torch.float16):
                _, hs0, _ = model(x, return_hidden=True)
            rms = float(hs0[best][0, -1].float().pow(2).mean().sqrt())
            scale = args.alpha * rms * np.sqrt(a["width"])

            # transfer: empty A, and put the piece on B
            dA = (W[A, 0] - W[A, piece]).float()
            dB = (W[B, piece] - W[B, 0]).float()
            d = dA / dA.norm() + dB / dB.norm()
            d = d / d.norm() * scale
            r = torch.randn_like(d); r = r / r.norm() * d.norm()

            p0, dec0 = run(None)
            p1, dec1 = run(d)
            p2, _ = run(r)

            # mass on moves leaving A and leaving B, under the true board's
            # move vocabulary; B-moves are illegal now, which is the point.
            idsA = [stoi[m.uci()] for m in board.legal_moves
                    if m.from_square == A and m.uci() in stoi]
            idsB = [j for u, j in stoi.items()
                    if len(u) >= 4 and u[:2] == chess.square_name(B)]
            if not idsA or not idsB:
                continue

            rec["outA"].append(float(p1[idsA].sum()) - float(p0[idsA].sum()))
            rec["outB"].append(float(p1[idsB].sum()) - float(p0[idsB].sum()))
            rec["outA_rand"].append(float(p2[idsA].sum()) - float(p0[idsA].sum()))
            rec["outB_rand"].append(float(p2[idsB].sum()) - float(p0[idsB].sum()))
            rec["flipA"].append(int(dec1[A].item() == 0))
            rec["flipB"].append(int(dec1[B].item() == piece))
            rec["game"].append(gi)
            tot += 1

    g = rec["game"]
    out = {"name": args.name, "alpha": args.alpha, "n": tot,
           "manipulation": {"flip_A_to_empty": float(np.mean(rec["flipA"])),
                            "flip_B_to_piece": float(np.mean(rec["flipB"]))}}
    both = np.array(rec["flipA"], bool) & np.array(rec["flipB"], bool)
    out["n_both_flipped"] = int(both.sum())
    for k in ("outA", "outB", "outA_rand", "outB_rand"):
        v = np.array(rec[k], float)
        m = float(v.mean()); lo, hi = boot_ci(v, g)
        out[k] = {"mean_delta": m, "ci_lo": lo, "ci_hi": hi,
                  "frac_positive": float((v > 0).mean())}
        # the effect where the manipulation demonstrably worked
        if both.sum() > 30:
            vb = v[both]; gb = np.array(g)[both]
            lo2, hi2 = boot_ci(vb, gb)
            out[k]["both_flipped"] = {
                "mean_delta": float(vb.mean()), "ci_lo": lo2, "ci_hi": hi2,
                "frac_positive": float((vb > 0).mean()), "n": int(both.sum())}
    json.dump(out, open(os.path.join(RES, f"{args.name}_transfer.json"), "w"),
              indent=1)

    print(f"[{args.name}] transfer edit A -> B, n={tot}, alpha={args.alpha}")
    print(f"  manipulation: A decoded empty {out['manipulation']['flip_A_to_empty']:.3f}, "
          f"B decoded as the piece {out['manipulation']['flip_B_to_piece']:.3f}")
    for k, lab in (("outA", "mass out of A, state edit"),
                   ("outA_rand", "mass out of A, random edit"),
                   ("outB", "mass out of B, state edit"),
                   ("outB_rand", "mass out of B, random edit")):
        v = out[k]
        bf = v.get("both_flipped")
        extra = (f"   | both flipped n={bf['n']}: {bf['mean_delta']:+.4f} "
                 f"[{bf['ci_lo']:+.4f}, {bf['ci_hi']:+.4f}]") if bf else ""
        print(f"  {lab:32s} {v['mean_delta']:+.4f} "
              f"[{v['ci_lo']:+.4f}, {v['ci_hi']:+.4f}]  "
              f"pos {v['frac_positive']:.2f}{extra}")


if __name__ == "__main__":
    main()
