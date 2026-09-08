"""The calibrated single-square edit, on Chess-GPT.

Third of the three load-bearing measurements. The edit writes a square's
occupancy direction into the residual stream and asks whether probability mass
follows: if the decoded board is a belief the policy reads, emptying a square the
side to move occupies should remove mass from moves that start there.

Two things differ from our own models and both are handled rather than assumed.
The edit basis is the probe's weight rows, so it is rebuilt against the probe
fitted on THIS model rather than reused from ours. And the edit site is the
character at which a move begins, not a move token, so the mass being measured is
the probability of the move's first character rather than of a whole move.

That last point forces a change of observable. With one token per move we could
read mass on moves leaving square A directly. Character-level SAN spreads a move
over several tokens, and its first character identifies the piece for pieces and
the destination file for pawns, so "mass leaving A" is not a quantity the
first-character distribution expresses. Instead we measure the total probability
of the first characters consistent with any legal move from A, which is the
closest well-defined analogue, and report the manipulation check separately so a
failed edit is never read as a null effect.

Controls: a norm-matched random direction at the same site and magnitude, and a
strength sweep, since a real representational edit should show a graded response
rather than a step.
"""
import os, json, argparse
import numpy as np
import torch
import torch.nn as nn
import chess
from chessgpt import load, encode, STOI, ITOS
from chessgpt_nano import convert as nano_convert
from chessgpt_data import build, load_games, random_legal_games

ROOT = os.path.join(os.path.dirname(__file__), "..")
RES = os.path.join(ROOT, "results")
EXT = os.path.join(ROOT, "data", "ext")
NCLS = 13


def boot_ci(v, g, n_boot=500, seed=0):
    rng = np.random.default_rng(seed)
    v, g = np.asarray(v, float), np.asarray(g)
    u = np.unique(g)
    by = {k: np.where(g == k)[0] for k in u}
    o = [v[np.concatenate([by[k] for k in
         rng.choice(u, len(u), replace=True)])].mean() for _ in range(n_boot)]
    return float(np.percentile(o, 2.5)), float(np.percentile(o, 97.5))


def first_chars_from(board, sq):
    """First SAN characters consistent with some legal move leaving `sq`."""
    out = set()
    for mv in board.legal_moves:
        if mv.from_square != sq:
            continue
        san = board.san(mv)
        if san and san[0] in STOI:
            out.add(STOI[san[0]])
    return sorted(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--games", type=int, default=1500)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--mults", type=float, nargs="+",
                    default=[0.25, 0.5, 1.0, 2.0, 4.0])
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
    W = probe.weight.reshape(64, NCLS, pk["d"]).float()
    d_model = pk["d"]

    games = (random_legal_games(args.games, np.random.default_rng(0),
                                max_plies=args.maxply)
             if args.dist == "random" else load_games("eval", args.games))
    rng = np.random.default_rng(0)

    # hook the block whose OUTPUT is hidden_states[bl]; hidden_states[0] is the
    # embedding, so hidden_states[i] is the output of block i-1
    if bl == 0:
        raise SystemExit("best layer is the embedding; no block to hook")
    blk = model.transformer.h[bl - 1]

    rec = {"state": [], "rand": [], "flip": [], "game": []}
    sweep = {m: {"d": [], "g": []} for m in args.mults}
    n_done = 0

    def run(ids, pos, edit):
        store = {}

        def hook(mod, inp, out):
            o = out
            if edit is not None:
                if isinstance(o, tuple):
                    h0 = o[0].clone()
                    h0[:, pos] = h0[:, pos] + edit
                    o = (h0,) + o[1:]
                else:
                    o = o.clone()
                    o[:, pos] = o[:, pos] + edit
            store["h"] = o[0] if isinstance(o, tuple) else o
            return o

        hd = blk.register_forward_hook(hook)
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.float16):
            lg = model(torch.tensor([ids], dtype=torch.long,
                                    device="cuda")).logits
        hd.remove()
        h = store["h"][0, pos].float()
        return torch.softmax(lg.float()[0, pos], -1), \
            probe(h).reshape(64, NCLS).argmax(-1)

    with torch.no_grad():
        for gi, g in enumerate(games):
            if n_done >= args.n:
                break
            text, sites = build(g, max_plies=args.maxply)
            ids = encode(text)[:1023]
            for s in sites:
                if n_done >= args.n:
                    break
                if s["ply"] < args.minply or s["char_index"] >= len(ids):
                    continue
                pos = s["char_index"]
                b = chess.Board(s["fen"])
                srcs = sorted({m.from_square for m in b.legal_moves})
                if not srcs:
                    continue
                A = int(rng.choice(srcs))
                piece = int(s["occ"][A])
                if piece == 0:
                    continue
                ids_ctx = ids[:pos + 1]
                cand = first_chars_from(b, A)
                if not cand:
                    continue

                p0, dec0 = run(ids_ctx, pos, None)
                # calibrate to the residual scale at this site
                with torch.amp.autocast("cuda", dtype=torch.float16):
                    hs = model(torch.tensor([ids_ctx], dtype=torch.long,
                                            device="cuda"),
                               output_hidden_states=True).hidden_states
                rms = float(hs[bl][0, pos].float().pow(2).mean().sqrt())
                scale = args.alpha * rms * np.sqrt(d_model)

                v = (W[A, 0] - W[A, piece]).float()
                dvec = v / v.norm() * scale
                r = torch.randn_like(dvec)
                rvec = r / r.norm() * dvec.norm()

                p1, dec1 = run(ids_ctx, pos, dvec)
                p2, _ = run(ids_ctx, pos, rvec)
                m0 = float(p0[cand].sum())
                rec["state"].append(float(p1[cand].sum()) - m0)
                rec["rand"].append(float(p2[cand].sum()) - m0)
                rec["flip"].append(float(int(dec1[A]) == 0))
                rec["game"].append(gi)
                for mlt in args.mults:
                    pm, _ = run(ids_ctx, pos, dvec * mlt)
                    sweep[mlt]["d"].append(float(pm[cand].sum()) - m0)
                    sweep[mlt]["g"].append(gi)
                n_done += 1

    out = {"model": label, "distribution": args.dist, "n": n_done,
           "alpha": args.alpha, "layer": bl,
           "n_games": int(len(set(rec["game"])))}
    for k in ("state", "rand"):
        lo, hi = boot_ci(rec[k], rec["game"])
        out[k] = {"mean": float(np.mean(rec[k])), "ci_lo": lo, "ci_hi": hi}
    out["manipulation_flip"] = float(np.mean(rec["flip"]))
    diff = np.array(rec["state"]) - np.array(rec["rand"])
    lo, hi = boot_ci(diff, rec["game"])
    out["state_minus_random"] = {"mean": float(diff.mean()),
                                 "ci_lo": lo, "ci_hi": hi}
    out["sweep"] = {}
    for mlt in args.mults:
        lo2, hi2 = boot_ci(sweep[mlt]["d"], sweep[mlt]["g"])
        out["sweep"][str(mlt)] = {"mean": float(np.mean(sweep[mlt]["d"])),
                                  "ci_lo": lo2, "ci_hi": hi2}

    print(f"\n[{label}] calibrated edit, dist={args.dist}, n={n_done} sites "
          f"from {out['n_games']} games, layer {bl}, alpha {args.alpha}")
    print(f"  manipulation: square decoded empty after the edit "
          f"{out['manipulation_flip']:.3f}")
    print(f"\n  change in mass on first characters of moves leaving that square")
    for k, lab in (("state", "state edit"), ("rand", "norm-matched random")):
        r = out[k]
        print(f"    {lab:>22} {r['mean']:+.5f} "
              f"[{r['ci_lo']:+.5f}, {r['ci_hi']:+.5f}]")
    r = out["state_minus_random"]
    print(f"    {'difference':>22} {r['mean']:+.5f} "
          f"[{r['ci_lo']:+.5f}, {r['ci_hi']:+.5f}]")
    print(f"\n  strength sweep")
    for mlt in args.mults:
        r = out["sweep"][str(mlt)]
        print(f"    {mlt:5.2f}x  {r['mean']:+.5f} "
              f"[{r['ci_lo']:+.5f}, {r['ci_hi']:+.5f}]")

    json.dump(out, open(os.path.join(RES, f"{safe}_edit_{args.dist}.json"),
                        "w"), indent=1)


if __name__ == "__main__":
    main()
