"""Positive control: corrupt a belief ourselves, then repair it.

S2 found that correcting the model's state fixes roughly seven percent of natural
failures, and that installing the entire correct board barely improves on that.
That sentence only survives review if the repair channel demonstrably works. If
induced corruptions are also unrecoverable, the instrument is broken and the
finding is an artefact of a weak edit.

So we take positions where the model's top-1 move is legal and its belief is
already correct at both squares that move touches, corrupt the belief at those
squares, confirm the corruption pushed the model into an illegal move, and then
repair by the same procedure S1 and S2 use: read the decoded belief, compare to
the oracle, install true occupancy at the squares that disagree.

What this does and does not establish. It is a ceiling on the channel, not an
independent test of it. The repair direction is computed from the corrupted
decode, and where the decode lands exactly on the class we corrupted toward, the
repair is close to the algebraic inverse of the corruption, so a high recovery
rate is partly guaranteed by construction. That is the point: it bounds what the
edit mechanism can achieve at best, and the natural rate is then read against
that ceiling rather than against one hundred percent.

Reported against a norm-matched random repair, so recovery cannot be credited to
the mere act of perturbing an already-corrupted state.
"""
import os, json, argparse
import numpy as np
import torch
import chess
from probes import load_split, build

DATA = os.path.join(os.path.dirname(__file__), "..", "data", "proc")
RUNS = os.path.join(os.path.dirname(__file__), "..", "runs")
RES = os.path.join(os.path.dirname(__file__), "..", "results")


def boot_ci(vals, games, n_boot=500, seed=0):
    rng = np.random.default_rng(seed)
    vals = np.asarray(vals, float)
    games = np.asarray(games)
    uniq = np.unique(games)
    by = {g: np.where(games == g)[0] for g in uniq}
    out = []
    for _ in range(n_boot):
        sel = np.concatenate([by[g] for g in
                              rng.choice(uniq, len(uniq), replace=True)])
        out.append(vals[sel].mean())
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="attn_8L256_s0")
    ap.add_argument("--n", type=int, default=700)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--minply", type=int, default=20)
    ap.add_argument("--games", type=int, default=3000)
    args = ap.parse_args()

    model, a, ck = build(os.path.join(RUNS, f"{args.name}.pt"))
    res = json.load(open(os.path.join(RES, f"{args.name}.json")))
    bl = res["best_layer"]
    pls = torch.load(os.path.join(RES, f"{args.name}_probes.pt"),
                     weights_only=False)
    probe = torch.nn.Linear(a["width"], 64 * 13).cuda()
    probe.load_state_dict(pls[bl])
    probe.eval()
    W = probe.weight.reshape(64, 13, a["width"]).float()

    stoi = json.load(open(os.path.join(DATA, "vocab.json")))
    itos = {v: k for k, v in stoi.items()}
    toks, lens, occ = load_split("eval", args.games)
    rng = np.random.default_rng(0)

    def forward(x, edit):
        store = {}

        def hook(mod, inp, out):
            o = out
            if edit is not None:
                o = o.clone()
                o[:, -1] = o[:, -1] + edit
            store["h"] = o
            return o

        hd = model.blocks[bl].register_forward_hook(hook)
        with torch.amp.autocast("cuda", dtype=torch.float16):
            lg, _, _ = model(x)
        hd.remove()
        return torch.softmax(lg.float()[0, -1], -1), \
            probe(store["h"][0, -1].float()).reshape(64, 13).argmax(-1)

    rec = {"corrupted_illegal": [], "recovered": [], "recovered_rand": [],
           "corrupt_took": [], "repair_decode_ok": [], "game": []}
    n_done = 0
    with torch.no_grad():
        for gi in range(len(toks)):
            if n_done >= args.n:
                break
            T = int(lens[gi])
            if T < args.minply + 4:
                continue
            board = chess.Board()
            for t in range(T - 1):
                if n_done >= args.n:
                    break
                if t >= args.minply:
                    x = torch.from_numpy(
                        toks[gi:gi + 1, :t + 1].astype(np.int64)).cuda()
                    p0, dec0 = forward(x, None)
                    top1 = int(p0.argmax())
                    try:
                        mv = chess.Move.from_uci(itos.get(top1, ""))
                        legal = mv in board.legal_moves
                    except Exception:
                        mv, legal = None, False
                    if mv is not None and legal:
                        truth = np.asarray(occ[gi, t]).astype(int)
                        bel = dec0.cpu().numpy()
                        act = [mv.from_square, mv.to_square]
                        if all(bel[s] == truth[s] for s in act):
                            with torch.amp.autocast("cuda",
                                                    dtype=torch.float16):
                                _, hs0, _ = model(x, return_hidden=True)
                            rms = float(hs0[bl][0, -1].float()
                                        .pow(2).mean().sqrt())
                            scale = args.alpha * rms * np.sqrt(a["width"])

                            # corrupt: tell it the source square is empty
                            src = mv.from_square
                            cls_true = int(truth[src])
                            if cls_true == 0:
                                board.push_uci(itos[int(toks[gi, t + 1])])
                                continue
                            v = W[src, 0] - W[src, cls_true]
                            dc = v / v.norm() * scale
                            p1, dec1 = forward(x, dc)
                            b1 = dec1.cpu().numpy()
                            took = int(b1[src] != cls_true)
                            nt1 = int(p1.argmax())
                            try:
                                m1 = chess.Move.from_uci(itos.get(nt1, ""))
                                ill1 = m1 not in board.legal_moves
                            except Exception:
                                m1, ill1 = None, True

                            if took and ill1:
                                # repair by the S1/S2 procedure, from the
                                # corrupted decode against the oracle
                                # m1 is None when the corrupted top-1 does not
                                # parse as a move. Guard on the value, not on
                                # the name: an earlier version tested
                                # `'m1' in dir()`, which is true even when m1
                                # holds a stale move from a previous position,
                                # and would then repair the wrong squares.
                                a1 = ([m1.from_square, m1.to_square]
                                      if m1 is not None else act)
                                divs = [s for s in set(act + a1)
                                        if b1[s] != truth[s]]
                                d = torch.zeros(a["width"], device="cuda")
                                for s in divs:
                                    vv = W[s, int(truth[s])] - W[s, int(b1[s])]
                                    if float(vv.norm()) > 1e-6:
                                        d = d + vv / vv.norm()
                                if float(d.norm()) > 1e-6:
                                    d = d / d.norm() * scale
                                p2, dec2 = forward(x, dc + d)
                                nt2 = int(p2.argmax())
                                try:
                                    ok2 = chess.Move.from_uci(
                                        itos.get(nt2, "")) in board.legal_moves
                                except Exception:
                                    ok2 = False
                                r_ = torch.randn_like(d)
                                r_ = r_ / r_.norm() * d.norm()
                                p3, _ = forward(x, dc + r_)
                                nt3 = int(p3.argmax())
                                try:
                                    ok3 = chess.Move.from_uci(
                                        itos.get(nt3, "")) in board.legal_moves
                                except Exception:
                                    ok3 = False
                                b2 = dec2.cpu().numpy()
                                rec["recovered"].append(float(ok2))
                                rec["recovered_rand"].append(float(ok3))
                                rec["repair_decode_ok"].append(
                                    float(np.mean([b2[s] == truth[s]
                                                   for s in divs]))
                                    if divs else 1.0)
                                rec["game"].append(gi)
                                n_done += 1
                            rec["corrupt_took"].append(float(took))
                            rec["corrupted_illegal"].append(float(ill1))
                try:
                    board.push_uci(itos[int(toks[gi, t + 1])])
                except Exception:
                    break

    g = rec["game"]
    out = {"name": args.name, "alpha": args.alpha,
           "n_recovery": int(len(rec["recovered"])),
           "n_games": int(len(np.unique(g))),
           "corruption_took_decode": float(np.mean(rec["corrupt_took"])),
           "corruption_caused_illegal": float(np.mean(rec["corrupted_illegal"])),
           "natural_comparison": {"memory_plus_mixed_partial": 0.1143,
                                  "memory_plus_mixed_full": 0.1271}}
    for k in ("recovered", "recovered_rand", "repair_decode_ok"):
        v = np.array(rec[k], float)
        lo, hi = boot_ci(v, g)
        out[k] = {"mean": float(v.mean()), "ci_lo": lo, "ci_hi": hi}

    d = np.array(rec["recovered"]) - np.array(rec["recovered_rand"])
    lo, hi = boot_ci(d, g)
    out["recovered_minus_random"] = {"mean": float(d.mean()),
                                     "ci_lo": lo, "ci_hi": hi}
    json.dump(out, open(os.path.join(RES, f"{args.name}_induced.json"), "w"),
              indent=1)

    print(f"\n[{args.name}] induced corruption and repair, "
          f"n={out['n_recovery']} recoverable cases from "
          f"{out['n_games']} games")
    print(f"  corruption changed the decode          "
          f"{out['corruption_took_decode']:.4f}")
    print(f"  corruption made the top-1 illegal      "
          f"{out['corruption_caused_illegal']:.4f}")
    print(f"\n  RECOVERY, repair restores a legal top-1  "
          f"{out['recovered']['mean']:.4f} "
          f"[{out['recovered']['ci_lo']:.4f}, {out['recovered']['ci_hi']:.4f}]")
    print(f"  norm-matched random repair               "
          f"{out['recovered_rand']['mean']:.4f} "
          f"[{out['recovered_rand']['ci_lo']:.4f}, "
          f"{out['recovered_rand']['ci_hi']:.4f}]")
    print(f"  difference                               "
          f"{out['recovered_minus_random']['mean']:+.4f} "
          f"[{out['recovered_minus_random']['ci_lo']:+.4f}, "
          f"{out['recovered_minus_random']['ci_hi']:+.4f}]")
    print(f"  decode restored at repaired squares      "
          f"{out['repair_decode_ok']['mean']:.4f}")
    print(f"\n  NATURAL failures repaired to legal       0.1143 (partial), "
          f"0.1271 (whole board)")


if __name__ == "__main__":
    main()
