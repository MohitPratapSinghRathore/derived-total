"""Test-time state repair on a frozen model, using no oracle.

Paper 1 showed the state representation is causally read by the policy, so
writing a better state should change behaviour. The obstacle is knowing what to
write without an oracle.

This uses the model against itself. Fidelity is high early and decays with
depth, so an earlier representation is more trustworthy than the current one.
The transition operator, fitted on frozen activations by the consistency
objective alone, can carry a state forward one move at a time using only tokens
already in the context. So we take the model's own belief from k steps back,
roll it forward under the moves that actually occurred, and inject the result.

    z_repaired = T( ... T(z(t-k), x_{t-k+1}) ..., x_t )
    h(t) <- h(t) + eta * W (z_repaired - z(t))

The oracle is never read. Two outcomes are measured, because Paper 1 showed they
can come apart: whether the decoded board gets closer to the truth, and whether
the emitted move is more often legal.

Controls, all at the same magnitude:
  no-op    inject nothing, giving the base rate
  random   a norm-matched random direction, so any effect of merely perturbing
           the stream is visible
  stale    inject the older state WITHOUT rolling it forward. This is the
           control that matters: if it does as well as repair, the transition
           operator contributes nothing and the method reduces to overwriting a
           decayed vector with an older one.
"""
import os, json, argparse
import numpy as np
import torch
import torch.nn.functional as F
import chess
from probes import load_split, build
from scr import StateConsistency

DATA = os.path.join(os.path.dirname(__file__), "..", "data", "proc")
RUNS = os.path.join(os.path.dirname(__file__), "..", "runs")
RES = os.path.join(os.path.dirname(__file__), "..", "results")
BUCKETS = [(20, 30), (30, 40), (40, 50), (50, 60), (60, 80)]
MODES = ["noop", "repair", "stale", "random"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="attn_8L256_s0")
    ap.add_argument("--games", type=int, default=600)
    ap.add_argument("--lookback", type=int, default=12)
    ap.add_argument("--eta", type=float, default=1.0)
    args = ap.parse_args()

    model, a, ck = build(os.path.join(RUNS, f"{args.name}.pt"))
    stoi = json.load(open(os.path.join(DATA, "vocab.json")))
    itos = {v: k for k, v in stoi.items()}

    tr = torch.load(os.path.join(RUNS, f"{args.name}_trans.pt"),
                    map_location="cuda", weights_only=False)
    dim = tr["args"]["dim"]; layer = tr["args"]["layer"]
    scr = StateConsistency(a["width"], d_state=dim, vocab=len(stoi)).cuda()
    scr.load_state_dict(tr["scr"]); scr.eval()

    # the paper-1 probe, used only to report whether the repaired state is
    # closer to the truth; it plays no part in choosing what to inject
    res = json.load(open(os.path.join(RES, f"{args.name}.json")))
    pls = torch.load(os.path.join(RES, f"{args.name}_probes.pt"), weights_only=False)
    probe = torch.nn.Linear(a["width"], 64 * 13).cuda()
    probe.load_state_dict(pls[res["best_layer"]]); probe.eval()
    plyr = res["best_layer"]

    Wback = torch.linalg.pinv(scr.proj.weight.detach())      # (d,k)
    toks, lens, occ = load_split("eval", args.games)
    nb = len(BUCKETS)
    ill = {m: np.zeros(nb) for m in MODES}
    occacc = {m: np.zeros(nb) for m in MODES}
    tot = np.zeros(nb); occn = np.zeros(nb)

    def bucket(t):
        return next((k for k, (lo, hi) in enumerate(BUCKETS) if lo <= t < hi), None)

    with torch.no_grad():
        for gi in range(len(toks)):
            T = int(lens[gi])
            if T < 30:
                continue
            seq = torch.tensor([int(v) for v in toks[gi, :T]], dtype=torch.long).cuda()
            pos = [t for t in range(T - 1)
                   if bucket(t) is not None and t > args.lookback + 1]
            if not pos:
                continue
            P = len(pos)

            # one causal forward gives every h[t] and every clean logit
            _, hs, _ = model(seq[None], return_hidden=True)
            h_all = hs[layer].float()[0]                     # (T,d)
            hp_all = hs[plyr].float()[0]

            pos_t = torch.tensor(pos, device="cuda")
            t0 = pos_t - args.lookback
            z_now = F.normalize(scr.proj(h_all[pos_t]), dim=-1)
            z_old = F.normalize(scr.proj(h_all[t0]), dim=-1)

            # roll every position's old state forward in parallel
            z = z_old.clone()
            for step in range(1, args.lookback + 1):
                mv = scr.move(seq[t0 + step])
                z = F.normalize(scr.trans(torch.cat([z, mv], dim=-1)), dim=-1)

            deltas = {
                "noop": torch.zeros(P, a["width"], device="cuda"),
                "repair": (z - z_now) @ Wback.t(),
                "stale": (z_old - z_now) @ Wback.t(),
            }
            r = torch.randn(P, a["width"], device="cuda")
            r = r / r.norm(dim=-1, keepdim=True) * deltas["repair"].norm(
                dim=-1, keepdim=True)
            deltas["random"] = r

            X = seq[None].expand(P, -1)                      # (P,T)
            board_states = np.asarray(occ[gi]).astype(np.int64)

            for mname, delta in deltas.items():
                if mname == "noop":
                    logits, hs2, _ = model(X[:1], return_hidden=True)
                    lg = logits.float()[0, pos_t]
                    hp = hs2[plyr].float()[0, pos_t]
                else:
                    def hook(mod, inp, out, d=delta, p=pos_t):
                        o = out.clone()
                        o[torch.arange(len(p), device="cuda"), p] += args.eta * d
                        return o
                    hd = model.blocks[layer].register_forward_hook(hook)
                    logits, hs2, _ = model(X, return_hidden=True)
                    hd.remove()
                    lg = logits.float()[torch.arange(P, device="cuda"), pos_t]
                    hp = hs2[plyr].float()[torch.arange(P, device="cuda"), pos_t]

                pred = lg.argmax(-1).cpu().numpy()
                dec = probe(hp).reshape(len(pos), 64, 13).argmax(-1).cpu().numpy()

                board = chess.Board()
                ti = 0
                for t in range(T - 1):
                    if ti < len(pos) and pos[ti] == t:
                        b = bucket(t)
                        mv = itos.get(int(pred[ti]), "")
                        try:
                            legal = chess.Move.from_uci(mv) in board.legal_moves
                        except Exception:
                            legal = False
                        ill[mname][b] += (not legal)
                        truth = board_states[t]
                        m = truth > 0
                        if m.sum():
                            occacc[mname][b] += float((dec[ti][m] == truth[m]).mean())
                            if mname == "noop":
                                occn[b] += 1
                        if mname == "noop":
                            tot[b] += 1
                        ti += 1
                    board.push_uci(itos[int(seq[t + 1])])
            if tot.sum() > 6000:
                break

    out = {"name": args.name, "lookback": args.lookback, "eta": args.eta,
           "buckets": BUCKETS, "n": tot.tolist(),
           "illegal": {m: (ill[m] / np.maximum(tot, 1)).tolist() for m in MODES},
           "occ_acc": {m: (occacc[m] / np.maximum(occn, 1)).tolist() for m in MODES}}
    json.dump(out, open(os.path.join(RES, f"{args.name}_refresh.json"), "w"),
              indent=1)

    print(f"[{args.name}] test-time repair, lookback {args.lookback}, eta {args.eta}, "
          f"n={int(tot.sum())}")
    print("\nilleg al-move rate")
    print(f"{'ply':>9} " + " ".join(f"{m:>9}" for m in MODES))
    for i, (lo, hi) in enumerate(BUCKETS):
        if tot[i]:
            print(f"{str(lo)+'-'+str(hi):>9} "
                  + " ".join(f"{out['illegal'][m][i]:9.4f}" for m in MODES))
    print("\ndecoded board accuracy on occupied squares")
    print(f"{'ply':>9} " + " ".join(f"{m:>9}" for m in MODES))
    for i, (lo, hi) in enumerate(BUCKETS):
        if occn[i]:
            print(f"{str(lo)+'-'+str(hi):>9} "
                  + " ".join(f"{out['occ_acc'][m][i]:9.4f}" for m in MODES))


if __name__ == "__main__":
    main()
