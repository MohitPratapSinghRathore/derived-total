"""Test-time state repair, using no oracle.

Paper 1 showed that editing the state representation changes the action. That
makes repair conceivable: if the state at depth has drifted, and we can write to
it, we might be able to write a better one.

The difficulty is knowing what to write. An oracle would tell us, and an oracle
is exactly what a deployed system does not have.

This uses the model against itself. State fidelity is high early and decays with
depth, so an early representation is more trustworthy than a late one. The
learned transition operator can carry a state forward one step at a time using
only the move tokens, which are in the context. So we can take the model's own
belief from a shallower position, roll it forward under the moves that actually
occurred, and inject the result at the current position.

    z_repaired(t) = T( ... T(T(z(t-k), x_{t-k+1}), x_{t-k+2}) ..., x_t )
    h(t) <- h(t) + eta * W (z_repaired(t) - z(t))

Nothing here reads the true board. The inputs are the model's own earlier
representation and the tokens it has already been given.

Three controls decide whether any improvement is real:
  no-op          inject nothing, to establish the base rate
  random         inject a norm-matched random direction
  stale          inject the shallow state WITHOUT rolling it forward, which
                 tests whether any benefit comes from the transition operator
                 or merely from overwriting a decayed vector with an older one
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


def load_scr(name, width, vocab, dim):
    ck = torch.load(os.path.join(RUNS, f"{name}.pt"), map_location="cuda",
                    weights_only=False)
    if ck.get("scr") is None:
        return None
    m = StateConsistency(width, d_state=dim, vocab=vocab).cuda()
    m.load_state_dict(ck["scr"])
    m.eval()
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--games", type=int, default=800)
    ap.add_argument("--lookback", type=int, default=12)
    ap.add_argument("--eta", type=float, default=0.5)
    args = ap.parse_args()

    model, a, ck = build(os.path.join(RUNS, f"{args.name}.pt"))
    stoi = json.load(open(os.path.join(DATA, "vocab.json")))
    itos = {v: k for k, v in stoi.items()}
    scr = load_scr(args.name, a["width"], len(stoi), a.get("scr_dim", 64))
    if scr is None:
        print("this checkpoint has no state-consistency module")
        return
    layer = a.get("scr_layer", -1)

    # a write direction: least-squares pseudo-inverse of the projection, so a
    # change in the state subspace maps back into the residual stream
    P = scr.proj.weight.detach()                       # (k,d)
    Wback = torch.linalg.pinv(P)                       # (d,k)

    toks, lens, occ = load_split("eval", args.games)
    nb = len(BUCKETS)
    modes = ["noop", "repair", "random", "stale"]
    ill = {m: np.zeros(nb) for m in modes}
    tot = np.zeros(nb)
    rng = np.random.default_rng(0)

    with torch.no_grad():
        for gi in range(len(toks)):
            T = int(lens[gi])
            if T < 30:
                continue
            board = chess.Board()
            seq = [int(v) for v in toks[gi, :T]]
            for t in range(T - 1):
                bi = next((k for k, (lo, hi) in enumerate(BUCKETS)
                           if lo <= t < hi), None)
                if bi is not None and t > args.lookback + 2:
                    x = torch.tensor(seq[:t + 1], dtype=torch.long).cuda()[None]
                    _, hs, _ = model(x, return_hidden=True)
                    h = hs[layer].float()

                    # the model's own belief a few steps back, rolled forward
                    t0 = t - args.lookback
                    z = F.normalize(scr.proj(h[0, t0]), dim=-1)
                    for u in range(t0 + 1, t + 1):
                        mv = scr.move(torch.tensor([seq[u]]).cuda())[0]
                        z = F.normalize(scr.trans(torch.cat([z, mv])), dim=-1)
                    z_now = F.normalize(scr.proj(h[0, t]), dim=-1)
                    z_stale = F.normalize(scr.proj(h[0, t0]), dim=-1)

                    deltas = {
                        "noop": None,
                        "repair": Wback @ (z - z_now),
                        "stale": Wback @ (z_stale - z_now),
                    }
                    r = torch.randn(a["width"], device="cuda")
                    ref = deltas["repair"]
                    deltas["random"] = r / r.norm() * ref.norm()

                    for mname, delta in deltas.items():
                        if delta is None:
                            logits, _, _ = model(x)
                        else:
                            def hook(mod, inp, out, d=delta):
                                o = out.clone()
                                o[:, -1] = o[:, -1] + args.eta * d
                                return o
                            hd = model.blocks[layer].register_forward_hook(hook)
                            logits, _, _ = model(x)
                            hd.remove()
                        mv = itos.get(int(logits.float()[0, -1].argmax()), "")
                        try:
                            legal = chess.Move.from_uci(mv) in board.legal_moves
                        except Exception:
                            legal = False
                        ill[mname][bi] += (not legal)
                    tot[bi] += 1
                board.push_uci(itos[seq[t + 1]])
            if tot.sum() > 4000:
                break

    out = {"name": args.name, "lookback": args.lookback, "eta": args.eta,
           "buckets": BUCKETS, "n": tot.tolist(),
           "illegal": {m: (ill[m] / np.maximum(tot, 1)).tolist() for m in modes}}
    json.dump(out, open(os.path.join(RES, f"{args.name}_refresh.json"), "w"),
              indent=1)
    print(f"[{args.name}] test-time repair, lookback {args.lookback}, eta {args.eta}")
    print(f"{'ply':>9} {'no-op':>8} {'repair':>8} {'random':>8} {'stale':>8} {'n':>6}")
    for i, (lo, hi) in enumerate(BUCKETS):
        if tot[i]:
            print(f"{str(lo)+'-'+str(hi):>9} "
                  + " ".join(f"{out['illegal'][m][i]:8.4f}" for m in modes)
                  + f" {int(tot[i]):6d}")


if __name__ == "__main__":
    main()
