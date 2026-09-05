"""If the state is present but entangled, can the model itself use it?

The entanglement measurement asks what a decoder can recover. This asks the
question that matters: whether the model's own behaviour improves when the
read-out is given the capacity to disentangle. The model's output head is a
linear map on the final residual stream, so if board state becomes nonlinearly
encoded with depth, that head faces the same limitation our linear probe does.

The backbone stays frozen throughout. We retrain only the read-out, so any
change is attributable to the read-out and not to the representation.

The control carries the argument. A retrained linear head of the same budget,
on the same activations and the same data, absorbs the credit for retraining as
such. Only the difference between the nonlinear head and that retrained linear
head is evidence about entanglement, and it is a prediction with a shape: the
gap should widen with depth, because that is where the entanglement is.
"""
import os, json, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import chess
from probes import load_split, build

DATA = os.path.join(os.path.dirname(__file__), "..", "data", "proc")
RUNS = os.path.join(os.path.dirname(__file__), "..", "runs")
RES = os.path.join(os.path.dirname(__file__), "..", "results")
BUCKETS = [(0, 10), (10, 20), (20, 30), (30, 40), (40, 50),
           (50, 60), (60, 80), (80, 120)]


def head_of(kind, d, V, hidden):
    if kind == "linear":
        return nn.Linear(d, V).cuda()
    return nn.Sequential(nn.Linear(d, hidden), nn.GELU(),
                         nn.Linear(hidden, V)).cuda()


def train(kind, model, toks, lens, d, V, hidden, epochs, bs=48, seed=0):
    head = head_of(kind, d, V, hidden)
    opt = torch.optim.AdamW(head.parameters(), lr=1e-3)
    rng = np.random.default_rng(seed)
    N, L = toks.shape
    for _ in range(epochs):
        order = rng.permutation(N)
        for i in range(0, N, bs):
            idx = np.sort(order[i:i + bs])
            x = torch.from_numpy(toks[idx]).cuda()
            ln = torch.from_numpy(lens[idx]).cuda()
            with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.float16):
                _, hs, _ = model(x, return_hidden=True)
            h = hs[-1].float()[:, :-1]
            tgt = x[:, 1:]
            v = torch.arange(L - 1, device="cuda")[None] < (ln[:, None] - 1)
            loss = F.cross_entropy(head(h)[v], tgt[v])
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    head.eval()
    return head


@torch.no_grad()
def measure(heads, model, toks, lens, itos, games):
    nb = len(BUCKETS)
    bidx = np.full(200, -1)
    for bi, (lo, hi) in enumerate(BUCKETS):
        bidx[lo:hi] = bi
    allk = ["model"] + list(heads)
    ill = {k: np.zeros(nb) for k in allk}
    nll = {k: np.zeros(nb) for k in allk}
    tot = np.zeros(nb)
    per_game = {k: {} for k in allk}
    for gi in range(min(games, len(toks))):
        T = int(lens[gi])
        if T < 12:
            continue
        seq = torch.from_numpy(toks[gi:gi + 1, :T].astype(np.int64)).cuda()
        with torch.amp.autocast("cuda", dtype=torch.float16):
            logits, hs, _ = model(seq, return_hidden=True)
        h = hs[-1].float()[0]
        out = {"model": logits.float()[0]}
        for k, hd in heads.items():
            out[k] = hd(h).float()
        board = chess.Board()
        for t in range(T - 1):
            bi = bidx[t]
            tgt = int(seq[0, t + 1])
            if bi >= 0:
                tot[bi] += 1
                for k in out:
                    lg = out[k][t]
                    mv = itos.get(int(lg.argmax()), "")
                    try:
                        ok = chess.Move.from_uci(mv) in board.legal_moves
                    except Exception:
                        ok = False
                    ill[k][bi] += 0.0 if ok else 1.0
                    nll[k][bi] += float(F.cross_entropy(lg[None],
                                        torch.tensor([tgt], device="cuda")))
                    per_game[k].setdefault(gi, []).append(0.0 if ok else 1.0)
            board.push_uci(itos[tgt])
    return ill, nll, tot, per_game


def boot(per_game, keys, n_boot=400, seed=0):
    rng = np.random.default_rng(seed)
    gids = sorted(per_game[keys[0]])
    out = {k: [] for k in keys}
    for _ in range(n_boot):
        pick = rng.choice(gids, len(gids), replace=True)
        for k in keys:
            v = np.concatenate([per_game[k][g] for g in pick])
            out[k].append(v.mean())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="attn_8L256_s0")
    ap.add_argument("--fit_games", type=int, default=6000)
    ap.add_argument("--eval_games", type=int, default=1500)
    ap.add_argument("--hidden", type=int, default=1024)
    ap.add_argument("--epochs", type=int, default=3)
    args = ap.parse_args()

    model, a, ck = build(os.path.join(RUNS, f"{args.name}.pt"))
    stoi = json.load(open(os.path.join(DATA, "vocab.json")))
    itos = {v: k for k, v in stoi.items()}
    V, d = len(stoi), a["width"]

    ftk, flen, _ = load_split("train_probe", args.fit_games)
    etk, elen, _ = load_split("eval", args.eval_games)

    heads = {}
    for kind in ("linear", "mlp"):
        print(f"  training {kind} read-out on frozen activations", flush=True)
        heads[f"retrained_{kind}"] = train(kind, model, ftk, flen, d, V,
                                           args.hidden, args.epochs)

    ill, nll, tot, per_game = measure(heads, model, etk, elen, itos,
                                      args.eval_games)
    keys = list(ill)
    bs = boot(per_game, keys)
    diff = np.array(bs["retrained_mlp"]) - np.array(bs["retrained_linear"])

    out = {"name": args.name, "buckets": BUCKETS, "n": tot.tolist(),
           "illegal": {k: (ill[k] / np.maximum(tot, 1)).tolist() for k in keys},
           "nll": {k: (nll[k] / np.maximum(tot, 1)).tolist() for k in keys},
           "overall": {k: float(np.sum(ill[k]) / max(tot.sum(), 1)) for k in keys},
           "mlp_minus_linear": {"mean": float(diff.mean()),
                                "ci_lo": float(np.percentile(diff, 2.5)),
                                "ci_hi": float(np.percentile(diff, 97.5))}}
    json.dump(out, open(os.path.join(RES, f"{args.name}_nlhead.json"), "w"),
              indent=1)

    print(f"\n[{args.name}] illegal top-1 rate by depth, frozen backbone")
    print(f"{'ply':>9} " + " ".join(f"{k:>18}" for k in keys))
    for i, (lo, hi) in enumerate(BUCKETS):
        if tot[i] < 50:
            continue
        print(f"{str(lo)+'-'+str(hi):>9} "
              + " ".join(f"{ill[k][i]/tot[i]:18.4f}" for k in keys))
    print("\noverall: " + "  ".join(f"{k} {out['overall'][k]:.4f}" for k in keys))
    m = out["mlp_minus_linear"]
    print(f"nonlinear minus retrained-linear: {m['mean']:+.4f} "
          f"[{m['ci_lo']:+.4f}, {m['ci_hi']:+.4f}] (game-clustered)")


if __name__ == "__main__":
    main()
