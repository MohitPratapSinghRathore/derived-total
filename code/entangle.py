"""Is the state lost with depth, or only harder to read out linearly?

Every fidelity curve in this line of work, ours included, is measured with a
linear probe. That conflates two very different things: information leaving the
representation, and information remaining but becoming nonlinearly encoded.

The scale ladder hints that the second matters. Adding fifty percent more width
buys seven times the state fidelity that adding fifty percent more layers does.
If the bottleneck were serial computation depth, layers would dominate. Width
dominating instead points at representational room, which is what a superposition
or interference account predicts.

This measures the gap directly. On identical activations, splits and budgets we
fit a linear read-out and a nonlinear one, and compare their board accuracy
against depth. Two controls bound the obvious objection that a bigger decoder
simply fits more:

  random weights   the same two decoders on an untrained model of identical
                   shape, which bounds what decoder capacity alone achieves
  control task     the same two decoders against labels paired with another
                   game's board at the same depth, which bounds memorisation

If the nonlinear gap widens with depth while the controls stay flat, then part of
what the field reports as state decay is decoder weakness rather than information
loss, and the state is present but entangled.
"""
import os, json, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from probes import load_split, build

DATA = os.path.join(os.path.dirname(__file__), "..", "data", "proc")
RUNS = os.path.join(os.path.dirname(__file__), "..", "runs")
RES = os.path.join(os.path.dirname(__file__), "..", "results")
BUCKETS = [(0, 10), (10, 20), (20, 30), (30, 40), (40, 50),
           (50, 60), (60, 80), (80, 120)]
NCLS = 13


def make_head(kind, d, hidden):
    if kind == "linear":
        return nn.Linear(d, 64 * NCLS).cuda()
    return nn.Sequential(nn.Linear(d, hidden), nn.GELU(),
                         nn.Linear(hidden, 64 * NCLS)).cuda()


def fit(kind, model, layer, toks, lens, occ, d, hidden, epochs, bs=48,
        shuffle_labels=False, seed=0):
    head = make_head(kind, d, hidden)
    opt = torch.optim.AdamW(head.parameters(), lr=1e-3)
    rng = np.random.default_rng(seed)
    N, L = toks.shape
    for ep in range(epochs):
        order = rng.permutation(N)
        for i in range(0, N, bs):
            idx = np.sort(order[i:i + bs])
            x = torch.from_numpy(toks[idx]).cuda()
            s_idx = np.sort(rng.permutation(N)[:len(idx)]) if shuffle_labels else idx
            s = torch.from_numpy(np.asarray(occ[s_idx]).astype(np.int64)).cuda()
            v = torch.from_numpy(np.arange(L)[None] < lens[idx][:, None]).cuda()
            with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.float16):
                _, hs, _ = model(x, return_hidden=True)
            h = hs[layer].float()
            lg = head(h).reshape(len(idx), L, 64, NCLS)
            loss = F.cross_entropy(lg[v].reshape(-1, NCLS), s[v].reshape(-1))
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    return head


@torch.no_grad()
def evaluate(head, model, layer, toks, lens, occ, bs=48, shuffle_labels=False,
             seed=1):
    nb = len(BUCKETS)
    hit = np.zeros(nb); tot = np.zeros(nb)
    N, L = toks.shape
    bidx = np.full(L, -1)
    for bi, (lo, hi) in enumerate(BUCKETS):
        bidx[lo:hi] = bi
    rng = np.random.default_rng(seed)
    for i in range(0, N, bs):
        idx = np.arange(i, min(i + bs, N))
        x = torch.from_numpy(toks[idx]).cuda()
        s_idx = np.sort(rng.permutation(N)[:len(idx)]) if shuffle_labels else idx
        s = np.asarray(occ[s_idx]).astype(np.int64)
        with torch.amp.autocast("cuda", dtype=torch.float16):
            _, hs, _ = model(x, return_hidden=True)
        pred = head(hs[layer].float()).reshape(len(idx), L, 64, NCLS
                                               ).argmax(-1).cpu().numpy()
        for r, gi in enumerate(idx):
            T = int(lens[gi])
            for t in range(T):
                bi = bidx[t]
                if bi < 0:
                    continue
                m = s[r, t] > 0
                if m.sum():
                    hit[bi] += float((pred[r, t][m] == s[r, t][m]).mean())
                    tot[bi] += 1
    return hit / np.maximum(tot, 1), tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="attn_8L256_s0")
    ap.add_argument("--fit_games", type=int, default=3000)
    ap.add_argument("--eval_games", type=int, default=2000)
    ap.add_argument("--hidden", type=int, default=1024)
    ap.add_argument("--epochs", type=int, default=3)
    args = ap.parse_args()

    model, a, ck = build(os.path.join(RUNS, f"{args.name}.pt"))
    res = json.load(open(os.path.join(RES, f"{args.name}.json")))
    layer = res["best_layer"]
    d = a["width"]

    ftk, flen, focc = load_split("train_probe", args.fit_games)
    etk, elen, eocc = load_split("eval", args.eval_games)

    out = {"name": args.name, "layer": layer, "hidden": args.hidden,
           "buckets": BUCKETS, "curves": {}}

    rand_model, _, _ = build(os.path.join(RUNS, f"{args.name}.pt"), randomised=True)

    jobs = [
        ("linear", model, False, "linear"),
        ("mlp", model, False, "mlp"),
        ("linear_randomweights", rand_model, False, "linear"),
        ("mlp_randomweights", rand_model, False, "mlp"),
        ("linear_controltask", model, True, "linear"),
        ("mlp_controltask", model, True, "mlp"),
    ]
    for label, mdl, shuf, kind in jobs:
        print(f"  fitting {label}", flush=True)
        head = fit(kind, mdl, layer, ftk, flen, focc, d, args.hidden,
                   args.epochs, shuffle_labels=shuf)
        acc, n = evaluate(head, mdl, layer, etk, elen, eocc, shuffle_labels=shuf)
        out["curves"][label] = acc.tolist()
        out["n"] = n.tolist()

    json.dump(out, open(os.path.join(RES, f"{args.name}_entangle.json"), "w"),
              indent=1)

    print(f"\n[{args.name}] occupied-square accuracy, linear against nonlinear")
    keys = ["linear", "mlp", "linear_randomweights", "mlp_randomweights",
            "linear_controltask", "mlp_controltask"]
    print(f"{'ply':>9} " + " ".join(f"{k.replace('_',' ')[:11]:>12}" for k in keys)
          + f" {'gap':>7}")
    for i, (lo, hi) in enumerate(BUCKETS):
        if out["n"][i] < 50:
            continue
        row = " ".join(f"{out['curves'][k][i]:12.4f}" for k in keys)
        gap = out["curves"]["mlp"][i] - out["curves"]["linear"][i]
        print(f"{str(lo)+'-'+str(hi):>9} {row} {gap:+7.4f}")


if __name__ == "__main__":
    main()
