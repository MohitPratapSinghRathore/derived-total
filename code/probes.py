"""Stage 4: linear state probes -> F^(l)(t), H_eps, and E_state.

Probes are fit on train_probe (game-disjoint from eval) and all numbers are
reported on eval.  Controls: (a) Hewitt-Liang control task with shuffled square
labels, (b) randomised-weight model of identical architecture.
"""
import os, json, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from model import ChessLM

DATA = os.path.join(os.path.dirname(__file__), "..", "data", "proc")
RUNS = os.path.join(os.path.dirname(__file__), "..", "runs")
NCLS = 13


def load_split(split, n=None):
    d = np.load(os.path.join(DATA, f"{split}.npz"))
    toks, lens = d["toks"].astype(np.int64), d["lens"].astype(np.int64)
    occ = np.load(os.path.join(DATA, f"{split}_occ.npy"), mmap_mode="r")
    if n:
        toks, lens, occ = toks[:n], lens[:n], occ[:n]
    return toks, lens, occ


def build(ckpt, randomised=False, device="cuda"):
    c = torch.load(ckpt, map_location=device, weights_only=False)
    a = c["args"]
    V = len(json.load(open(os.path.join(DATA, "vocab.json"))))
    m = ChessLM(V, d=a["width"], n_layer=a["layers"], n_head=a["heads"],
                max_len=161, alsb=a["alsb"], d_s=a["d_s"], k=a["k"],
                ffn_mult=a["ffn_mult"]).to(device)
    if not randomised:
        m.load_state_dict(c["model"])
    m.eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m, a, c


@torch.no_grad()
def hiddens(model, x):
    with torch.amp.autocast("cuda", dtype=torch.float16):
        logits, hs, _ = model(x, return_hidden=True)
    return logits.float(), [h.float() for h in hs]


def fit_probes(model, a, toks, lens, occ, epochs=3, bs=48, control=False, seed=0):
    """One linear head per layer, all trained in the same forward pass."""
    dev = "cuda"
    nl = a["layers"]
    probes = [nn.Linear(a["width"], 64 * NCLS).to(dev) for _ in range(nl)]
    opt = torch.optim.AdamW([p for pr in probes for p in pr.parameters()], lr=1e-3)
    N, L = toks.shape
    rng = np.random.default_rng(seed)
    # control task: fixed random relabelling of the 13 classes, per square
    if control:
        perm = np.stack([rng.permutation(NCLS) for _ in range(64)])  # (64,13)
    order = np.arange(N)
    for ep in range(epochs):
        rng.shuffle(order)
        tot = n = 0
        for i in range(0, N, bs):
            idx = np.sort(order[i:i + bs])
            x = torch.from_numpy(toks[idx]).to(dev)
            s = np.asarray(occ[idx]).astype(np.int64)
            if control:
                s = perm[np.arange(64)[None, None, :], s]
            s = torch.from_numpy(s).to(dev)
            valid = torch.from_numpy(np.arange(L)[None] < lens[idx][:, None]).to(dev)
            _, hs = hiddens(model, x)
            loss = 0
            for li in range(nl):
                lg = probes[li](hs[li]).reshape(-1, L, 64, NCLS)
                loss = loss + F.cross_entropy(lg[valid].reshape(-1, NCLS),
                                              s[valid].reshape(-1))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot += float(loss); n += 1
        print(f"    probe epoch {ep} loss/layer {tot/n/nl:.4f}", flush=True)
    return probes


@torch.no_grad()
def evaluate(model, a, probes, toks, lens, occ, bs=48, buckets=None):
    """F^(l)(t) exact-position accuracy + per-square acc + teacher-forced illegal rate."""
    dev = "cuda"
    nl = a["layers"]
    N, L = toks.shape
    if buckets is None:
        buckets = [(0, 10), (10, 20), (20, 30), (30, 40), (40, 50),
                   (50, 60), (60, 80), (80, 120), (120, 161)]
    nb = len(buckets)
    exact = np.zeros((nl, nb)); sqacc = np.zeros((nl, nb)); cnt = np.zeros(nb)
    bidx = np.full(L, -1)
    for bi, (lo, hi) in enumerate(buckets):
        bidx[lo:hi] = bi
    bidx_t = torch.from_numpy(bidx).to(dev)

    for i in range(0, N, bs):
        idx = np.arange(i, min(i + bs, N))
        x = torch.from_numpy(toks[idx]).to(dev)
        s = torch.from_numpy(np.asarray(occ[idx]).astype(np.int64)).to(dev)
        valid = torch.from_numpy(np.arange(L)[None] < lens[idx][:, None]).to(dev)
        _, hs = hiddens(model, x)
        bb = bidx_t[None].expand(len(idx), L)
        # apply each probe once per batch, not once per (layer, bucket)
        corrs = [(probes[li](hs[li]).reshape(len(idx), L, 64, NCLS).argmax(-1) == s)
                 for li in range(nl)]
        for bi in range(nb):
            m = valid & (bb == bi)
            nm = int(m.sum())
            if nm == 0:
                continue
            cnt[bi] += nm
            for li in range(nl):
                corr = corrs[li]
                sqacc[li, bi] += float(corr[m].float().mean()) * nm
                exact[li, bi] += float(corr.all(-1)[m].float().sum())
    return exact / np.maximum(cnt, 1), sqacc / np.maximum(cnt, 1), cnt, buckets
