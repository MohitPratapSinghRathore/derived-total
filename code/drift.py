"""Is the state being lost, or is the representation rotating with depth?

Every fidelity number in this paper comes from one linear decoder fitted across
all depths. Falling accuracy at depth is read as the state becoming less
available. There is an alternative that a single global decoder cannot rule out:
the state may remain equally present while its linear basis moves with depth, so
that one fixed decoder simply transfers badly.

The reversible-cycle experiment controls position difficulty but still uses the
global probe, so it inherits the same ambiguity.

This separates the two. We fit a separate linear probe inside each depth bucket
and evaluate every probe on every bucket, giving a train-depth by test-depth
transfer matrix.

  If depth-local probes also collapse at depth, the information is genuinely
  less linearly available.
  If depth-local probes stay strong while cross-depth transfer decays, the
  representation is drifting rather than disappearing.

The diagonal is within-depth accuracy, the global row is the probe used
elsewhere, and the gap between them is what the paper is allowed to call loss.
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
BUCKETS = [(10, 20), (20, 30), (30, 40), (40, 50), (50, 60), (60, 80)]
NCLS = 13


def collect(model, layer, toks, lens, occ, bs=32, cap=40000):
    """Activations and oracle occupancy, grouped by depth bucket."""
    H = {b: [] for b in range(len(BUCKETS))}
    S = {b: [] for b in range(len(BUCKETS))}
    with torch.no_grad():
        for i in range(0, len(toks), bs):
            idx = np.arange(i, min(i + bs, len(toks)))
            x = torch.from_numpy(toks[idx]).cuda()
            with torch.amp.autocast("cuda", dtype=torch.float16):
                _, hs, _ = model(x, return_hidden=True)
            h = hs[layer].float().cpu()
            o = np.asarray(occ[idx]).astype(np.int64)
            for r, gi in enumerate(idx):
                T = int(lens[gi])
                for bi, (lo, hi) in enumerate(BUCKETS):
                    if len(H[bi]) >= cap:
                        continue
                    for t in range(lo, min(hi, T - 1)):
                        H[bi].append(h[r, t].numpy())
                        S[bi].append(o[r, t])
            if all(len(v) >= cap for v in H.values()):
                break
    return ({b: np.stack(v) for b, v in H.items() if v},
            {b: np.stack(v) for b, v in S.items() if v})


def fit_probe(X, Y, d, epochs=6, bs=512, lr=1e-3):
    """One linear head, 64 squares by 13 classes, on the given activations."""
    pr = nn.Linear(d, 64 * NCLS).cuda()
    opt = torch.optim.AdamW(pr.parameters(), lr=lr)
    Xt = torch.from_numpy(X).cuda(); Yt = torch.from_numpy(Y).cuda()
    n = len(Xt)
    for _ in range(epochs):
        perm = torch.randperm(n, device="cuda")
        for i in range(0, n, bs):
            j = perm[i:i + bs]
            lg = pr(Xt[j]).reshape(-1, 64, NCLS)
            loss = F.cross_entropy(lg.reshape(-1, NCLS), Yt[j].reshape(-1))
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    return pr


@torch.no_grad()
def occ_acc(pr, X, Y, bs=4096):
    """Accuracy on occupied squares only, matching the paper's measure."""
    hit = tot = 0
    for i in range(0, len(X), bs):
        x = torch.from_numpy(X[i:i + bs]).cuda()
        y = torch.from_numpy(Y[i:i + bs]).cuda()
        p = pr(x).reshape(-1, 64, NCLS).argmax(-1)
        m = y > 0
        hit += float(((p == y) & m).sum()); tot += float(m.sum())
    return hit / max(tot, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="attn_8L256_s0")
    ap.add_argument("--games", type=int, default=2000)
    ap.add_argument("--cap", type=int, default=30000)
    args = ap.parse_args()

    model, a, c = build(os.path.join(RUNS, f"{args.name}.pt"))
    res = json.load(open(os.path.join(RES, f"{args.name}.json")))
    layer = res["best_layer"]
    d = a["width"]

    ptk, plen, pocc = load_split("train_probe", args.games)
    etk, elen, eocc = load_split("eval", args.games)
    print("collecting activations", flush=True)
    Htr, Str = collect(model, layer, ptk, plen, pocc, cap=args.cap)
    Hte, Ste = collect(model, layer, etk, elen, eocc, cap=args.cap // 2)

    # depth-local probes, each fitted on its own bucket of the probe split
    local = {}
    for b in Htr:
        print(f"  fitting probe for bucket {BUCKETS[b]}", flush=True)
        local[b] = fit_probe(Htr[b], Str[b], d)

    # a global probe on pooled depths, matching how the paper's probe is fitted
    Xall = np.concatenate([Htr[b] for b in sorted(Htr)])
    Yall = np.concatenate([Str[b] for b in sorted(Str)])
    print("  fitting pooled probe", flush=True)
    glob = fit_probe(Xall, Yall, d)

    M = np.full((len(BUCKETS), len(BUCKETS)), np.nan)
    for tb in sorted(local):
        for eb in sorted(Hte):
            M[tb, eb] = occ_acc(local[tb], Hte[eb], Ste[eb])
    gl = [occ_acc(glob, Hte[b], Ste[b]) if b in Hte else np.nan
          for b in range(len(BUCKETS))]

    out = {"name": args.name, "layer": layer, "buckets": BUCKETS,
           "transfer": M.tolist(), "global_probe": gl,
           "within_depth": [float(M[b, b]) for b in range(len(BUCKETS))]}
    json.dump(out, open(os.path.join(RES, f"{args.name}_drift.json"), "w"), indent=1)

    print(f"\n[{args.name}] occupied-square accuracy, train depth by test depth")
    hdr = "  train\\test " + " ".join(f"{lo:>7}" for lo, _ in BUCKETS)
    print(hdr)
    for tb, (lo, _) in enumerate(BUCKETS):
        row = " ".join(f"{M[tb, eb]:7.3f}" if np.isfinite(M[tb, eb]) else "    n/a"
                       for eb in range(len(BUCKETS)))
        print(f"  {lo:>10} {row}")
    print("  " + "-" * (12 + 8 * len(BUCKETS)))
    print(f"  {'pooled':>10} " + " ".join(f"{g:7.3f}" for g in gl))
    print(f"  {'diagonal':>10} " + " ".join(f"{M[b, b]:7.3f}"
                                            for b in range(len(BUCKETS))))


if __name__ == "__main__":
    main()
