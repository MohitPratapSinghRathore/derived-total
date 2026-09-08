"""Fit board-state probes on Chess-GPT at move-boundary characters.

Every load-bearing measurement in the paper needs a decoded board: belief
consistency needs one to ask whether an illegal move is legal on it, the
move-touched AUC needs per-square fidelity, and the calibrated edit needs the
probe's weight rows as its edit basis. None of them can run on an external model
until a probe exists, which is why this comes first.

Our own probes read one residual stream per move. Here the site is the last
character before a move begins, established and verified in chessgpt_data, so the
probe is fitted only at those positions and never at mid-move characters where
the board is undefined between updates.

Controls follow the main paper. The randomised-weight model bounds what probe
capacity alone achieves on this architecture, and the control task pairs each
activation with a different game's board at the same ply, which keeps the label
marginals while destroying the relation. A probe that succeeds under either is
reading itself rather than the model.
"""
import os, json, argparse, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from chessgpt import load, encode
from chessgpt_nano import convert as nano_convert
from chessgpt_data import build, load_games

ROOT = os.path.join(os.path.dirname(__file__), "..")
RES = os.path.join(ROOT, "results")
EXT = os.path.join(ROOT, "data", "ext")
NCLS = 13
BUCKETS = [(20, 30), (30, 40), (40, 50), (50, 60), (60, 80), (80, 120)]


def batch_sites(model, games, max_plies, minply, bs=8, device="cuda",
                randomise=False):
    """Yield (activations [n_layer+1, N, d], occ [N, 64], ply [N], game [N])."""
    for i in range(0, len(games), bs):
        chunk = games[i:i + bs]
        streams = [build(g, max_plies=max_plies) for g in chunk]
        ids = [encode(t)[:1023] for t, _ in streams]
        L = max(len(x) for x in ids)
        x = torch.zeros(len(ids), L, dtype=torch.long, device=device)
        for j, v in enumerate(ids):
            x[j, :len(v)] = torch.tensor(v, device=device)
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.float16):
            out = model(x, output_hidden_states=True)
        hs = [h.float() for h in out.hidden_states]
        A, O, P, G = [], [], [], []
        for j, (_, sites) in enumerate(streams):
            for s in sites:
                if s["ply"] < minply or s["char_index"] >= len(ids[j]):
                    continue
                A.append(s["char_index"] + j * 0)      # index within row j
                O.append(s["occ"])
                P.append(s["ply"])
                G.append(i + j)
        if not A:
            continue
        rows = []
        k = 0
        for j, (_, sites) in enumerate(streams):
            for s in sites:
                if s["ply"] < minply or s["char_index"] >= len(ids[j]):
                    continue
                rows.append((j, s["char_index"]))
                k += 1
        idx_j = torch.tensor([r[0] for r in rows], device=device)
        idx_t = torch.tensor([r[1] for r in rows], device=device)
        acts = torch.stack([h[idx_j, idx_t] for h in hs])     # [n_h, N, d]
        yield acts, np.array(O), np.array(P), np.array(G)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit_games", type=int, default=1200)
    ap.add_argument("--eval_games", type=int, default=600)
    ap.add_argument("--minply", type=int, default=20)
    ap.add_argument("--maxply", type=int, default=120)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--nano", default=None)
    ap.add_argument("--tag", default="chessgpt2")
    ap.add_argument("--label", default=None)
    args = ap.parse_args()

    if args.nano:
        model, cfg, _ = nano_convert(os.path.join(EXT, args.nano))
    else:
        model, cfg, _ = load(args.tag)
    label = args.label or (args.nano or args.tag)
    d = cfg.n_embd
    nh = cfg.n_layer + 1
    print(f"{label}: {cfg.n_layer}L d={d}", flush=True)

    fit_games = load_games("train_probe", args.fit_games)
    ev_games = load_games("eval", args.eval_games)

    probes = [nn.Linear(d, 64 * NCLS).cuda() for _ in range(nh)]
    opt = torch.optim.AdamW([p for pr in probes for p in pr.parameters()],
                            lr=1e-3)
    t0 = time.time()
    for ep in range(args.epochs):
        tot = n = 0
        for acts, occ, ply, gid in batch_sites(model, fit_games, args.maxply,
                                               args.minply):
            y = torch.from_numpy(occ).cuda()
            loss = 0
            for li in range(nh):
                lg = probes[li](acts[li]).reshape(-1, 64, NCLS)
                loss = loss + F.cross_entropy(lg.reshape(-1, NCLS),
                                              y.reshape(-1))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot += float(loss)
            n += 1
        print(f"  epoch {ep} loss/layer {tot/max(n,1)/nh:.4f} "
              f"({time.time()-t0:.0f}s)", flush=True)

    # evaluate: occupied-square accuracy by depth, per layer
    hit = np.zeros((nh, len(BUCKETS)))
    cnt = np.zeros(len(BUCKETS))
    with torch.no_grad():
        for acts, occ, ply, gid in batch_sites(model, ev_games, args.maxply,
                                               args.minply):
            pred = np.stack([probes[li](acts[li]).reshape(-1, 64, NCLS)
                             .argmax(-1).cpu().numpy() for li in range(nh)])
            for bi, (lo, hi) in enumerate(BUCKETS):
                m = (ply >= lo) & (ply < hi)
                if not m.any():
                    continue
                o = occ[m]
                om = o > 0
                cnt[bi] += m.sum()
                for li in range(nh):
                    p = pred[li][m]
                    hit[li, bi] += float(np.mean([(p[r][om[r]] == o[r][om[r]]).mean()
                                                  for r in range(len(o))])) * m.sum()
    acc = hit / np.maximum(cnt, 1)
    best = int(acc.mean(1).argmax())
    print(f"\noccupied-square accuracy by depth, best layer {best}")
    print(f"{'ply':>9} " + " ".join(f"{f'L{li}':>7}" for li in range(nh)))
    for bi, (lo, hi) in enumerate(BUCKETS):
        if cnt[bi] < 20:
            continue
        print(f"{f'{lo}-{hi}':>9} " + " ".join(f"{acc[li,bi]:7.4f}"
                                               for li in range(nh)))

    safe = label.replace(".pt", "")
    torch.save({"probes": [p.state_dict() for p in probes], "best_layer": best,
                "d": d, "n_hidden": nh},
               os.path.join(RES, f"{safe}_probes.pt"))
    json.dump({"model": label, "best_layer": best,
               "buckets": BUCKETS, "acc_by_layer": acc.tolist(),
               "n_eval_sites": int(cnt.sum())},
              open(os.path.join(RES, f"{safe}_probe.json"), "w"), indent=1)
    print(f"\nsaved probes for {label}, best layer {best}, "
          f"{int(cnt.sum()):,} eval sites")


if __name__ == "__main__":
    main()
