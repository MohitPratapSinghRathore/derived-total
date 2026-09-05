"""Does rolling the state forward actually produce a better state?

Test-time repair had no effect. Two explanations are possible and they call for
different conclusions, so they must be separated before anything is written.

  (a) the injection is too weak to change the forward pass, in which case the
      method is untested rather than refuted
  (b) the rolled-forward state is no better than the current one, in which case
      the transition operator does not track the board and the method fails at
      its root

This decides between them. We fit a linear read-out from the consistency
subspace to board occupancy, then compare three states at the same position:

  z_now      the model's state at t
  z_rolled   its state from k steps back, carried forward by the operator
  z_stale    its state from k steps back, not carried forward

If z_rolled decodes the board better than z_now, the operator repairs and only
the write is failing. If it does not, the operator is not carrying board state.

The read-out here is a measuring instrument only. It is fitted on a held-out
split and is never used to choose what to inject, so the repair procedure itself
stays oracle-free.
"""
import os, json, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from probes import load_split, build
from scr import StateConsistency, MultiStepConsistency

DATA = os.path.join(os.path.dirname(__file__), "..", "data", "proc")
RUNS = os.path.join(os.path.dirname(__file__), "..", "runs")
RES = os.path.join(os.path.dirname(__file__), "..", "results")
BUCKETS = [(20, 30), (30, 40), (40, 50), (50, 60), (60, 80)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="attn_8L256_s0")
    ap.add_argument("--fit_games", type=int, default=1500)
    ap.add_argument("--eval_games", type=int, default=800)
    ap.add_argument("--lookback", type=int, default=12)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--trans", default="_trans",
                   help="suffix of the operator checkpoint to evaluate")
    args = ap.parse_args()

    model, a, ck = build(os.path.join(RUNS, f"{args.name}.pt"))
    stoi = json.load(open(os.path.join(DATA, "vocab.json")))
    tr = torch.load(os.path.join(RUNS, f"{args.name}{args.trans}.pt"),
                    map_location="cuda", weights_only=False)
    dim, layer = tr["args"]["dim"], tr["args"]["layer"]
    ms = tr["args"].get("multistep", 0)
    cls = MultiStepConsistency if ms > 0 else StateConsistency
    kw = {"max_k": ms} if ms > 0 else {}
    scr = cls(a["width"], d_state=dim, vocab=len(stoi), **kw).cuda()
    scr.load_state_dict(tr["scr"]); scr.eval()

    # ---- fit a read-out from the consistency subspace to occupancy
    ftk, flen, focc = load_split("train_probe", args.fit_games)
    head = nn.Linear(dim, 64 * 13).cuda()
    opt = torch.optim.AdamW(head.parameters(), lr=1e-3)
    for ep in range(args.epochs):
        for i in range(0, len(ftk), 48):
            idx = np.arange(i, min(i + 48, len(ftk)))
            x = torch.from_numpy(ftk[idx]).cuda()
            s = torch.from_numpy(np.asarray(focc[idx]).astype(np.int64)).cuda()
            v = torch.from_numpy(np.arange(ftk.shape[1])[None]
                                 < flen[idx][:, None]).cuda()
            with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.float16):
                _, hs, _ = model(x, return_hidden=True)
                z = scr.project(hs[layer].float())
            lg = head(z).reshape(len(idx), -1, 64, 13)
            loss = F.cross_entropy(lg[v].reshape(-1, 13), s[v].reshape(-1))
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        print(f"  read-out epoch {ep} loss {float(loss):.4f}", flush=True)

    # ---- compare the three states at matched positions
    etk, elen, eocc = load_split("eval", args.eval_games)
    nb = len(BUCKETS)
    acc = {k: np.zeros(nb) for k in ("now", "rolled", "stale")}
    dnorm = np.zeros(nb); hnorm = np.zeros(nb); n = np.zeros(nb)

    def bucket(t):
        return next((k for k, (lo, hi) in enumerate(BUCKETS) if lo <= t < hi), None)

    with torch.no_grad():
        for gi in range(len(etk)):
            T = int(elen[gi])
            if T < 30:
                continue
            seq = torch.tensor([int(v) for v in etk[gi, :T]], dtype=torch.long).cuda()
            pos = [t for t in range(T - 1)
                   if bucket(t) is not None and t > args.lookback + 1]
            if not pos:
                continue
            _, hs, _ = model(seq[None], return_hidden=True)
            h = hs[layer].float()[0]
            pt = torch.tensor(pos, device="cuda")
            t0 = pt - args.lookback
            z_now = scr.project(h[None])[0][pt]
            z_old = scr.project(h[None])[0][t0]
            z = z_old.clone()
            for step in range(1, args.lookback + 1):
                mv = scr.move(seq[t0 + step])
                z = F.normalize(scr.trans(torch.cat([z, mv], dim=-1)), dim=-1)

            truth = np.asarray(eocc[gi]).astype(np.int64)[pos]
            for key, zz in (("now", z_now), ("rolled", z), ("stale", z_old)):
                dec = head(zz).reshape(len(pos), 64, 13).argmax(-1).cpu().numpy()
                for j, t in enumerate(pos):
                    b = bucket(t)
                    m = truth[j] > 0
                    if m.sum():
                        acc[key][b] += float((dec[j][m] == truth[j][m]).mean())
            Wb = torch.linalg.pinv(scr.proj.weight.detach())
            d = (z - z_now) @ Wb.t()
            for j, t in enumerate(pos):
                b = bucket(t)
                dnorm[b] += float(d[j].norm())
                hnorm[b] += float(h[t].norm())
                n[b] += 1
            if n.sum() > 5000:
                break

    out = {"name": args.name, "lookback": args.lookback, "buckets": BUCKETS,
           "n": n.tolist(),
           "occ_acc": {k: (acc[k] / np.maximum(n, 1)).tolist() for k in acc},
           "delta_norm": (dnorm / np.maximum(n, 1)).tolist(),
           "hidden_norm": (hnorm / np.maximum(n, 1)).tolist(),
           "delta_over_hidden": (dnorm / np.maximum(hnorm, 1e-9)).tolist()}
    json.dump(out, open(os.path.join(RES, f"{args.name}{args.trans}_rollout.json"), "w"),
              indent=1)

    print(f"\n[{args.name}] board accuracy decoded from the consistency subspace")
    print(f"{'ply':>9} {'z_now':>9} {'z_rolled':>9} {'z_stale':>9} "
          f"{'|delta|/|h|':>12}")
    for i, (lo, hi) in enumerate(BUCKETS):
        if n[i]:
            print(f"{str(lo)+'-'+str(hi):>9} "
                  f"{out['occ_acc']['now'][i]:9.4f} "
                  f"{out['occ_acc']['rolled'][i]:9.4f} "
                  f"{out['occ_acc']['stale'][i]:9.4f} "
                  f"{out['delta_over_hidden'][i]:12.4f}")


if __name__ == "__main__":
    main()
