"""Learn the transition operator on a FROZEN model, using no oracle.

Training the consistency objective jointly with the language model degrades the
language model: at weight 0.5 validation loss is far worse than the baseline at
the same step. That reproduces the pattern of the auxiliary state objective in
Paper 1, where an extra loss bought representation quality at the cost of the
primary task.

Freezing the model removes the conflict entirely and makes the method more
useful rather than less. Nothing about the base model changes, so the operator
can be fitted to a model that already exists, and the repair becomes an
inference-time intervention rather than a training recipe.

What is learned is small: a projection into a state subspace, a move embedding,
and a linear transition. The supervision is the consistency condition alone,
which reads only the model's own activations and the tokens already in the
context. The oracle is never consulted.
"""
import os, json, argparse, time
import numpy as np
import torch
import torch.nn.functional as F
from probes import load_split, build
from scr import StateConsistency

DATA = os.path.join(os.path.dirname(__file__), "..", "data", "proc")
RUNS = os.path.join(os.path.dirname(__file__), "..", "runs")
RES = os.path.join(os.path.dirname(__file__), "..", "results")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="attn_8L256_s0")
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--layer", type=int, default=-1)
    ap.add_argument("--lr", type=float, default=1e-3)
    args = ap.parse_args()

    model, a, ck = build(os.path.join(RUNS, f"{args.name}.pt"))
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()

    stoi = json.load(open(os.path.join(DATA, "vocab.json")))
    d = np.load(os.path.join(DATA, "train_probe.npz"))
    toks, lens = d["toks"].astype(np.int64), d["lens"].astype(np.int64)

    scr = StateConsistency(a["width"], d_state=args.dim, vocab=len(stoi)).cuda()
    opt = torch.optim.AdamW(scr.parameters(), lr=args.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr,
                                                total_steps=args.steps,
                                                pct_start=0.05)
    rng = np.random.default_rng(0)
    L = toks.shape[1]
    t0 = time.time()
    log = []

    for step in range(1, args.steps + 1):
        idx = np.sort(rng.integers(0, len(toks), args.bs))
        x = torch.from_numpy(toks[idx]).cuda()
        pad = torch.from_numpy(np.arange(L)[None] >= lens[idx][:, None]).cuda()
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.float16):
            _, hs, _ = model(x[:, :-1], return_hidden=True)
        h = hs[args.layer].float()
        x_next = x[:, 1:][:, :h.shape[1]]
        valid = (~pad[:, :-1])[:, :h.shape[1]]
        loss, acc = scr(h, x_next, valid)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(scr.parameters(), 1.0)
        opt.step(); sched.step()
        if step % 500 == 0 or step == args.steps:
            log.append({"step": step, "loss": float(loss), "acc": float(acc)})
            print(f"  step {step} consistency loss {float(loss):.4f} "
                  f"acc {float(acc):.3f} [{time.time()-t0:.0f}s]", flush=True)

    out = os.path.join(RUNS, f"{args.name}_trans.pt")
    torch.save({"scr": scr.state_dict(), "args": vars(args), "log": log}, out)
    print("saved", os.path.basename(out), flush=True)


if __name__ == "__main__":
    main()
