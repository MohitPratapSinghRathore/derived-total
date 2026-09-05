"""Stage 3: train one model cell.

Every condition (attention-only / ALSB lambda=0 / ALSB lambda>0 / param-matched
control) runs through this same script; only argv differs.
"""
import os, sys, json, time, math, argparse
import numpy as np
import torch
import torch.nn.functional as F
from model import ChessLM, param_count
from scr import StateConsistency

DATA = os.path.join(os.path.dirname(__file__), "..", "data", "proc")
CKPT = os.path.join(os.path.dirname(__file__), "..", "runs")


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True)
    p.add_argument("--layers", type=int, default=8)
    p.add_argument("--width", type=int, default=256)
    p.add_argument("--heads", type=int, default=8)
    p.add_argument("--ffn_mult", type=float, default=4.0)
    p.add_argument("--alsb", action="store_true")
    p.add_argument("--d_s", type=int, default=64)
    p.add_argument("--k", type=int, default=4)
    p.add_argument("--lam", type=float, default=0.0)
    p.add_argument("--scr", type=float, default=0.0,
                   help="weight on state-consistency regularisation")
    p.add_argument("--scr_dim", type=int, default=64)
    p.add_argument("--scr_layer", type=int, default=-1,
                   help="layer whose residual stream is regularised; -1 = last")
    p.add_argument("--steps", type=int, default=12000)
    p.add_argument("--bs", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main():
    a = get_args()
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    dev = "cuda"
    stoi = json.load(open(os.path.join(DATA, "vocab.json")))
    d = np.load(os.path.join(DATA, "train_lm.npz"))
    toks, lens = d["toks"].astype(np.int64), d["lens"].astype(np.int64)
    N, L = toks.shape

    need_state = a.alsb and a.lam > 0
    occ = np.load(os.path.join(DATA, "train_lm_occ.npy"), mmap_mode="r") if need_state else None

    # held-out slice of train_lm for val loss (still disjoint from eval split)
    n_val = 2000
    val_idx = np.arange(N - n_val, N)
    tr_hi = N - n_val

    model = ChessLM(len(stoi), d=a.width, n_layer=a.layers, n_head=a.heads,
                    max_len=L, alsb=a.alsb, d_s=a.d_s, k=a.k,
                    ffn_mult=a.ffn_mult).to(dev)
    scr = None
    if a.scr > 0:
        scr = StateConsistency(a.width, d_state=a.scr_dim, vocab=len(stoi)).to(dev)
    print(f"{a.name}: params {param_count(model)/1e6:.2f}M"
          + (f" + scr {sum(q.numel() for q in scr.parameters())/1e6:.2f}M"
             if scr else ""), flush=True)
    params = list(model.parameters()) + (list(scr.parameters()) if scr else [])
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.01, betas=(0.9, 0.95))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.steps,
                                                pct_start=0.05)
    scaler = torch.amp.GradScaler("cuda")
    rng = np.random.default_rng(a.seed)

    def batch(idx):
        x = torch.from_numpy(toks[idx]).to(dev)
        pad = torch.from_numpy((np.arange(L)[None] >= lens[idx][:, None])).to(dev)
        s = None
        if need_state:
            s = torch.from_numpy(np.asarray(occ[idx]).astype(np.int64)).to(dev)
        return x, pad, s

    t0 = time.time()
    log = []
    for step in range(1, a.steps + 1):
        model.train()
        idx = np.sort(rng.integers(0, tr_hi, a.bs))
        x, pad, s = batch(idx)
        with torch.amp.autocast("cuda", dtype=torch.float16):
            logits, hs_all, state_logits = model(x[:, :-1],
                                                 return_hidden=scr is not None)
            tgt = x[:, 1:].clone()
            tgt[pad[:, 1:]] = -100
            loss_lm = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                                      tgt.reshape(-1), ignore_index=-100)
            loss = loss_lm
            loss_st = torch.tensor(0.0, device=dev)
            loss_scr = torch.tensor(0.0, device=dev)
            scr_acc = torch.tensor(0.0, device=dev)
            if scr is not None:
                h = hs_all[a.scr_layer].float()
                # token read after position t is x[:, t+1]
                x_next = x[:, 1:][:, :h.shape[1]]
                valid = (~pad[:, :-1])[:, :h.shape[1]]
                loss_scr, scr_acc = scr(h, x_next, valid)
                loss = loss + a.scr * loss_scr
            if need_state:
                sl = state_logits.reshape(a.bs, L - 1, 64, 13)
                st = s[:, :-1]                      # state after consuming token t
                m = ~pad[:, :-1]
                loss_st = F.cross_entropy(sl[m].reshape(-1, 13), st[m].reshape(-1))
                loss = loss_lm + a.lam * loss_st
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        scaler.step(opt)
        scaler.update()
        sched.step()

        if step % 500 == 0 or step == a.steps:
            model.eval()
            with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.float16):
                vls = []
                for j in range(0, n_val, 64):
                    vi = val_idx[j:j + 64]
                    x, pad, _ = batch(vi)
                    lg, _, _ = model(x[:, :-1])
                    tg = x[:, 1:].clone(); tg[pad[:, 1:]] = -100
                    vls.append(F.cross_entropy(lg.reshape(-1, lg.size(-1)),
                                               tg.reshape(-1), ignore_index=-100).item())
            v = float(np.mean(vls))
            log.append({"step": step, "train_lm": loss_lm.item(),
                        "state": float(loss_st), "scr": float(loss_scr),
                        "scr_acc": float(scr_acc), "val": v})
            print(f"  step {step} lm {loss_lm.item():.4f} state {float(loss_st):.4f} "
                  f"scr {float(loss_scr):.4f} acc {float(scr_acc):.3f} "
                  f"val {v:.4f} [{time.time()-t0:.0f}s]", flush=True)

    os.makedirs(CKPT, exist_ok=True)
    torch.save({"model": model.state_dict(), "args": vars(a), "log": log,
                "params": param_count(model),
                "scr": scr.state_dict() if scr is not None else None},
               os.path.join(CKPT, f"{a.name}.pt"))
    print("saved", a.name, flush=True)


if __name__ == "__main__":
    main()
