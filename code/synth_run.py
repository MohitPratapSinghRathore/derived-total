"""Domain 2 experiment: train, probe, and decompose error on variable-state tracking.

Produces, per condition and seed:
  answer accuracy vs depth        (end-task)
  probe state accuracy vs depth   (E_state)
  E_plan = P(wrong answer | the two queried variables are decodable at the query)

The E_plan conditioning is exact here: we know precisely which two variables the
answer depends on, so "state intact" is not an approximation.
"""
import os, json, time, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from model import ChessLM, param_count
from synth import gen, TOK, VOCAB, SEQ, NVAR, MOD, MAX_STEPS

RES = os.path.join(os.path.dirname(__file__), "..", "results")
DEPTH_BUCKETS = [(0, 8), (8, 16), (16, 24), (24, 32), (32, 41)]
VAL0 = TOK["val0"]


def make_data(n_train, n_probe, n_eval, seed=1234):
    tr = gen(n_train, seed=seed)
    pr = gen(n_probe, seed=seed + 1)
    ev = gen(n_eval, seed=seed + 2)
    return tr, pr, ev


def train(cond, seed, tr, steps, bs, dev="cuda", d=256, n_layer=8, k=4, d_s=64,
          lr=3e-4, ffn_mult=4.0):
    toks, lens, state, anspos = tr
    torch.manual_seed(seed); np.random.seed(seed)
    alsb = cond.startswith("alsb")
    lam = 1.0 if cond == "alsb_l1" else 0.0
    m = ChessLM(VOCAB, d=d, n_layer=n_layer, n_head=8, max_len=SEQ, alsb=alsb,
                d_s=d_s, k=k, ffn_mult=ffn_mult, n_state_cls=MOD,
                n_state_slots=NVAR).to(dev)
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=0.01, betas=(0.9, 0.95))
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.05)
    scaler = torch.amp.GradScaler("cuda")
    rng = np.random.default_rng(seed)
    N = len(toks)
    t0 = time.time()
    for st in range(1, steps + 1):
        idx = rng.integers(0, N, bs)
        x = torch.from_numpy(toks[idx]).to(dev)
        pad = torch.from_numpy(np.arange(SEQ)[None] >= lens[idx][:, None]).to(dev)
        m.train()
        with torch.amp.autocast("cuda", dtype=torch.float16):
            logits, _, sl = m(x[:, :-1])
            tgt = x[:, 1:].clone(); tgt[pad[:, 1:]] = -100
            loss = F.cross_entropy(logits.reshape(-1, VOCAB), tgt.reshape(-1), ignore_index=-100)
            if lam > 0:
                s = torch.from_numpy(state[idx]).to(dev)[:, :-1]
                mm = ~pad[:, :-1]
                slg = sl.reshape(bs, SEQ - 1, NVAR, MOD)
                loss = loss + lam * F.cross_entropy(slg[mm].reshape(-1, MOD),
                                                    s[mm].reshape(-1))
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        scaler.step(opt); scaler.update(); sch.step()
        if st % 1000 == 0:
            print(f"    {cond} s{seed} step {st} loss {loss.item():.4f} "
                  f"[{time.time()-t0:.0f}s]", flush=True)
    return m


def fit_probe(m, pr, epochs=2, bs=64, dev="cuda", d=256, n_layer=8):
    """Probe every layer for the 8 variable values; keep the best layer."""
    toks, lens, state, anspos = pr
    probes = [nn.Linear(d, NVAR * MOD).to(dev) for _ in range(n_layer)]
    opt = torch.optim.AdamW([p for pr_ in probes for p in pr_.parameters()], lr=1e-3)
    N = len(toks)
    for ep in range(epochs):
        order = np.random.permutation(N)
        for i in range(0, N, bs):
            idx = order[i:i + bs]
            x = torch.from_numpy(toks[idx]).to(dev)
            s = torch.from_numpy(state[idx]).to(dev)
            v = torch.from_numpy(np.arange(SEQ)[None] < lens[idx][:, None]).to(dev)
            with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.float16):
                _, hs, _ = m(x, return_hidden=True)
            loss = 0
            for li in range(n_layer):
                lg = probes[li](hs[li].float()).reshape(len(idx), SEQ, NVAR, MOD)
                loss = loss + F.cross_entropy(lg[v].reshape(-1, MOD), s[v].reshape(-1))
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    return probes


@torch.no_grad()
def measure(m, probes, ev, dev="cuda", bs=64, n_layer=8):
    toks, lens, state, anspos = ev
    N = len(toks)
    nb = len(DEPTH_BUCKETS)
    st_hit = np.zeros((n_layer, nb)); st_n = np.zeros(nb)
    ans_hit = np.zeros(nb); ans_n = np.zeros(nb)
    plan_bad = np.zeros(nb); plan_n = np.zeros(nb)
    for i in range(0, N, bs):
        idx = np.arange(i, min(i + bs, N))
        x = torch.from_numpy(toks[idx]).to(dev)
        s = torch.from_numpy(state[idx]).to(dev)
        with torch.amp.autocast("cuda", dtype=torch.float16):
            logits, hs, _ = m(x, return_hidden=True)
        logits = logits.float()
        ap = anspos[idx]
        nsteps = (ap - 4) // 4
        # end-task: token predicted at position ap-1 should be the answer token
        pred_ans = logits[np.arange(len(idx)), ap - 1].argmax(-1).cpu().numpy()
        true_ans = toks[idx, ap]
        # probe state at the query position (ap-3 = <query> token)
        qpos = ap - 3
        preds = []
        for li in range(n_layer):
            pl = probes[li](hs[li].float()).reshape(len(idx), SEQ, NVAR, MOD).argmax(-1)
            preds.append(pl)
        for r in range(len(idx)):
            b = next((bi for bi, (lo, hi) in enumerate(DEPTH_BUCKETS)
                      if lo <= nsteps[r] < hi), None)
            if b is None:
                continue
            st_n[b] += 1; ans_n[b] += 1
            ok_ans = int(pred_ans[r] == true_ans[r])
            ans_hit[b] += ok_ans
            for li in range(n_layer):
                st_hit[li, b] += float((preds[li][r, qpos[r]] == s[r, qpos[r]]).all())
        # E_plan is computed separately, by measure_eplan, at the best layer
    return dict(state_acc=(st_hit / np.maximum(st_n, 1)),
                ans_acc=(ans_hit / np.maximum(ans_n, 1)),
                n=st_n)


@torch.no_grad()
def measure_eplan(m, probe, layer, ev, dev="cuda", bs=64):
    """E_plan: P(wrong answer | both queried variables decodable at the query)."""
    toks, lens, state, anspos = ev
    nb = len(DEPTH_BUCKETS)
    bad = np.zeros(nb); n = np.zeros(nb)
    e_state_bad = np.zeros(nb); e_state_n = np.zeros(nb)
    for i in range(0, len(toks), bs):
        idx = np.arange(i, min(i + bs, len(toks)))
        x = torch.from_numpy(toks[idx]).to(dev)
        s = torch.from_numpy(state[idx]).to(dev)
        with torch.amp.autocast("cuda", dtype=torch.float16):
            logits, hs, _ = m(x, return_hidden=True)
        logits = logits.float()
        ap = anspos[idx]; qpos = ap - 3
        nsteps = (ap - 4) // 4
        pred_ans = logits[np.arange(len(idx)), ap - 1].argmax(-1).cpu().numpy()
        pl = probe(hs[layer].float()).reshape(len(idx), SEQ, NVAR, MOD).argmax(-1)
        for r in range(len(idx)):
            b = next((bi for bi, (lo, hi) in enumerate(DEPTH_BUCKETS)
                      if lo <= nsteps[r] < hi), None)
            if b is None:
                continue
            va = int(toks[idx[r], ap[r] - 2] - TOK["v0"])
            vb = int(toks[idx[r], ap[r] - 1] - TOK["v0"])
            q = qpos[r]
            intact = bool((pl[r, q, va] == s[r, q, va]) and (pl[r, q, vb] == s[r, q, vb]))
            e_state_n[b] += 1
            e_state_bad[b] += (not intact)
            if intact:
                n[b] += 1
                bad[b] += int(pred_ans[r] != toks[idx[r], ap[r]])
    return dict(eplan=(bad / np.maximum(n, 1)).tolist(), eplan_n=n.tolist(),
                estate=(e_state_bad / np.maximum(e_state_n, 1)).tolist())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conds", default="attn,attnpm,alsb_l0,alsb_l1")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--n_train", type=int, default=200000)
    args = ap.parse_args()
    os.makedirs(RES, exist_ok=True)

    print("generating data", flush=True)
    tr, pr, ev = make_data(args.n_train, 20000, 20000)
    out = {}
    for cond in args.conds.split(","):
        for seed in [int(s) for s in args.seeds.split(",")]:
            tag = f"synth_{cond}_s{seed}"
            fp = os.path.join(RES, f"{tag}.json")
            if os.path.exists(fp):
                print("[skip]", tag, flush=True); continue
            print(f"[run] {tag}", flush=True)
            fm = 4.156 if cond == "attnpm" else 4.0
            m = train(cond if cond != "attnpm" else "attn", seed, tr,
                      args.steps, args.bs, ffn_mult=fm)
            m.eval()
            probes = fit_probe(m, pr)
            res = measure(m, probes, ev)
            best = int(np.argmax(res["state_acc"].mean(1)))
            ep = measure_eplan(m, probes[best], best, ev)
            rec = dict(cond=cond, seed=seed, params=param_count(m),
                       buckets=DEPTH_BUCKETS, best_layer=best,
                       state_acc=res["state_acc"].tolist(),
                       ans_acc=res["ans_acc"].tolist(),
                       n=res["n"].tolist(), **ep)
            json.dump(rec, open(fp, "w"), indent=1)
            print(f"  {tag}: ans_acc {np.round(res['ans_acc'],3)} "
                  f"E_state {np.round(ep['estate'],3)} E_plan {np.round(ep['eplan'],3)}",
                  flush=True)
    print("SYNTH COMPLETE", flush=True)


if __name__ == "__main__":
    main()
