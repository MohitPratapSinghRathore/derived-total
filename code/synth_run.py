"""Domain 2 experiment: train, probe, and decompose error by reasoning depth.

Per condition and seed we report, in depth buckets:
  answer accuracy              (end task)
  E_state  = P(the two queried variables are NOT both decodable at the query)
  E_plan   = P(wrong answer | both queried variables ARE decodable)

The E_plan conditioning is exact here rather than approximate: the generator
records which two variables each answer depends on, so "state intact" is a
statement about precisely the quantities the answer needs.
"""
import os, json, time, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from model import ChessLM, param_count
import synth
from synth import gen, TOK, VOCAB, SEQ, NVAR, MOD, MAX_QUERIES

RES = os.path.join(os.path.dirname(__file__), "..", "results")
BUCKETS = [(1, 8), (8, 14), (14, 20), (20, 26), (26, 33)]
DEV = "cuda"


def bucket_of(d):
    for i, (lo, hi) in enumerate(BUCKETS):
        if lo <= d < hi:
            return i
    return None


def build(cond, seed, d=256, n_layer=8, k=4, d_s=64, ffn_mult=4.0):
    torch.manual_seed(seed); np.random.seed(seed)
    alsb = cond.startswith("alsb")
    return ChessLM(VOCAB, d=d, n_layer=n_layer, n_head=8, max_len=SEQ, alsb=alsb,
                   d_s=d_s, k=k, ffn_mult=ffn_mult, n_state_cls=MOD,
                   n_state_slots=NVAR).to(DEV)


def train(m, cond, seed, data, steps, bs, lr=3e-4):
    toks, lens, state = data[0], data[1], data[2]
    lam = 1.0 if cond == "alsb_l1" else 0.0
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=0.01, betas=(0.9, 0.95))
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps,
                                              pct_start=0.05)
    scaler = torch.amp.GradScaler("cuda")
    rng = np.random.default_rng(seed)
    N = len(toks)
    t0 = time.time()
    log = []
    for st in range(1, steps + 1):
        idx = rng.integers(0, N, bs)
        x = torch.from_numpy(toks[idx].astype(np.int64)).to(DEV)
        pad = torch.from_numpy(np.arange(SEQ)[None] >= lens[idx][:, None]).to(DEV)
        m.train()
        with torch.amp.autocast("cuda", dtype=torch.float16):
            logits, _, sl = m(x[:, :-1])
            tgt = x[:, 1:].clone(); tgt[pad[:, 1:]] = -100
            loss_lm = F.cross_entropy(logits.reshape(-1, VOCAB), tgt.reshape(-1),
                                      ignore_index=-100)
            loss = loss_lm
            if lam > 0:
                s = torch.from_numpy(state[idx].astype(np.int64)).to(DEV)[:, :-1]
                mm = ~pad[:, :-1]
                slg = sl.reshape(bs, SEQ - 1, NVAR, MOD)
                loss = loss + lam * F.cross_entropy(slg[mm].reshape(-1, MOD),
                                                    s[mm].reshape(-1))
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        scaler.step(opt); scaler.update(); sch.step()
        if st % 1000 == 0 or st == steps:
            log.append({"step": st, "lm": loss_lm.item()})
            print(f"    {cond} s{seed} step {st} lm {loss_lm.item():.4f} "
                  f"[{time.time()-t0:.0f}s]", flush=True)
    return log


def fit_probe(m, data, n_layer, d=256, epochs=2, bs=64):
    toks, lens, state = data[0], data[1], data[2]
    probes = [nn.Linear(d, NVAR * MOD).to(DEV) for _ in range(n_layer)]
    opt = torch.optim.AdamW([p for pr in probes for p in pr.parameters()], lr=1e-3)
    N = len(toks)
    for ep in range(epochs):
        order = np.random.permutation(N)
        for i in range(0, N, bs):
            idx = order[i:i + bs]
            x = torch.from_numpy(toks[idx].astype(np.int64)).to(DEV)
            s = torch.from_numpy(state[idx].astype(np.int64)).to(DEV)
            v = torch.from_numpy(np.arange(SEQ)[None] < lens[idx][:, None]).to(DEV)
            with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.float16):
                _, hs, _ = m(x, return_hidden=True)
            loss = 0
            for li in range(n_layer):
                lg = probes[li](hs[li].float()).reshape(len(idx), SEQ, NVAR, MOD)
                loss = loss + F.cross_entropy(lg[v].reshape(-1, MOD), s[v].reshape(-1))
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    return probes


@torch.no_grad()
def measure(m, probes, data, n_layer, bs=64):
    toks, lens, state, qpos, qdepth, qvars = data
    nb = len(BUCKETS)
    ans_hit = np.zeros(nb); ans_n = np.zeros(nb)
    st_hit = np.zeros((n_layer, nb)); st_n = np.zeros(nb)
    plan_bad = np.zeros(nb); plan_n = np.zeros(nb)
    est_bad = np.zeros(nb)
    N = len(toks)
    for i in range(0, N, bs):
        idx = np.arange(i, min(i + bs, N))
        x = torch.from_numpy(toks[idx].astype(np.int64)).to(DEV)
        s = torch.from_numpy(state[idx].astype(np.int64)).to(DEV)
        with torch.amp.autocast("cuda", dtype=torch.float16):
            logits, hs, _ = m(x, return_hidden=True)
        logits = logits.float()
        pred = logits.argmax(-1).cpu().numpy()          # pred[r,t] predicts token t+1
        pl = [probes[li](hs[li].float()).reshape(len(idx), SEQ, NVAR, MOD).argmax(-1)
              for li in range(n_layer)]
        for r, gi in enumerate(idx):
            for qi in range(MAX_QUERIES):
                pos = int(qpos[gi, qi])
                if pos < 0:
                    continue
                b = bucket_of(int(qdepth[gi, qi]))
                if b is None:
                    continue
                va, vb = int(qvars[gi, qi, 0]), int(qvars[gi, qi, 1])
                qtok = pos - 3                       # the <query> token position
                ans_n[b] += 1
                ok_ans = int(pred[r, pos - 1] == toks[gi, pos])
                ans_hit[b] += ok_ans
                st_n[b] += 1
                for li in range(n_layer):
                    st_hit[li, b] += float((pl[li][r, qtok] == s[r, qtok]).all())
                # exact conditioning: only the two variables the answer needs
                best = None   # filled by caller; here use last layer for E_plan
                intact = bool((pl[-1][r, qtok, va] == s[r, qtok, va]) and
                              (pl[-1][r, qtok, vb] == s[r, qtok, vb]))
                est_bad[b] += (not intact)
                if intact:
                    plan_n[b] += 1
                    plan_bad[b] += (1 - ok_ans)
    return dict(ans_acc=(ans_hit / np.maximum(ans_n, 1)),
                state_acc=(st_hit / np.maximum(st_n, 1)),
                estate=(est_bad / np.maximum(st_n, 1)),
                eplan=(plan_bad / np.maximum(plan_n, 1)),
                eplan_n=plan_n, n=ans_n)


@torch.no_grad()
def measure_eplan_at(m, probe, layer, data, bs=64):
    """E_state / E_plan using the probe at the chosen layer."""
    toks, lens, state, qpos, qdepth, qvars = data
    nb = len(BUCKETS)
    plan_bad = np.zeros(nb); plan_n = np.zeros(nb)
    est_bad = np.zeros(nb); est_n = np.zeros(nb)
    for i in range(0, len(toks), bs):
        idx = np.arange(i, min(i + bs, len(toks)))
        x = torch.from_numpy(toks[idx].astype(np.int64)).to(DEV)
        s = torch.from_numpy(state[idx].astype(np.int64)).to(DEV)
        with torch.amp.autocast("cuda", dtype=torch.float16):
            logits, hs, _ = m(x, return_hidden=True)
        pred = logits.float().argmax(-1).cpu().numpy()
        pl = probe(hs[layer].float()).reshape(len(idx), SEQ, NVAR, MOD).argmax(-1)
        for r, gi in enumerate(idx):
            for qi in range(MAX_QUERIES):
                pos = int(qpos[gi, qi])
                if pos < 0:
                    continue
                b = bucket_of(int(qdepth[gi, qi]))
                if b is None:
                    continue
                va, vb = int(qvars[gi, qi, 0]), int(qvars[gi, qi, 1])
                q = pos - 3
                ok_ans = int(pred[r, pos - 1] == toks[gi, pos])
                intact = bool((pl[r, q, va] == s[r, q, va]) and
                              (pl[r, q, vb] == s[r, q, vb]))
                est_n[b] += 1
                est_bad[b] += (not intact)
                if intact:
                    plan_n[b] += 1
                    plan_bad[b] += (1 - ok_ans)
    return dict(estate=(est_bad / np.maximum(est_n, 1)).tolist(),
                eplan=(plan_bad / np.maximum(plan_n, 1)).tolist(),
                eplan_n=plan_n.tolist())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conds", default="attn,attnpm,alsb_l0,alsb_l1")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--n_train", type=int, default=120000)
    ap.add_argument("--n_probe", type=int, default=4000)
    ap.add_argument("--n_eval", type=int, default=4000)
    args = ap.parse_args()
    os.makedirs(RES, exist_ok=True)

    print("generating data", flush=True)
    tr = gen(args.n_train, seed=1234)
    pr = gen(args.n_probe, seed=1235)
    ev = gen(args.n_eval, seed=1236)

    for cond in args.conds.split(","):
        for seed in [int(s) for s in args.seeds.split(",")]:
            tag = f"synth_{cond}_s{seed}"
            fp = os.path.join(RES, f"{tag}.json")
            if os.path.exists(fp):
                print("[skip]", tag, flush=True); continue
            print(f"[run] {tag}", flush=True)
            fm = 4.156 if cond == "attnpm" else 4.0
            m = build(cond, seed, ffn_mult=fm)
            log = train(m, cond, seed, tr, args.steps, args.bs)
            m.eval()
            n_layer = len(m.blocks)
            probes = fit_probe(m, pr, n_layer)
            res = measure(m, probes, ev, n_layer)
            best = int(np.argmax(res["state_acc"].mean(1)))
            ep = measure_eplan_at(m, probes[best], best, ev)
            rec = dict(cond=cond, seed=seed, params=param_count(m),
                       buckets=BUCKETS, best_layer=best, train_log=log,
                       ans_acc=res["ans_acc"].tolist(),
                       state_acc=res["state_acc"].tolist(),
                       n=res["n"].tolist(), **ep)
            json.dump(rec, open(fp, "w"), indent=1)
            print(f"  {tag}: ans {np.round(res['ans_acc'],3)} "
                  f"E_state {np.round(ep['estate'],3)} E_plan {np.round(ep['eplan'],3)}",
                  flush=True)
    print("SYNTH COMPLETE", flush=True)


if __name__ == "__main__":
    main()
