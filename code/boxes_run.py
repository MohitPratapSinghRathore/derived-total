"""Domain 2 experiment: permutation tracking, with the same decomposition.

Per condition and seed, in depth buckets:
  answer accuracy
  E_state  = P(the queried object's location is not decodable at the query)
  E_plan   = P(wrong answer | that location IS decodable)
  BELIEF   = P(a wrong answer is the box the model's own probe says holds it)

The belief measurement is exact here, more so than in chess: the answer is one
discrete symbol, so we can ask directly whether a wrong answer is the RIGHT
answer under the model's decoded assignment. Controls are the same in spirit:
the decoded assignment from another position at equal depth, and chance.
"""
import os, json, time, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from model import ChessLM, param_count
from boxes import gen, answer_weights, TOK, VOCAB, SEQ, NOBJ, MAX_QUERIES, BOX0

RES = os.path.join(os.path.dirname(__file__), "..", "results")
BUCKETS = [(1, 3), (3, 6), (6, 10), (10, 15), (15, 20), (20, 25)]
DEV = "cuda"


def bucket_of(d):
    for i, (lo, hi) in enumerate(BUCKETS):
        if lo <= d < hi:
            return i
    return None


def build(cond, seed, d=256, n_layer=8, k=4, d_s=64, ffn_mult=4.0):
    torch.manual_seed(seed); np.random.seed(seed)
    return ChessLM(VOCAB, d=d, n_layer=n_layer, n_head=8, max_len=SEQ,
                   alsb=cond.startswith("alsb"), d_s=d_s, k=k, ffn_mult=ffn_mult,
                   n_state_cls=NOBJ, n_state_slots=NOBJ).to(DEV)


def train(m, cond, seed, data, steps, bs, lr=3e-4, w=None):
    toks, lens, state = data[0], data[1], data[2]
    lam = 1.0 if cond == "alsb_l1" else 0.0
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=0.01, betas=(0.9, 0.95))
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps,
                                              pct_start=0.05)
    scaler = torch.amp.GradScaler("cuda")
    rng = np.random.default_rng(seed)
    N = len(toks); t0 = time.time(); log = []
    for st in range(1, steps + 1):
        idx = rng.integers(0, N, bs)
        x = torch.from_numpy(toks[idx].astype(np.int64)).to(DEV)
        pad = torch.from_numpy(np.arange(SEQ)[None] >= lens[idx][:, None]).to(DEV)
        m.train()
        with torch.amp.autocast("cuda", dtype=torch.float16):
            logits, _, sl = m(x[:, :-1])
            tgt = x[:, 1:].clone(); tgt[pad[:, 1:]] = -100
            if w is None:
                loss_lm = F.cross_entropy(logits.reshape(-1, VOCAB), tgt.reshape(-1),
                                          ignore_index=-100)
            else:
                ww = torch.from_numpy(w[idx][:, 1:]).to(DEV)
                per = F.cross_entropy(logits.reshape(-1, VOCAB), tgt.reshape(-1),
                                      ignore_index=-100, reduction="none")
                per = per.reshape(tgt.shape)
                loss_lm = (per * ww).sum() / ww[tgt != -100].sum()
            loss = loss_lm
            if lam > 0:
                s = torch.from_numpy(state[idx].astype(np.int64)).to(DEV)[:, :-1]
                mm = ~pad[:, :-1]
                slg = sl.reshape(bs, SEQ - 1, NOBJ, NOBJ)
                loss = loss + lam * F.cross_entropy(slg[mm].reshape(-1, NOBJ),
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


def fit_probe(m, data, n_layer, d=256, epochs=4, bs=64):
    """Fit at QUERY positions only.

    An earlier version trained across all token positions. State information is
    concentrated at the query, so the average gradient was dominated by positions
    carrying no signal and the probe collapsed to the marginal, reporting exact
    chance while the model answered perfectly. Probe where the decision is made.
    """
    toks, lens, state, qpos = data[0], data[1], data[2], data[3]
    probes = [nn.Linear(d, NOBJ * NOBJ).to(DEV) for _ in range(n_layer)]
    opt = torch.optim.AdamW([p for pr in probes for p in pr.parameters()], lr=1e-3)
    N = len(toks)
    for _ in range(epochs):
        order = np.random.permutation(N)
        for i in range(0, N, bs):
            idx = order[i:i + bs]
            x = torch.from_numpy(toks[idx].astype(np.int64)).to(DEV)
            s = torch.from_numpy(state[idx].astype(np.int64)).to(DEV)
            vm = np.zeros((len(idx), SEQ), dtype=bool)
            for r, gi in enumerate(idx):
                for q in qpos[gi]:
                    if q >= 0:
                        vm[r, int(q) - 1] = True      # the query object token
            v = torch.from_numpy(vm).to(DEV)
            with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.float16):
                _, hs, _ = m(x, return_hidden=True)
            loss = 0
            for li in range(n_layer):
                lg = probes[li](hs[li].float()).reshape(len(idx), SEQ, NOBJ, NOBJ)
                loss = loss + F.cross_entropy(lg[v].reshape(-1, NOBJ), s[v].reshape(-1))
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    return probes


@torch.no_grad()
def measure(m, probes, data, n_layer, bs=64, seed=0):
    toks, lens, state, qpos, qdepth, qobj = data
    nb = len(BUCKETS)
    ans_hit = np.zeros(nb); ans_n = np.zeros(nb)
    st_hit = np.zeros((n_layer, nb)); st_n = np.zeros(nb)
    qobj_hit = np.zeros((n_layer, nb))
    rng = np.random.default_rng(seed)
    bank = {b: [] for b in range(nb)}
    per_layer_cases = {li: {"est_bad": np.zeros(nb), "plan_bad": np.zeros(nb),
                            "plan_n": np.zeros(nb), "bel_hit": np.zeros(nb),
                            "bel_n": np.zeros(nb), "mis_hit": np.zeros(nb),
                            "mis_n": np.zeros(nb)} for li in range(n_layer)}
    for i in range(0, len(toks), bs):
        idx = np.arange(i, min(i + bs, len(toks)))
        x = torch.from_numpy(toks[idx].astype(np.int64)).to(DEV)
        s = torch.from_numpy(state[idx].astype(np.int64)).to(DEV)
        with torch.amp.autocast("cuda", dtype=torch.float16):
            logits, hs, _ = m(x, return_hidden=True)
        pred = logits.float().argmax(-1).cpu().numpy()
        dec = [probes[li](hs[li].float()).reshape(len(idx), SEQ, NOBJ, NOBJ)
               .argmax(-1).cpu().numpy() for li in range(n_layer)]
        for r, gi in enumerate(idx):
            for qi in range(MAX_QUERIES):
                pos = int(qpos[gi, qi])
                if pos < 0:
                    continue
                b = bucket_of(int(qdepth[gi, qi]))
                if b is None:
                    continue
                o = int(qobj[gi, qi])
                qtok = pos - 1                       # the object token
                true_box = int(state[gi, pos, o])
                pred_box = int(pred[r, pos - 1]) - BOX0
                ok = (pred_box == true_box)
                ans_n[b] += 1; ans_hit[b] += ok
                st_n[b] += 1
                for li in range(n_layer):
                    st_hit[li, b] += float((dec[li][r, qtok] == state[gi, qtok]).all())
                    qobj_hit[li, b] += float(dec[li][r, qtok, o] == int(state[gi, qtok, o]))
                    C = per_layer_cases[li]
                    believed = int(dec[li][r, qtok, o])
                    intact = (believed == true_box)
                    C["est_bad"][b] += (not intact)
                    if intact:
                        C["plan_n"][b] += 1
                        C["plan_bad"][b] += (not ok)
                    if not ok:
                        # belief consistency: is the wrong answer right under
                        # the model's own decoded assignment?
                        C["bel_n"][b] += 1
                        C["bel_hit"][b] += (pred_box == believed)
                        if bank[b]:
                            other = bank[b][int(rng.integers(len(bank[b])))]
                            C["mis_n"][b] += 1
                            C["mis_hit"][b] += (pred_box == int(other[o]))
                if len(bank[b]) < 500:
                    bank[b].append(dec[-1][r, qtok].copy())
    return dict(ans_acc=ans_hit / np.maximum(ans_n, 1),
                state_acc=st_hit / np.maximum(st_n, 1),
                qobj_acc=qobj_hit / np.maximum(st_n, 1),
                n=ans_n, st_n=st_n, cases=per_layer_cases)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conds", default="attn,attnpm,alsb_l0,alsb_l1")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--steps", type=int, default=12000)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--n_train", type=int, default=150000)
    ap.add_argument("--n_probe", type=int, default=4000)
    ap.add_argument("--n_eval", type=int, default=4000)
    args = ap.parse_args()
    os.makedirs(RES, exist_ok=True)

    print("generating data", flush=True)
    tr = gen(args.n_train, seed=2024)
    pr = gen(args.n_probe, seed=2025)
    ev = gen(args.n_eval, seed=2026)
    trw = answer_weights(tr[0], tr[1], tr[3], weight=5.0)

    for cond in args.conds.split(","):
        for seed in [int(s) for s in args.seeds.split(",")]:
            tag = f"boxes_{cond}_s{seed}"
            fp = os.path.join(RES, f"{tag}.json")
            if os.path.exists(fp):
                print("[skip]", tag, flush=True); continue
            print(f"[run] {tag}", flush=True)
            m = build(cond, seed, ffn_mult=4.156 if cond == "attnpm" else 4.0)
            log = train(m, cond, seed, tr, args.steps, args.bs, w=trw)
            m.eval()
            nl = len(m.blocks)
            probes = fit_probe(m, pr, nl)
            res = measure(m, probes, ev, nl)
            best = int(np.argmax(res["qobj_acc"].mean(1)))
            C = res["cases"][best]
            rec = dict(cond=cond, seed=seed, params=param_count(m), buckets=BUCKETS,
                       best_layer=best, train_log=log, chance=1.0 / NOBJ,
                       ans_acc=res["ans_acc"].tolist(),
                       state_acc=res["state_acc"].tolist(),
                       qobj_acc=res["qobj_acc"].tolist(),
                       n=res["n"].tolist(),
                       estate=(C["est_bad"] / np.maximum(res["st_n"], 1)).tolist(),
                       eplan=(C["plan_bad"] / np.maximum(C["plan_n"], 1)).tolist(),
                       eplan_n=C["plan_n"].tolist(),
                       belief=(C["bel_hit"] / np.maximum(C["bel_n"], 1)).tolist(),
                       belief_n=C["bel_n"].tolist(),
                       belief_mismatched=(C["mis_hit"] / np.maximum(C["mis_n"], 1)).tolist(),
                       belief_overall=float(C["bel_hit"].sum() / max(C["bel_n"].sum(), 1)),
                       belief_mismatched_overall=float(C["mis_hit"].sum() / max(C["mis_n"].sum(), 1)))
            torch.save({"model": m.state_dict(), "cond": cond, "seed": seed},
                       os.path.join(RES, "..", "runs", f"{tag}.pt"))
            json.dump(rec, open(fp, "w"), indent=1)
            print(f"  {tag}: ans {np.round(res['ans_acc'], 3)}", flush=True)
            print(f"      probe(queried obj) {np.round(res['qobj_acc'][best], 3)}", flush=True)
            print(f"      E_state {np.round(rec['estate'], 3)}  "
                  f"E_plan {np.round(rec['eplan'], 3)}", flush=True)
            print(f"      belief {rec['belief_overall']:.3f} vs mismatched "
                  f"{rec['belief_mismatched_overall']:.3f} "
                  f"(chance {1.0/(NOBJ-1):.3f})", flush=True)
    print("BOXES COMPLETE", flush=True)


if __name__ == "__main__":
    main()
