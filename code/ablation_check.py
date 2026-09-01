"""Why does zeroing the recurrent read-out improve behaviour?

Table 4 shows the lambda=1 model with W_out zeroed at test time producing a LOWER
illegal-move rate and a LONGER horizon than the same model intact. That is
counterintuitive: the weights were trained with the read-out in place, so removing
it should hurt. Before explaining it we measure it properly.

We report, for every seed: language-model loss with and without the read-out, the
norm of what the read-out injects relative to the residual stream it is added to,
and the illegal-move rate. If ablation lowers LM loss as well, the read-out was
net harmful to next-move prediction and the auxiliary objective bought
decodability at the cost of behaviour. If ablation raises LM loss while lowering
the illegal rate, the two measures are pulling apart and that needs saying.
"""
import os, json
import numpy as np
import torch
import torch.nn.functional as F
from probes import load_split, build

DATA = os.path.join(os.path.dirname(__file__), "..", "data", "proc")
RUNS = os.path.join(os.path.dirname(__file__), "..", "runs")
RES = os.path.join(os.path.dirname(__file__), "..", "results")


@torch.no_grad()
def lm_loss(model, toks, lens, bs=32, n=1500):
    tot, cnt = 0.0, 0
    for i in range(0, min(n, len(toks)), bs):
        idx = np.arange(i, min(i + bs, min(n, len(toks))))
        x = torch.from_numpy(toks[idx]).cuda()
        pad = torch.from_numpy(np.arange(toks.shape[1])[None] >= lens[idx][:, None]).cuda()
        with torch.amp.autocast("cuda", dtype=torch.float16):
            logits, _, _ = model(x[:, :-1])
        tgt = x[:, 1:].clone(); tgt[pad[:, 1:]] = -100
        l = F.cross_entropy(logits.float().reshape(-1, logits.size(-1)),
                            tgt.reshape(-1), ignore_index=-100, reduction="sum")
        tot += float(l); cnt += int((tgt != -100).sum())
    return tot / max(cnt, 1)


@torch.no_grad()
def injection_ratio(model, toks, lens, bs=16, n=200):
    """||W_out z|| relative to ||h|| at the insertion points."""
    ratios = []
    hooks, store = [], {}

    for name, mod in model.alsb_mods.items():
        def mk(nm):
            def hook(m, inp, out):
                x = inp[0]
                h_before = x
                z = out[1] if isinstance(out, tuple) else None
                if z is not None:
                    inj = m.w_out(z)
                    store[nm] = (float(inj.norm(dim=-1).mean()),
                                 float(h_before.norm(dim=-1).mean()))
                return out
            return hook
        hooks.append(mod.register_forward_hook(mk(name)))

    for i in range(0, min(n, len(toks)), bs):
        idx = np.arange(i, min(i + bs, min(n, len(toks))))
        x = torch.from_numpy(toks[idx]).cuda()
        with torch.amp.autocast("cuda", dtype=torch.float16):
            model(x[:, :-1])
        for nm, (a, b) in store.items():
            ratios.append(a / max(b, 1e-6))
    for h in hooks:
        h.remove()
    return float(np.mean(ratios)) if ratios else None


def main():
    toks, lens, occ = load_split("eval", 1500)
    out = {}
    for s in (0, 1, 2):
        name = f"alsb_l1_8L256_s{s}"
        ck = os.path.join(RUNS, f"{name}.pt")
        if not os.path.exists(ck):
            continue
        model, a, c = build(ck)
        intact = lm_loss(model, toks, lens)
        ratio = injection_ratio(model, toks, lens)
        for mod in model.alsb_mods.values():
            torch.nn.init.zeros_(mod.w_out.weight)
            torch.nn.init.zeros_(mod.w_out.bias)
        ablated = lm_loss(model, toks, lens)
        out[name] = {"lm_loss_intact": intact, "lm_loss_ablated": ablated,
                     "delta": ablated - intact, "injection_ratio": ratio}
        print(f"{name}: LM loss intact {intact:.4f} -> ablated {ablated:.4f} "
              f"(delta {ablated-intact:+.4f}); injection/residual norm {ratio:.4f}",
              flush=True)
    json.dump(out, open(os.path.join(RES, "readout_ablation_check.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
