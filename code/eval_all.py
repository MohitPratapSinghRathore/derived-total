"""Stage 5: full diagnostic for one checkpoint -> results/<name>.json"""
import os, sys, json, argparse
import numpy as np
import torch
from probes import load_split, build, fit_probes, evaluate
from legality import illegal_rate

DATA = os.path.join(os.path.dirname(__file__), "..", "data", "proc")
RUNS = os.path.join(os.path.dirname(__file__), "..", "runs")
RES = os.path.join(os.path.dirname(__file__), "..", "results")


def horizon(curve, buckets, eps=0.05):
    """H_eps: first bucket midpoint where exact-state fidelity drops below 1-eps."""
    for (lo, hi), v in zip(buckets, curve):
        if v < 1 - eps:
            return lo
    return buckets[-1][1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--probe_games", type=int, default=6000)
    ap.add_argument("--eval_games", type=int, default=6000)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--controls", action="store_true")
    ap.add_argument("--ablate_z", action="store_true")
    args = ap.parse_args()
    os.makedirs(RES, exist_ok=True)

    ck = os.path.join(RUNS, f"{args.name}.pt")
    model, a, c = build(ck)
    if args.ablate_z:
        # test-time ablation: zero the ALSB read-out, keep everything else
        for mod in model.alsb_mods.values():
            torch.nn.init.zeros_(mod.w_out.weight)
            torch.nn.init.zeros_(mod.w_out.bias)

    ptk, plen, pocc = load_split("train_probe", args.probe_games)
    etk, elen, eocc = load_split("eval", args.eval_games)
    stoi = json.load(open(os.path.join(DATA, "vocab.json")))
    itos = {v: k for k, v in stoi.items()}

    out = {"name": args.name, "args": a, "params": c["params"],
           "train_log": c["log"], "ablate_z": args.ablate_z}

    print(f"[{args.name}] fitting probes", flush=True)
    probes = fit_probes(model, a, ptk, plen, pocc, epochs=args.epochs)
    exact, sqacc, cnt, buckets = evaluate(model, a, probes, etk, elen, eocc)
    best = int(np.argmax(exact.mean(1)))
    out.update(buckets=buckets, counts=cnt.tolist(),
               fidelity_exact=exact.tolist(), fidelity_sq=sqacc.tolist(),
               best_layer=best,
               H05=horizon(exact[best], buckets, 0.05),
               H20=horizon(exact[best], buckets, 0.20))

    if args.controls:
        print(f"[{args.name}] control task (shuffled labels)", flush=True)
        cp = fit_probes(model, a, ptk, plen, pocc, epochs=args.epochs, control=True)
        ce, cs, _, _ = evaluate(model, a, cp, etk, elen, eocc)
        out["control_shuffled_sq"] = cs.tolist()

        print(f"[{args.name}] randomised-weight baseline", flush=True)
        rm, _, _ = build(ck, randomised=True)
        rp = fit_probes(rm, a, ptk, plen, pocc, epochs=args.epochs)
        re_, rs, _, _ = evaluate(rm, a, rp, etk, elen, eocc)
        out["control_random_exact"] = re_.tolist()
        out["control_random_sq"] = rs.tolist()

    print(f"[{args.name}] illegal-move rate", flush=True)
    ill, lmass, tot, _ = illegal_rate(model, etk, elen, itos)
    out.update(illegal=ill.tolist(), legal_mass=lmass.tolist(), illegal_n=tot.tolist())

    torch.save([pr.state_dict() for pr in probes],
               os.path.join(RES, f"{args.name}_probes.pt"))
    tag = args.name + ("_zablate" if args.ablate_z else "")
    json.dump(out, open(os.path.join(RES, f"{tag}.json"), "w"), indent=1)
    print(f"[{tag}] H05={out['H05']} best_layer={best} "
          f"illegal@40-50={ill[4]:.4f}", flush=True)


if __name__ == "__main__":
    main()
