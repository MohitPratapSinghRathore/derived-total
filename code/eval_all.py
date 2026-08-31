"""Stage 5: full diagnostic for one checkpoint -> results/<name>.json

Horizon reporting.  The pre-registered H_eps was defined on exact-position
fidelity.  On these models that metric hits 0 by ply ~20 for EVERY condition, so
it cannot discriminate between them.  We therefore report three horizons and
apply all three identically to every condition:

  H_exact(eps)  first ply bucket where exact-position fidelity < 1-eps  (as pre-registered;
                retained even though it saturates, so the degeneracy is visible)
  H_occ(eps)    first ply bucket where occupied-square state accuracy < 1-eps
  H_ill(eps)    first ply bucket where the top-1 illegal-move rate exceeds eps
                (behavioural, monotone, and the direct observable for E_state)

This change was made after seeing baseline data.  It is a fix for metric
degeneracy, not a selection among outcomes: the definitions are fixed before the
ALSB conditions are evaluated and are applied unchanged to all of them.
"""
import os, sys, json, argparse
import numpy as np
import torch
from probes import (load_split, build, fit_probes, evaluate, majority_baseline)
from legality import illegal_rate

DATA = os.path.join(os.path.dirname(__file__), "..", "data", "proc")
RUNS = os.path.join(os.path.dirname(__file__), "..", "runs")
RES = os.path.join(os.path.dirname(__file__), "..", "results")


def first_below(curve, buckets, thr):
    for (lo, hi), v in zip(buckets, curve):
        if v < thr:
            return lo
    return buckets[-1][1]


def first_above(curve, buckets, thr):
    for (lo, hi), v in zip(buckets, curve):
        if v > thr:
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
    r = evaluate(model, a, probes, etk, elen, eocc)
    buckets = r["buckets"]
    best = int(np.argmax(r["occ"].mean(1)))     # layer with most decodable state
    out.update(buckets=buckets, counts=r["cnt"].tolist(),
               fidelity_exact=r["exact"].tolist(), fidelity_sq=r["sq"].tolist(),
               fidelity_occ=r["occ"].tolist(), nwrong=r["nwrong"].tolist(),
               best_layer=best)

    mb_sq, mb_occ = majority_baseline(ptk, plen, pocc, buckets)
    out["majority_sq"] = mb_sq.tolist()
    out["majority_occ"] = mb_occ.tolist()

    print(f"[{args.name}] illegal-move rate", flush=True)
    ill, lmass, tot, _ = illegal_rate(model, etk, elen, itos)
    out.update(illegal=ill.tolist(), legal_mass=lmass.tolist(), illegal_n=tot.tolist())

    out["H_exact_05"] = first_below(r["exact"][best], buckets, 0.95)
    out["H_occ_05"] = first_below(r["occ"][best], buckets, 0.95)
    out["H_occ_20"] = first_below(r["occ"][best], buckets, 0.80)
    out["H_ill_05"] = first_above(np.array(ill), buckets, 0.05)
    out["H_ill_10"] = first_above(np.array(ill), buckets, 0.10)

    if args.controls:
        print(f"[{args.name}] control: cross-game label pairing", flush=True)
        cp = fit_probes(model, a, ptk, plen, pocc, epochs=args.epochs, control=True)
        rc = evaluate(model, a, cp, etk, elen, eocc, control=True)
        out["control_shuffled_occ"] = rc["occ"].tolist()
        out["control_shuffled_sq"] = rc["sq"].tolist()

        print(f"[{args.name}] control: randomised-weight model", flush=True)
        rm, _, _ = build(ck, randomised=True)
        rp = fit_probes(rm, a, ptk, plen, pocc, epochs=args.epochs)
        rr = evaluate(rm, a, rp, etk, elen, eocc)
        out["control_random_occ"] = rr["occ"].tolist()
        out["control_random_sq"] = rr["sq"].tolist()
        out["control_random_exact"] = rr["exact"].tolist()

    torch.save([pr.state_dict() for pr in probes],
               os.path.join(RES, f"{args.name}_probes.pt"))
    tag = args.name + ("_zablate" if args.ablate_z else "")
    json.dump(out, open(os.path.join(RES, f"{tag}.json"), "w"), indent=1)
    b4 = 4
    print(f"[{tag}] best_layer={best} H_ill_05={out['H_ill_05']} "
          f"occ@40-50={r['occ'][best][b4]:.4f} (maj {mb_occ[b4]:.4f}) "
          f"illegal@40-50={ill[b4]:.4f}", flush=True)


if __name__ == "__main__":
    main()
