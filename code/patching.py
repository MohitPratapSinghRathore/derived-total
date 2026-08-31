"""Causal test: if we induce a false belief, does the model act on it?

The belief-consistency result is observational. It shows that when the model errs
it errs coherently with a decayed state, but not that the state representation
drives the action. This intervenes.

At one position we push the residual stream along the probe direction that
encodes "this square is empty" for a square the side to move actually occupies,
then ask whether the model stops proposing moves from that square.

Two things make or break this experiment, and both are reported.

  MANIPULATION CHECK. Did the edit actually change the decoded belief? An
  intervention that fails to move the probe read-out cannot tell us anything, so
  we report the fraction of cases where the decoded square really did flip to
  empty, and analyse the behavioural effect on that subset.

  NORM-MATCHED CONTROL. A random direction of identical norm, applied at the same
  position. If a random push moves the action as much as the state direction
  does, the effect is perturbation sensitivity rather than belief.
"""
import os, json, argparse
import numpy as np
import torch
import chess
from probes import load_split, build

DATA = os.path.join(os.path.dirname(__file__), "..", "data", "proc")
RUNS = os.path.join(os.path.dirname(__file__), "..", "runs")
RES = os.path.join(os.path.dirname(__file__), "..", "results")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--alpha", type=float, default=6.0,
                    help="edit strength in units of residual-stream RMS")
    args = ap.parse_args()

    model, a, c = build(os.path.join(RUNS, f"{args.name}.pt"))
    res = json.load(open(os.path.join(RES, f"{args.name}.json")))
    best = res["best_layer"]
    pls = torch.load(os.path.join(RES, f"{args.name}_probes.pt"), weights_only=False)
    probe = torch.nn.Linear(a["width"], 64 * 13).cuda()
    probe.load_state_dict(pls[best]); probe.eval()
    W = probe.weight.reshape(64, 13, a["width"])

    toks, lens, occ = load_split("eval", 1200)
    stoi = json.load(open(os.path.join(DATA, "vocab.json")))
    itos = {v: k for k, v in stoi.items()}
    rng = np.random.default_rng(0)

    rec = {"state": [], "random": []}
    flipped = 0
    tot = 0

    with torch.no_grad():
        for gi in range(len(toks)):
            if tot >= args.n:
                break
            T = int(lens[gi])
            if T < 25:
                continue
            t = int(rng.integers(12, min(T - 2, 70)))
            x = torch.from_numpy(toks[gi:gi + 1, :t + 1]).cuda()

            board = chess.Board()
            for u in range(1, t + 1):
                board.push_uci(itos[int(toks[gi, u])])
            srcs = sorted({m.from_square for m in board.legal_moves})
            if len(srcs) < 2:
                continue
            sq = int(rng.choice(srcs))
            legal_from_sq = [stoi[m.uci()] for m in board.legal_moves
                             if m.from_square == sq and m.uci() in stoi]
            if not legal_from_sq:
                continue

            def run(edit):
                store = {}

                def hook(mod, inp, out):
                    if edit is not None:
                        out = out.clone()
                        out[:, -1] = out[:, -1] + edit
                    store["h"] = out
                    return out
                hd = model.blocks[best].register_forward_hook(hook)
                with torch.amp.autocast("cuda", dtype=torch.float16):
                    lg, _, _ = model(x)
                hd.remove()
                h = store["h"][0, -1].float()
                decoded = probe(h).reshape(64, 13).argmax(-1)
                return torch.softmax(lg.float()[0, -1], -1), int(decoded[sq])

            p0, dec0 = run(None)
            # scale the edit relative to the residual stream's own magnitude
            with torch.amp.autocast("cuda", dtype=torch.float16):
                _, hs0, _ = model(x, return_hidden=True)
            rms = float(hs0[best][0, -1].float().pow(2).mean().sqrt())
            cur = int(occ[gi, t, sq])
            if cur == 0:
                continue
            d = (W[sq, 0] - W[sq, cur]).float()
            d = d / d.norm() * args.alpha * rms * np.sqrt(a["width"])
            p1, dec1 = run(d)
            # norm-matched random direction
            r = torch.randn_like(d); r = r / r.norm() * d.norm()
            p2, dec2 = run(r)

            m0 = float(p0[legal_from_sq].sum())
            m1 = float(p1[legal_from_sq].sum())
            m2 = float(p2[legal_from_sq].sum())
            did_flip = (dec1 == 0)
            flipped += bool(did_flip)
            tot += 1
            rec["state"].append((m0, m1, bool(did_flip)))
            rec["random"].append((m0, m2, bool(dec2 == 0)))

    st = np.array([(a_, b_) for a_, b_, _ in rec["state"]])
    fl = np.array([f for _, _, f in rec["state"]])
    rd = np.array([(a_, b_) for a_, b_, _ in rec["random"]])

    def summarise(arr, mask=None):
        if mask is not None:
            arr = arr[mask]
        if len(arr) == 0:
            return {"n": 0}
        drop = arr[:, 0] - arr[:, 1]
        rel = drop / np.maximum(arr[:, 0], 1e-8)
        return {"n": int(len(arr)),
                "mass_before": float(arr[:, 0].mean()),
                "mass_after": float(arr[:, 1].mean()),
                "mean_abs_drop": float(drop.mean()),
                "mean_rel_drop": float(rel.mean()),
                "frac_decreased": float((drop > 0).mean())}

    out = {"name": args.name, "layer": best, "alpha": args.alpha, "n": int(tot),
           "manipulation_check_flip_rate": float(flipped / max(tot, 1)),
           "state_edit_all": summarise(st),
           "state_edit_where_belief_flipped": summarise(st, fl),
           "state_edit_where_belief_unchanged": summarise(st, ~fl),
           "norm_matched_random_edit": summarise(rd)}
    # append to a calibration record so the choice of edit strength is auditable
    cal_p = os.path.join(RES, f"{args.name}_patch_sweep.json")
    cal = json.load(open(cal_p)) if os.path.exists(cal_p) else {}
    cal[str(args.alpha)] = {
        "flip_rate": out["manipulation_check_flip_rate"],
        "state_mass_before": out["state_edit_all"]["mass_before"],
        "state_mass_after": out["state_edit_all"]["mass_after"],
        "state_frac_decreased": out["state_edit_all"]["frac_decreased"],
        "random_mass_after": out["norm_matched_random_edit"]["mass_after"],
        "random_frac_decreased": out["norm_matched_random_edit"]["frac_decreased"],
        "n": out["n"]}
    json.dump(cal, open(cal_p, "w"), indent=1)
    json.dump(out, open(os.path.join(RES, f"{args.name}_patch.json"), "w"), indent=1)

    print(f"[{args.name}] causal patching at layer {best}, n={tot}")
    print(f"  manipulation check: decoded square flipped to empty in "
          f"{out['manipulation_check_flip_rate']:.3f} of cases")
    for k in ("state_edit_all", "state_edit_where_belief_flipped",
              "state_edit_where_belief_unchanged", "norm_matched_random_edit"):
        s = out[k]
        if s.get("n"):
            print(f"  {k:38s} n={s['n']:4d} mass {s['mass_before']:.3f} -> "
                  f"{s['mass_after']:.3f}  rel drop {s['mean_rel_drop']:+.3f}  "
                  f"decreased {s['frac_decreased']:.3f}")


if __name__ == "__main__":
    main()
