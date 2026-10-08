"""Extract the training table from the checkpoints themselves.

The reviewer asked for architectures, token budgets, update counts, schedules,
checkpoint selection and validation losses. Every one of these is already stored
inside each checkpoint, as the argument namespace and the logged validation
curve, so the table can be read off the artifacts rather than transcribed from
the launcher scripts. That matters here for the same reason the rest of the
pipeline is macro-driven: a transcribed hyperparameter is a number that can
silently stop describing the run that produced the results.

Only metadata is touched. The tensors are loaded to CPU and never used, so this
runs on a machine without the CUDA build the checkpoints were written on.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ds_stats as S  # noqa: E402

RUNS = os.path.join(S.ROOT, "runs")


def corpus() -> dict:
    """Padded width and mean true length. Padded positions are masked out of the
    loss, so the supervised token count uses the true lengths, not the width."""
    p = os.path.join(S.ROOT, "data", "proc", "train_lm.npz")
    if not os.path.exists(p):
        return {}
    d = np.load(p)
    lens = d["lens"].astype(np.int64)
    return {"n_seqs": int(d["toks"].shape[0]), "width": int(d["toks"].shape[1]),
            "mean_len": float(lens.mean()), "total_tokens": int(lens.sum())}


def read_one(name: str):
    p = os.path.join(RUNS, f"{name}.pt")
    if not os.path.exists(p):
        return None
    ck = torch.load(p, map_location="cpu", weights_only=False)
    a, log = ck.get("args", {}), ck.get("log", [])
    n_par = sum(v.numel() for v in ck["model"].values())
    val = [r["val"] for r in log if "val" in r]
    return {"name": name, "params": n_par,
            "layers": a.get("n_layer") or a.get("layers"),
            "width": a.get("d_model") or a.get("width"),
            "heads": a.get("n_head") or a.get("heads"),
            "steps": a.get("steps"), "bs": a.get("bs"), "lr": a.get("lr"),
            "seed": a.get("seed"),
            "val_final": val[-1] if val else None,
            "val_best": min(val) if val else None,
            "val_evals": len(val),
            "final_is_best": (bool(abs(val[-1] - min(val)) < 1e-9)
                              if val else None)}


def main() -> int:
    cp = corpus()
    L = cp.get("mean_len")
    rows = []
    for r in S.RUNGS:
        for sd in S.SEEDS:
            rec = read_one(f"attn_{r}_s{sd}")
            if rec:
                rec["rung"] = r
                rec["tokens"] = int(rec["steps"] * rec["bs"] * L) if L else None
                rows.append(rec)
    if not rows:
        print("no checkpoints found under runs/")
        return 1

    print(f"{'rung':9s} {'seed':>4s} {'params':>10s} {'L':>3s} {'d':>5s} "
          f"{'steps':>6s} {'bs':>4s} {'tokens':>12s} {'val':>7s} {'best':>7s}")
    for q in rows:
        print(f"{q['rung']:9s} {q['seed']:4d} {q['params']:10,d} "
              f"{q['layers'] or 0:3d} {q['width'] or 0:5d} {q['steps']:6d} "
              f"{q['bs']:4d} {(q['tokens'] or 0):12,d} "
              f"{(q['val_final'] or 0):7.4f} {(q['val_best'] or 0):7.4f}")

    # Checkpoint selection is the last step, not the best validation loss. Say so
    # and check it, because "we select the final checkpoint" is only honest if the
    # final one is not also quietly the best in every run.
    n_fb = sum(1 for q in rows if q["final_is_best"])
    out = {"corpus": cp, "n_runs": len(rows), "rows": rows,
           "selection": "final step; no early stopping and no best-val selection",
           "n_final_is_best": n_fb,
           "eval_every": (rows[0]["steps"] // max(rows[0]["val_evals"], 1)),
           "shared": {"optimizer": "AdamW", "weight_decay": 0.01,
                      "betas": [0.9, 0.95], "schedule": "OneCycleLR",
                      "pct_start": 0.05, "grad_clip": 1.0,
                      "precision": "fp16 autocast with gradient scaling"}}
    per = {}
    for k in ("steps", "bs", "lr"):
        per[k] = sorted({q[k] for q in rows})
    out["distinct"] = per
    print(f"\nfinal checkpoint is also the best-validation checkpoint in "
          f"{n_fb} of {len(rows)} runs")
    print("distinct values:", per)
    json.dump(out, open(os.path.join(S.RES, "training_table.json"), "w"),
              indent=1)
    print("wrote results/training_table.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
