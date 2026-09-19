"""Per-source loss over the Pythia ladder, to test the consistency condition.

Governed by PREREGISTERED.md, written and committed before any model was loaded.
Nothing here may be changed in a way that alters a preregistered quantity: the
token budget, the check 1 tolerance, the weights convention, the breakdown
tolerance and the bootstrap are all fixed there.

The identity under test is the token-weighted MEAN of per-source loss, not a sum,
because loss is not an error rate. Everything else follows the paper unchanged: a
per-category curve per source, a separately fitted aggregate, and the question of
whether the two can both be right.

Output is appended to results/pythia_partition.csv after every model, one row per
model per source, carrying summed negative log likelihood and token counts rather
than means, so that any later re-weighting is possible without rerunning and so a
partial ladder is never silently averaged.
"""
import os, sys, csv, json, argparse, time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
RES = os.path.join(ROOT, "results")
CSV = os.path.join(RES, "pythia_partition.csv")
VAL_URL = "hf://datasets/monology/pile-uncopyrighted/val.jsonl.zst"

TOKEN_BUDGET = 200_000        # per source per model, preregistered
MAX_LEN = 1024                # per document, preregistered
CHECK1_TOL = 1e-6             # relative, preregistered

LADDER = {
    "EleutherAI/pythia-70m-deduped": 70_426_624,
    "EleutherAI/pythia-160m-deduped": 162_322_944,
    "EleutherAI/pythia-410m-deduped": 405_334_016,
    "EleutherAI/pythia-1b-deduped": 1_011_781_632,
    "EleutherAI/pythia-1.4b-deduped": 1_414_647_808,
    "EleutherAI/pythia-2.8b-deduped": 2_775_208_960,
    "EleutherAI/pythia-6.9b-deduped": 6_857_302_016,
    "EleutherAI/pythia-12b-deduped": 11_846_072_320,
}

FIELDS = ["model", "params", "source", "sum_nll", "tokens", "n_docs",
          "device", "dtype", "timestamp"]


# ----------------------------------------------------------------- corpus
def collect(tok, budget=TOKEN_BUDGET, max_len=MAX_LEN, limit_sources=None):
    """Stream the validation file once, tokenise, and hold documents per source
    until each source's token budget is met.

    Tokenising here rather than per model matters: all Pythia models share one
    tokenizer, so doing it once guarantees every rung sees exactly the same token
    sequences, which is what makes the weights constant and check 1b meaningful.
    """
    from datasets import load_dataset

    ds = load_dataset("json", data_files=VAL_URL, split="train", streaming=True)
    per_source, done = {}, set()
    scanned = 0
    for rec in ds:
        scanned += 1
        meta = rec.get("meta") or {}
        src = meta.get("pile_set_name")
        if not src:
            continue
        if limit_sources and src not in limit_sources:
            continue
        bucket = per_source.setdefault(src, {"ids": [], "tokens": 0})
        if bucket["tokens"] >= budget:
            done.add(src)
            # Stop when every source seen so far is satisfied and we have scanned
            # enough to be confident no new source is still to appear.
            if scanned > 20000 and done == set(per_source):
                break
            continue
        ids = tok(rec["text"], truncation=True, max_length=max_len).input_ids
        if len(ids) < 2:
            continue
        bucket["ids"].append(ids)
        bucket["tokens"] += len(ids) - 1     # predicted tokens
        if scanned % 5000 == 0:
            filled = sum(1 for v in per_source.values() if v["tokens"] >= budget)
            print(f"    scanned {scanned:,}; {filled}/{len(per_source)} sources full")
    return per_source, scanned


# ------------------------------------------------------------------ model
def score(model, docs, device):
    """Summed NLL and predicted-token count over one source's documents."""
    import torch

    total_nll, total_tok = 0.0, 0
    model.eval()
    with torch.no_grad():
        for ids in docs:
            t = torch.tensor([ids], device=device)
            out = model(t, labels=t)
            n = t.shape[1] - 1
            total_nll += float(out.loss) * n
            total_tok += n
    return total_nll, total_tok


def append_rows(rows):
    new = not os.path.exists(CSV)
    with open(CSV, "a", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        if new:
            w.writeheader()
        for r in rows:
            w.writerow(r)


def already_done(name):
    if not os.path.exists(CSV):
        return False
    with open(CSV, encoding="utf-8") as fh:
        return any(r["model"] == name for r in csv.DictReader(fh))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=list(LADDER)[:3])
    ap.add_argument("--budget", type=int, default=TOKEN_BUDGET)
    ap.add_argument("--dtype", default="auto")
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    tok = AutoTokenizer.from_pretrained("EleutherAI/pythia-70m-deduped")
    print("collecting corpus (once, shared by every rung)")
    t0 = time.time()
    per_source, scanned = collect(tok, budget=args.budget)
    print(f"  {len(per_source)} sources, {scanned:,} records scanned, "
          f"{time.time() - t0:.0f}s")
    for s, v in sorted(per_source.items()):
        print(f"    {s:24s} {len(v['ids']):5d} docs  {v['tokens']:8,d} tokens")

    for name in args.models:
        if already_done(name):
            print(f"{name}: already in the CSV, skipping")
            continue
        print(f"\n{name}")
        t0 = time.time()
        dt = torch.float16 if (device == "cuda" and args.dtype != "float32") \
            else torch.float32
        model = AutoModelForCausalLM.from_pretrained(name, dtype=dt).to(device)
        stamp = time.strftime("%Y-%m-%d %H:%M")
        rows = []
        for src, v in sorted(per_source.items()):
            nll, ntok = score(model, v["ids"], device)
            rows.append({"model": name, "params": LADDER[name], "source": src,
                         "sum_nll": f"{nll:.6f}", "tokens": ntok,
                         "n_docs": len(v["ids"]), "device": device,
                         "dtype": str(dt).replace("torch.", ""),
                         "timestamp": stamp})
            print(f"  {src:24s} loss {nll / ntok:.4f}  over {ntok:,} tokens")
        append_rows(rows)
        print(f"  appended {len(rows)} rows in {time.time() - t0:.0f}s")
        del model
        if device == "cuda":
            torch.cuda.empty_cache()

    print(f"\nCSV now at {CSV}")


if __name__ == "__main__":
    main()
