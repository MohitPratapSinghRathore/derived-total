"""Test the consistency condition on a public ladder, in a non-chess domain.

STATUS: WRITTEN BUT NOT RUN. It has never been executed, so treat it as a
specification rather than as a verified tool, and do not cite anything from it
until it has been run and its output checked. The machine this was written on has
no CUDA build of torch and no `datasets` installed. Requirements are at the bottom.

Why this ladder. The paper's empirical base is 18 chess models between 3.4M and
39.9M parameters, a 12-fold span, used to discuss forecasts read far beyond it. That
is the weakest part of the argument. Pythia offers checkpoints from 70M to 12B
trained on the Pile, a roughly 170-fold span reaching the scales we extrapolate to,
and the Pile is a partition into named sources that we did not define. Per-source
loss over that ladder gives exactly the configuration the paper is about: a
per-category curve per source, and an aggregate.

One adjustment. Loss is not an error rate, so the identity being tested is not a
sum. The aggregate loss is the token-weighted MEAN of the per-source losses,

    L(N) = sum_s w_s L_s(N),     w_s = (tokens from source s) / (total tokens),

with the weights fixed by the evaluation mixture rather than by the model. That is
still a linear identity in the parts, so everything in the paper goes through with
the sum replaced by a weighted sum: fitting each L_s as a power law and L as a
separate power law over-determines the same relation, and the largest per-source
exponent still bounds the aggregate's asymptotic slope from above. The weights must
be reported, because unlike the error-rate case they are a choice.

What to check when it runs:
  1. Do the weighted per-source losses reproduce the measured aggregate in range?
     If not, the weights are wrong and nothing downstream means anything.
  2. Does the fitted aggregate exponent differ from the largest per-source exponent?
  3. Is there a breakdown scale INSIDE Pythia's own observed range? If so, that is
     the paper's headline and it no longer rests on chess.
  4. Is the aggregate detectably curved in log-log, as it is on our ladder?
"""
import os, json, sys, argparse

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

# Pythia's deduplicated suite. Sizes are the published parameter counts.
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


def per_source_loss(model, tok, texts, device, max_len=1024):
    """Mean token-level cross entropy and the token count, for one source.

    Returns both because the aggregate is a token-weighted mean: summing losses
    without their token counts would silently weight a short source equally with
    a long one and break the identity the whole exercise is testing.
    """
    import torch

    total_nll, total_tok = 0.0, 0
    model.eval()
    with torch.no_grad():
        for t in texts:
            ids = tok(t, return_tensors="pt", truncation=True,
                      max_length=max_len).input_ids.to(device)
            if ids.shape[1] < 2:
                continue
            out = model(ids, labels=ids)
            n = ids.shape[1] - 1           # labels are shifted internally
            total_nll += float(out.loss) * n
            total_tok += n
    return (total_nll / total_tok if total_tok else float("nan")), total_tok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=list(LADDER)[:5])
    ap.add_argument("--docs-per-source", type=int, default=200)
    ap.add_argument("--out", default="results/pythia_partition.json")
    a = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from datasets import load_dataset

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("WARNING: no GPU. This will be slow and the larger rungs are "
              "impractical. Results from a truncated ladder are not a "
              "substitute for the full one.")

    # Per-source validation text. The original Pile is no longer distributed in
    # full; the uncopyrighted mirror keeps the source labels, which is what the
    # partition needs. Any mirror is acceptable provided every document carries
    # its source and the mirror is named in the output.
    ds = load_dataset("monology/pile-uncopyrighted", split="validation",
                      streaming=True)
    by_source = {}
    for rec in ds:
        src = rec.get("meta", {}).get("pile_set_name")
        if not src:
            continue
        bucket = by_source.setdefault(src, [])
        if len(bucket) < a.docs_per_source:
            bucket.append(rec["text"])
        if all(len(v) >= a.docs_per_source for v in by_source.values()) \
                and len(by_source) > 10:
            break
    print(f"{len(by_source)} sources, "
          f"{sum(len(v) for v in by_source.values())} documents")

    out = {"ladder": {}, "sources": sorted(by_source),
           "docs_per_source": a.docs_per_source,
           "corpus": "monology/pile-uncopyrighted validation split"}

    for name in a.models:
        print(f"loading {name}")
        tok = AutoTokenizer.from_pretrained(name)
        model = AutoModelForCausalLM.from_pretrained(
            name, torch_dtype=torch.float16 if device == "cuda" else torch.float32
        ).to(device)
        row = {"params": LADDER[name], "per_source": {}}
        for src, texts in sorted(by_source.items()):
            loss, ntok = per_source_loss(model, tok, texts, device)
            row["per_source"][src] = {"loss": loss, "tokens": ntok}
            print(f"  {src:24s} loss {loss:.4f} over {ntok:,} tokens")
        tot_tok = sum(v["tokens"] for v in row["per_source"].values())
        row["aggregate_loss"] = sum(
            v["loss"] * v["tokens"] for v in row["per_source"].values()) / tot_tok
        row["weights"] = {s: v["tokens"] / tot_tok
                          for s, v in row["per_source"].items()}
        out["ladder"][name] = row
        del model
        if device == "cuda":
            torch.cuda.empty_cache()

    # Now the actual test, reusing the paper's own checker.
    import closure_check as C
    names = [n for n in a.models if n in out["ladder"]]
    N = [out["ladder"][n]["params"] for n in names]
    srcs = out["sources"]
    parts = {s: [out["ladder"][n]["per_source"][s]["loss"] for n in names]
             for s in srcs}
    # weighted parts, so the identity is sum_s w_s L_s = L
    w = out["ladder"][names[-1]]["weights"]
    weighted = {s: [w[s] * v for v in vals] for s, vals in parts.items()}
    total = [out["ladder"][n]["aggregate_loss"] for n in names]

    fitted = C.from_points(N, weighted, total)
    rep = C.check(**fitted, report_at=(max(N), 1e11))
    out["check"] = dict(rep)
    print()
    print(rep)

    json.dump(out, open(os.path.join(HERE, "..", a.out), "w"), indent=1)


if __name__ == "__main__":
    main()

# Requirements, none of which are satisfied on the machine this was written on:
#   pip install "torch>=2.4" --index-url https://download.pytorch.org/whl/cu121
#   pip install transformers datasets accelerate
#   a GPU with at least 24 GB to reach the 6.9B rung in fp16, or CPU and patience
#   roughly 60 GB of disk for the checkpoints through 12B
# Run the small rungs first and confirm check 1 above before spending on the rest.
