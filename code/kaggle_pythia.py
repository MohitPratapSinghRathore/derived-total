"""Kaggle notebook source: the 1B and larger rungs of the Pythia partition test.

Paste this into a single Kaggle notebook cell, or upload it as a script. It is
self-contained: it installs what it needs, streams the corpus, runs the rungs in
ascending order, and appends to results/pythia_partition.csv in the repository
after EVERY model, pushing as it goes, so a session that dies at 6.9B still leaves
everything below it banked.

Governed by PREREGISTERED.md. The token budget, the weights convention, the check 1
tolerance and the bootstrap are fixed there and must not be adjusted here.

BEFORE RUNNING
  1. Settings -> Accelerator -> GPU T4 x2.
  2. Settings -> Internet -> On.
  3. Add a Kaggle secret named GITHUB_TOKEN holding a fine-grained personal access
     token with Contents: write on the derived-total repository only. Nothing else
     in this notebook needs credentials, and the token must not be pasted inline.
  4. Run. Roughly 6 to 9 hours for 1b through 12b at the preregistered budget, well
     inside the 30 GPU-hours per week the free tier allows, but past the 12-hour
     per-session limit is a risk, which is why every rung pushes as it completes.

WHY IT PUSHES PER RUNG
  A Kaggle session can be evicted without warning. Holding results in memory until
  the end would mean losing a full ladder to a timeout at the last model. Appending
  and pushing per rung costs a few seconds and makes the run resumable: on restart
  the script reads the CSV it already pushed and skips what is there.
"""

SETUP = r"""
!pip -q install "transformers>=4.44" "datasets>=2.20" accelerate bitsandbytes
"""

import os
import subprocess
import sys

# --------------------------------------------------------------- constants
REPO = "MohitPratapSinghRathore/derived-total"
WORKDIR = "/kaggle/working/derived-total"
VAL_URL = "hf://datasets/monology/pile-uncopyrighted/val.jsonl.zst"
TOKEN_BUDGET = 200_000
MAX_LEN = 1024
CSV_REL = "results/pythia_partition.csv"

LADDER = [
    ("EleutherAI/pythia-1b-deduped", 1_011_781_632),
    ("EleutherAI/pythia-1.4b-deduped", 1_414_647_808),
    ("EleutherAI/pythia-2.8b-deduped", 2_775_208_960),
    ("EleutherAI/pythia-6.9b-deduped", 6_857_302_016),
    ("EleutherAI/pythia-12b-deduped", 11_846_072_320),
]

FIELDS = ["model", "params", "source", "sum_nll", "tokens", "n_docs",
          "device", "dtype", "timestamp"]


def sh(cmd, **kw):
    print("$", cmd)
    return subprocess.run(cmd, shell=True, check=False, text=True, **kw)


def clone():
    from kaggle_secrets import UserSecretsClient
    tok = UserSecretsClient().get_secret("GITHUB_TOKEN")
    if os.path.isdir(WORKDIR):
        sh(f"cd {WORKDIR} && git pull --quiet")
    else:
        sh(f"git clone --quiet https://x-access-token:{tok}@github.com/{REPO}.git "
           f"{WORKDIR}")
    sh(f"cd {WORKDIR} && git config user.name 'Sriharsha-Meduri' && "
       f"git config user.email 'sriharshameduri07@gmail.com'")
    return tok


def push(msg):
    sh(f"cd {WORKDIR} && git add {CSV_REL} && "
       f"git commit -q -m \"{msg}\" && git push -q origin HEAD:main || true")


def done_models():
    import csv
    p = os.path.join(WORKDIR, CSV_REL)
    if not os.path.exists(p):
        return set()
    with open(p, encoding="utf-8") as fh:
        return {r["model"] for r in csv.DictReader(fh)}


def collect(tok):
    """Tokenise once, shared by every rung, so the weights are constant."""
    from datasets import load_dataset
    ds = load_dataset("json", data_files=VAL_URL, split="train", streaming=True)
    per_source, scanned = {}, 0
    for rec in ds:
        scanned += 1
        src = (rec.get("meta") or {}).get("pile_set_name")
        if not src:
            continue
        b = per_source.setdefault(src, {"ids": [], "tokens": 0})
        if b["tokens"] >= TOKEN_BUDGET:
            if scanned > 20000 and all(
                    v["tokens"] >= TOKEN_BUDGET for v in per_source.values()):
                break
            continue
        ids = tok(rec["text"], truncation=True, max_length=MAX_LEN).input_ids
        if len(ids) < 2:
            continue
        b["ids"].append(ids)
        b["tokens"] += len(ids) - 1
    return per_source


def load_model(name):
    """fp16 on one T4 up to 6.9B; 12B sharded across both, int8 as the fallback.

    Which path was taken is recorded per row, because a quantised rung is not
    comparable to an fp16 one without the reader knowing.
    """
    import torch
    from transformers import AutoModelForCausalLM

    if "12b" not in name:
        m = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float16)
        return m.to("cuda"), "cuda", "float16"
    try:
        m = AutoModelForCausalLM.from_pretrained(
            name, dtype=torch.float16, device_map="auto",
            max_memory={0: "14GiB", 1: "14GiB"})
        return m, "cuda:auto", "float16"
    except Exception as e:                      # noqa: BLE001
        print(f"  sharded load failed ({type(e).__name__}: {e}); "
              f"falling back to int8 on one T4. This is a LIMITATION and is "
              f"recorded in the dtype column.")
        from transformers import BitsAndBytesConfig
        m = AutoModelForCausalLM.from_pretrained(
            name, quantization_config=BitsAndBytesConfig(load_in_8bit=True),
            device_map={"": 0})
        return m, "cuda", "int8"


def score(model, docs, device):
    import torch
    dev = "cuda" if device.startswith("cuda") else device
    total_nll, total_tok = 0.0, 0
    model.eval()
    with torch.no_grad():
        for ids in docs:
            t = torch.tensor([ids], device=dev)
            out = model(t, labels=t)
            n = t.shape[1] - 1
            total_nll += float(out.loss) * n
            total_tok += n
    return total_nll, total_tok


def main():
    import csv
    import time
    import torch
    from transformers import AutoTokenizer

    clone()
    skip = done_models()
    print("already banked:", sorted(skip) or "nothing")

    tok = AutoTokenizer.from_pretrained("EleutherAI/pythia-70m-deduped")
    print("collecting corpus")
    per_source = collect(tok)
    for s, v in sorted(per_source.items()):
        print(f"  {s:24s} {len(v['ids']):5d} docs {v['tokens']:8,d} tokens")

    csv_path = os.path.join(WORKDIR, CSV_REL)
    for name, params in LADDER:
        if name in skip:
            print(f"{name}: banked, skipping")
            continue
        print(f"\n=== {name}")
        t0 = time.time()
        try:
            model, device, dtype = load_model(name)
        except Exception as e:                  # noqa: BLE001
            print(f"  LOAD FAILED: {type(e).__name__}: {e}")
            print("  stopping here; everything below this rung is already pushed")
            break
        stamp = time.strftime("%Y-%m-%d %H:%M")
        rows = []
        for src, v in sorted(per_source.items()):
            nll, ntok = score(model, v["ids"], device)
            rows.append({"model": name, "params": params, "source": src,
                         "sum_nll": f"{nll:.6f}", "tokens": ntok,
                         "n_docs": len(v["ids"]), "device": device,
                         "dtype": dtype, "timestamp": stamp})
            print(f"  {src:24s} loss {nll / ntok:.4f}")
        new = not os.path.exists(csv_path)
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        with open(csv_path, "a", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=FIELDS)
            if new:
                w.writeheader()
            w.writerows(rows)
        push(f"Pythia partition: {name} ({dtype}, {device})")
        print(f"  pushed in {time.time() - t0:.0f}s")
        del model
        torch.cuda.empty_cache()

    print("\ndone. Analysis runs locally: python code/pythia_analyse.py")


if __name__ == "__main__":
    main()
