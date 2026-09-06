"""Load Karvonen's nanoGPT chess checkpoints as HF GPT-2 models.

The S4 deliverable is the differential scaling result on a ladder that is not
ours. Karvonen's 6, 8 and 16 layer lichess checkpoints are exactly that: data and
tokenization held fixed, only depth varying, which is what our own ladder cannot
provide because it varies with our data and our notation.

Those checkpoints are nanoGPT rather than HF, and the two disagree in a way that
fails silently. nanoGPT uses nn.Linear, whose weight is [out, in]. HF GPT-2 uses
Conv1D, whose weight is [in, out]. Every attention and MLP projection therefore
has to be transposed, and a missed transpose produces a model that loads cleanly,
runs without error and emits noise.

So conversion is not trusted until the converted model plays legal chess. That is
the same rule applied to the reconstructed vocabulary, and for the same reason:
the failure mode here is silent, so the check has to be behavioural.
"""
import os, json, argparse
import torch
import chess

ROOT = os.path.join(os.path.dirname(__file__), "..")
EXT = os.path.join(ROOT, "data", "ext")

TRANSPOSE_SUFFIXES = (".attn.c_attn.weight", ".attn.c_proj.weight",
                      ".mlp.c_fc.weight", ".mlp.c_proj.weight")


def strip_prefix(sd):
    """torch.compile stores parameters under _orig_mod."""
    out = {}
    for k, v in sd.items():
        for p in ("_orig_mod.", "module."):
            if k.startswith(p):
                k = k[len(p):]
        out[k] = v
    return out


def convert(ckpt_path, device="cuda"):
    from transformers import GPT2LMHeadModel, GPT2Config
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = strip_prefix(ck["model"] if "model" in ck else ck)
    ma = ck.get("model_args", {})

    n_layer = ma.get("n_layer") or (
        1 + max(int(k.split(".")[2]) for k in sd if k.startswith("transformer.h.")))
    n_embd = ma.get("n_embd") or sd["transformer.wte.weight"].shape[1]
    n_head = ma.get("n_head", max(1, n_embd // 64))
    vocab = ma.get("vocab_size") or sd["transformer.wte.weight"].shape[0]
    block = ma.get("block_size") or sd["transformer.wpe.weight"].shape[0]

    cfg = GPT2Config(vocab_size=vocab, n_positions=block, n_embd=n_embd,
                     n_layer=n_layer, n_head=n_head,
                     bos_token_id=None, eos_token_id=None)
    model = GPT2LMHeadModel(cfg)
    tgt = model.state_dict()

    new = {}
    for k, v in sd.items():
        if k == "lm_head.weight":
            continue                      # tied to wte
        if k.endswith(TRANSPOSE_SUFFIXES):
            v = v.t().contiguous()
        new[k] = v
    # nanoGPT is usually bias-free; HF expects biases, so fill the gaps with zeros
    filled = 0
    for k, v in tgt.items():
        if k not in new and k.endswith(".bias"):
            new[k] = torch.zeros_like(v)
            filled += 1
    missing, unexpected = model.load_state_dict(new, strict=False)
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, cfg, {"filled_biases": filled,
                        "missing": [m for m in missing if m != "lm_head.weight"],
                        "unexpected": list(unexpected)}


@torch.no_grad()
def free_play(model, plies=20, device="cuda"):
    """Behavioural proof the conversion is right."""
    from chessgpt import encode, ITOS, STOI
    ids = encode(";1.")
    b = chess.Board()
    legal = total = 0
    for ply in range(plies):
        chars = []
        for _ in range(8):
            x = torch.tensor([ids[-1000:]], dtype=torch.long, device=device)
            nxt = int(model(x).logits[0, -1].argmax())
            ch = ITOS.get(nxt, "")
            if ch in (" ", ";"):
                break
            chars.append(nxt)
            ids.append(nxt)
        san = "".join(ITOS.get(c, "") for c in chars)
        total += 1
        try:
            mv = b.parse_san(san)
            ok = mv in b.legal_moves
        except Exception:
            ok = False
        if not ok:
            return legal, total, san
        legal += 1
        b.push(mv)
        ids.append(STOI[" "])
        if ply % 2 == 1:
            ids += encode(f"{ply // 2 + 2}.")
    return legal, total, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", nargs="+", default=None)
    args = ap.parse_args()
    files = args.files or sorted(
        f for f in os.listdir(EXT) if f.endswith(".pt"))
    for f in files:
        p = os.path.join(EXT, f)
        if not os.path.exists(p):
            print(f"{f}: missing")
            continue
        try:
            model, cfg, info = convert(p)
        except Exception as e:
            print(f"{f}: CONVERSION FAILED {type(e).__name__}: {e}")
            continue
        npar = sum(q.numel() for q in model.parameters())
        legal, total, bad = free_play(model)
        verdict = ("CONVERSION VERIFIED" if legal == total
                   else f"REJECTED (first bad move {bad!r})")
        print(f"{f}")
        print(f"  {cfg.n_layer}L d={cfg.n_embd} heads={cfg.n_head} "
              f"vocab={cfg.vocab_size}  params {npar:,}")
        if info["missing"]:
            print(f"  missing: {info['missing'][:4]}")
        if info["unexpected"]:
            print(f"  unexpected: {info['unexpected'][:4]}")
        print(f"  zero-filled biases: {info['filled_biases']}")
        print(f"  free play: {legal}/{total} legal  -> {verdict}\n")


if __name__ == "__main__":
    main()
