"""Adapter for Karvonen's character-level Chess-GPT, for S4 external validity.

Our own models emit one token per UCI move, so a probe reads exactly one residual
stream per move and the model's chosen move is an argmax over a move vocabulary.
Chess-GPT is character-level SAN PGN, and none of that holds:

  a move spans several character tokens, so board state has to be aligned to move
  boundaries inside the character stream rather than to token positions;

  the model's move is not an argmax, it is a short greedy decode that then has to
  parse as SAN before legality can even be asked;

  the edit site is the final character of a completed move rather than the last
  position, which has to be verified rather than assumed.

The HF repository ships only config.json and model.safetensors, with no
tokenizer, so the vocabulary is reconstructed. vocab_size is 32 and the SAN
character set has exactly 32 members, which is a tight constraint but not proof.
sanity_check below is the proof: if the mapping were wrong the model would emit
noise rather than legal continuations, so it must play a legal game before any
measurement built on it is trusted.
"""
import os, json
import numpy as np
import torch
import chess
import chess.pgn

ROOT = os.path.join(os.path.dirname(__file__), "..")
EXT = os.path.join(ROOT, "data", "ext")

# Reconstructed: exactly the 32 characters SAN PGN needs, sorted, which is how
# nanoGPT character vocabularies are built.
CHARS = sorted(list(" #+-.0123456789;=BKNOQRabcdefghx"))
STOI = {c: i for i, c in enumerate(CHARS)}
ITOS = {i: c for c, i in STOI.items()}


def encode(s):
    return [STOI[c] for c in s if c in STOI]


def decode(ids):
    return "".join(ITOS.get(int(i), "") for i in ids)


def load(name="chessgpt2", device="cuda"):
    from transformers import GPT2LMHeadModel, GPT2Config
    cfg = GPT2Config(**json.load(open(os.path.join(EXT, f"{name}_config.json"))))
    m = GPT2LMHeadModel(cfg)
    from safetensors.torch import load_file
    sd = load_file(os.path.join(EXT, f"{name}_model.safetensors"))
    missing, unexpected = m.load_state_dict(sd, strict=False)
    m.to(device).eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m, cfg, (missing, unexpected)


@torch.no_grad()
def greedy_move(model, prefix_ids, max_chars=8, device="cuda"):
    """Decode characters until the move terminates at a space.

    Returns the SAN string without its trailing space, plus the ids emitted, so
    a caller can locate the edit site at the final character of the move.
    """
    ids = list(prefix_ids)
    out = []
    for _ in range(max_chars):
        x = torch.tensor([ids[-1023:]], dtype=torch.long, device=device)
        lg = model(x).logits[0, -1]
        nxt = int(lg.argmax())
        ch = ITOS.get(nxt, "")
        if ch == " ":
            break
        out.append(nxt)
        ids.append(nxt)
        if len(out) >= max_chars:
            break
    return "".join(ITOS.get(i, "") for i in out), out


def game_to_pgn_string(uci_moves):
    """Render a UCI move list as the SAN PGN text Chess-GPT was trained on."""
    b = chess.Board()
    parts = [";"]
    for i, u in enumerate(uci_moves):
        try:
            mv = chess.Move.from_uci(u)
            if mv not in b.legal_moves:
                break
            san = b.san(mv)
        except Exception:
            break
        if i % 2 == 0:
            parts.append(f"{i//2 + 1}.")
        parts.append(san + " ")
        b.push(mv)
    return "".join(parts)


@torch.no_grad()
def sanity_check(model, n=3, device="cuda"):
    """The model must play legal chess before anything measured on it counts.

    A wrong vocabulary reconstruction produces noise, so this is the test that
    the reconstruction is right, not a nicety.
    """
    openings = [";1.e4 e5 2.", ";1.d4 d5 2.", ";1.e4 c5 2."]
    results = []
    for op in openings[:n]:
        ids = encode(op)
        san, _ = greedy_move(model, ids)
        b = chess.Board()
        # replay the opening so legality is asked of the right position
        txt = op[1:]
        for tok in txt.replace(".", " ").split():
            if tok.isdigit():
                continue
            try:
                b.push_san(tok)
            except Exception:
                pass
        ok = False
        try:
            mv = b.parse_san(san)
            ok = mv in b.legal_moves
        except Exception:
            ok = False
        results.append((op, san, ok))
    return results


def main():
    model, cfg, (missing, unexpected) = load()
    npar = sum(p.numel() for p in model.parameters())
    print(f"Chess-GPT loaded: {cfg.n_layer}L d={cfg.n_embd} "
          f"heads={cfg.n_head} vocab={cfg.vocab_size}")
    print(f"  parameters: {npar:,}")
    if missing:
        print(f"  MISSING keys: {list(missing)[:6]}")
    if unexpected:
        print(f"  unexpected keys: {list(unexpected)[:6]}")
    print(f"  vocab reconstructed as: {''.join(CHARS)!r}")

    print("\nsanity check -- must play legal moves or the vocabulary is wrong")
    res = sanity_check(model)
    for op, san, ok in res:
        print(f"  {op!r:16} -> {san!r:8} legal={ok}")
    n_ok = sum(1 for _, _, ok in res if ok)
    print(f"\n  {n_ok}/{len(res)} legal. "
          + ("vocabulary reconstruction CONFIRMED"
             if n_ok == len(res) else
             "REJECTED -- do not build measurements on this"))

    # longer free play, a stronger check than three openings
    print("\nfree play from the empty game, 20 plies:")
    ids = encode(";1.")
    b = chess.Board()
    legal = total = 0
    for ply in range(20):
        san, out = greedy_move(model, ids)
        try:
            mv = b.parse_san(san)
            good = mv in b.legal_moves
        except Exception:
            good = False
        total += 1
        if good:
            legal += 1
            b.push(mv)
            ids += out + [STOI[" "]]
            if ply % 2 == 1:
                ids += encode(f"{ply//2 + 2}.")
        else:
            print(f"  ply {ply}: {san!r} ILLEGAL, stopping")
            break
    print(f"  {legal}/{total} legal moves in free play")


if __name__ == "__main__":
    main()
