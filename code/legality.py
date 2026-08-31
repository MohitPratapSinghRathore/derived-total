"""E_state observable: teacher-forced illegal-move rate against ply.

At every position of an eval game we take the model's argmax next move and ask a
rules engine whether it is legal in the true position.  An illegal top-1 move is
direct evidence the position was lost, independent of any probe.
"""
import os, json
import numpy as np
import torch
import chess

DATA = os.path.join(os.path.dirname(__file__), "..", "data", "proc")


@torch.no_grad()
def illegal_rate(model, toks, lens, itos, bs=64, buckets=None, max_games=4000):
    dev = "cuda"
    N = min(len(toks), max_games)
    if buckets is None:
        buckets = [(0, 10), (10, 20), (20, 30), (30, 40), (40, 50),
                   (50, 60), (60, 80), (80, 120), (120, 161)]
    bad = np.zeros(len(buckets)); tot = np.zeros(len(buckets))
    legal_mass = np.zeros(len(buckets))
    for i in range(0, N, bs):
        idx = np.arange(i, min(i + bs, N))
        x = torch.from_numpy(toks[idx]).to(dev)
        with torch.amp.autocast("cuda", dtype=torch.float16):
            logits, _, _ = model(x[:, :-1])
        logits = logits.float()
        pred = logits.argmax(-1).cpu().numpy()
        probs = torch.softmax(logits, -1).cpu().numpy()
        for r, gi in enumerate(idx):
            board = chess.Board()
            T = int(lens[gi])
            for t in range(T - 1):
                bi = next((k for k, (lo, hi) in enumerate(buckets) if lo <= t < hi), None)
                if bi is not None:
                    mv = itos.get(int(pred[r, t]), "")
                    ok = False
                    try:
                        ok = chess.Move.from_uci(mv) in board.legal_moves
                    except Exception:
                        ok = False
                    bad[bi] += (not ok)
                    tot[bi] += 1
                    # probability mass the model puts on legal continuations
                    lm = 0.0
                    for m_ in board.legal_moves:
                        j = STOI.get(m_.uci())
                        if j is not None:
                            lm += probs[r, t, j]
                    legal_mass[bi] += lm
                nxt = itos[int(toks[gi, t + 1])]
                board.push_uci(nxt)
    return bad / np.maximum(tot, 1), legal_mass / np.maximum(tot, 1), tot, buckets


STOI = json.load(open(os.path.join(DATA, "vocab.json")))
