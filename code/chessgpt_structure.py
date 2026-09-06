"""Failure taxonomy and absolute rates for Chess-GPT, for S4.

A model that is not ours, with a different tokenization, different notation and
different training data. The taxonomy cannot transfer unchanged, for a reason
worth stating rather than patching around.

SAN does not name a source square. UCI e2e4 parses to a Move whatever the
position, so from_empty -- the class that scales away fastest in our ladder -- is
directly checkable. SAN names a piece type and a destination and leaves the
source implicit, so that class has no analogue here: the model cannot claim a
piece stands on an empty square, because it never names the square. The notation
structurally prevents our most local failure mode.

Classes come from san_classify, which preserves the distinction the argument
rests on: unreachable is local, determined by the piece and its path;
leaves_check is global, determined by the whole board.

Note on comparability. Chess-GPT is trained on far more data than our ladder and
its illegal-move rate is roughly an order of magnitude lower, so it does not sit
on our scaling curve. Comparing it to our models confounds tokenization with
training-data scale. Testing the differential scaling result needs Karvonen's own
6, 8 and 16 layer ladder, where data is held fixed and only size varies.

Generation is greedy with a KV cache forked per site, so the game is encoded once
and each site continues from the cached prefix.
"""
import os, json, argparse, time
import numpy as np
import torch
import chess
from chessgpt import load, encode, ITOS, STOI
from chessgpt_data import build, load_games
from san_classify import classify_san, CLASSES as SAN_CLASSES

RES = os.path.join(os.path.dirname(__file__), "..", "results")
CLASSES = SAN_CLASSES
STRATA = [(20, 40), (40, 60), (60, 120)]


def boot(labels, games, keys, n_boot=500, seed=0):
    rng = np.random.default_rng(seed)
    labels, games = np.asarray(labels), np.asarray(games)
    u = np.unique(games)
    by = {g: np.where(games == g)[0] for g in u}
    acc = {k: [] for k in keys}
    for _ in range(n_boot):
        sel = np.concatenate([by[g] for g in rng.choice(u, len(u), replace=True)])
        l = labels[sel]
        for k in keys:
            acc[k].append(float((l == k).mean()))
    return {k: (float(np.mean(acc[k])), float(np.percentile(acc[k], 2.5)),
                float(np.percentile(acc[k], 97.5))) for k in keys}


@torch.no_grad()
def decode_at(model, ids, upto, max_chars=8, device="cuda"):
    """Greedy-decode a move continuing from ids[:upto+1], using a KV cache."""
    ctx = torch.tensor([ids[:upto + 1][-1000:]], dtype=torch.long, device=device)
    out = model(ctx, use_cache=True)
    past = out.past_key_values
    nxt = int(out.logits[0, -1].argmax())
    chars = []
    for _ in range(max_chars):
        ch = ITOS.get(nxt, "")
        if ch in (" ", ";"):
            break
        chars.append(ch)
        step = model(torch.tensor([[nxt]], dtype=torch.long, device=device),
                     past_key_values=past, use_cache=True)
        past = step.past_key_values
        nxt = int(step.logits[0, -1].argmax())
    return "".join(chars)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=120)
    ap.add_argument("--minply", type=int, default=20)
    ap.add_argument("--maxply", type=int, default=120)
    ap.add_argument("--tag", default="chessgpt2")
    args = ap.parse_args()

    model, cfg, _ = load(args.tag if args.tag != "chessgpt2" else "chessgpt2")
    games = load_games("eval", args.games)
    labels, gids, plies = [], [], []
    n_pos = 0
    t0 = time.time()

    for gi, g in enumerate(games):
        text, sites = build(g, max_plies=args.maxply)
        ids = encode(text)
        for s in sites:
            if s["ply"] < args.minply:
                continue
            san = decode_at(model, ids, s["char_index"])
            b = chess.Board(s["fen"])
            n_pos += 1
            try:
                legal = b.parse_san(san) in b.legal_moves
            except Exception:
                legal = False
            if legal:
                continue
            labels.append(classify_san(b, san))
            gids.append(gi)
            plies.append(s["ply"])
        if gi % 20 == 0:
            print(f"  game {gi}/{len(games)}  positions {n_pos:,}  "
                  f"failures {len(labels):,}  ({time.time()-t0:.0f}s)",
                  flush=True)

    labels = np.array(labels)
    gids = np.array(gids)
    plies = np.array(plies)
    ill = len(labels) / max(n_pos, 1)
    sh = boot(labels, gids, CLASSES)

    out = {"model": "Karvonen 8L ChessGPT2", "params": 25_760_256,
           "n_positions": int(n_pos), "n_failures": int(len(labels)),
           "n_games": int(len(np.unique(gids))), "illegal_rate": float(ill),
           "shares": {k: {"mean": sh[k][0], "ci_lo": sh[k][1],
                          "ci_hi": sh[k][2]} for k in CLASSES},
           "absolute": {k: sh[k][0] * ill for k in CLASSES}}

    print(f"\n[Chess-GPT 8L, 25.8M] {n_pos:,} positions from "
          f"{len(np.unique(gids))} games")
    print(f"  illegal-move rate: {ill:.4f}\n")
    print(f"{'class':>16} {'share':>8}  {'95% CI':>20}  {'absolute':>10}")
    for k in CLASSES:
        m, lo, hi = sh[k]
        print(f"{k:>16} {m:8.4f}  [{lo:.4f}, {hi:.4f}]  {m*ill:10.4f}")

    print(f"\nby depth:")
    print(f"{'ply':>9} {'n':>7} " + " ".join(f"{c[:11]:>12}" for c in CLASSES))
    for lo_, hi_ in STRATA:
        m = (plies >= lo_) & (plies < hi_)
        if m.sum() < 20:
            continue
        out.setdefault("strata", {})[f"{lo_}-{hi_}"] = {
            "n": int(m.sum()),
            **{c: float((labels[m] == c).mean()) for c in CLASSES}}
        print(f"{str(lo_)+'-'+str(hi_):>9} {m.sum():7,} "
              + " ".join(f"{float((labels[m]==c).mean()):12.4f}"
                         for c in CLASSES))

    json.dump(out, open(os.path.join(RES, "chessgpt_structure.json"), "w"),
              indent=1)
    print(f"\nours at comparable scale (12L384, 22.8M): illegal 0.1453, "
          f"leaves_check share 0.4146")


if __name__ == "__main__":
    main()
