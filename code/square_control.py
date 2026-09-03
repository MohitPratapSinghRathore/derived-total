"""Is the touched-square advantage real, or an artifact of counting two squares?

Move-touched fidelity scores two squares; whole-board fidelity averages
sixty-four. A statistic over two concentrated variables can look sharper than one
over sixty-four diluted ones for reasons having nothing to do with relevance.
This runs the control that separates the two explanations.

Predictors of "the top-1 move is illegal", all scored identically within depth:

  whole board          probe error over all 64 squares
  two random squares   probe error over two squares drawn at random
  two random occupied  two squares drawn at random from those actually occupied,
                       which matches the occupancy profile of source and
                       destination more closely
  source only          the square the move leaves
  destination only     the square the move enters
  move-touched         source and destination together

If move-touched beats both random-pair controls, the advantage is about which
squares, not how many. The source and destination split also says which half of
the pair carries the signal.
"""
import os, json, argparse
import numpy as np
import torch
import chess
from probes import load_split, build

DATA = os.path.join(os.path.dirname(__file__), "..", "data", "proc")
RUNS = os.path.join(os.path.dirname(__file__), "..", "runs")
RES = os.path.join(os.path.dirname(__file__), "..", "results")
BUCKETS = [(10, 20), (20, 30), (30, 40), (40, 50), (50, 60), (60, 80)]


def auc(scores, labels):
    s = np.asarray(scores, float); y = np.asarray(labels, int)
    n1, n0 = int(y.sum()), int((1 - y).sum())
    if n1 == 0 or n0 == 0:
        return None
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), float); ranks[order] = np.arange(1, len(s) + 1)
    uniq, inv, cnt = np.unique(s, return_inverse=True, return_counts=True)
    sums = np.zeros(len(uniq)); np.add.at(sums, inv, ranks)
    ranks = (sums / cnt)[inv]
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def boot_ci(scores, labels, games, n_boot=300, seed=0):
    """Bootstrap over GAMES, not positions: positions within a game are not
    independent, so resampling positions would understate the interval."""
    rng = np.random.default_rng(seed)
    scores = np.asarray(scores, float); labels = np.asarray(labels, int)
    games = np.asarray(games)
    uniq = np.unique(games)
    idx_by_game = {g: np.where(games == g)[0] for g in uniq}
    out = []
    for _ in range(n_boot):
        pick = rng.choice(uniq, len(uniq), replace=True)
        sel = np.concatenate([idx_by_game[g] for g in pick])
        a = auc(scores[sel], labels[sel])
        if a is not None:
            out.append(a)
    if not out:
        return None, None
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="attn_8L256_s0")
    ap.add_argument("--games", type=int, default=2500)
    args = ap.parse_args()

    model, a, c = build(os.path.join(RUNS, f"{args.name}.pt"))
    res = json.load(open(os.path.join(RES, f"{args.name}.json")))
    best = res["best_layer"]
    pls = torch.load(os.path.join(RES, f"{args.name}_probes.pt"), weights_only=False)
    probe = torch.nn.Linear(a["width"], 64 * 13).cuda()
    probe.load_state_dict(pls[best]); probe.eval()

    toks, lens, occ = load_split("eval", args.games)
    stoi = json.load(open(os.path.join(DATA, "vocab.json")))
    itos = {v: k for k, v in stoi.items()}
    rng = np.random.default_rng(0)

    keys = ["whole_board", "two_random", "two_random_occupied", "source_only",
            "destination_only", "move_touched"]
    D = {k: [] for k in keys}
    D["y"] = []; D["game"] = []; D["bucket"] = []

    with torch.no_grad():
        for i in range(0, len(toks), 32):
            idx = np.arange(i, min(i + 32, len(toks)))
            x = torch.from_numpy(toks[idx]).cuda()
            s = torch.from_numpy(np.asarray(occ[idx]).astype(np.int64)).cuda()
            with torch.amp.autocast("cuda", dtype=torch.float16):
                logits, hs, _ = model(x[:, :-1], return_hidden=True)
            pred = logits.float().argmax(-1).cpu().numpy()
            dec = probe(hs[best].float()).reshape(len(idx), -1, 64, 13).argmax(-1)
            ok = (dec == s[:, :dec.shape[1]]).cpu().numpy()
            occ_np = np.asarray(occ[idx]).astype(np.int64)
            for r, gi in enumerate(idx):
                board = chess.Board()
                T = int(lens[gi])
                for t in range(min(T - 1, ok.shape[1])):
                    bi = next((k for k, (lo, hi) in enumerate(BUCKETS)
                               if lo <= t < hi), None)
                    if bi is not None:
                        mv = itos.get(int(pred[r, t]), "")
                        try:
                            m_ = chess.Move.from_uci(mv)
                            legal = m_ in board.legal_moves
                        except Exception:
                            m_, legal = None, False
                        if m_ is not None:
                            row = ok[r, t]
                            occupied = np.nonzero(occ_np[r, t] > 0)[0]
                            rs = rng.choice(64, 2, replace=False)
                            ro = (rng.choice(occupied, 2, replace=False)
                                  if len(occupied) >= 2 else rs)
                            D["whole_board"].append(int((~row).sum()))
                            D["two_random"].append(int((~row[rs]).sum()))
                            D["two_random_occupied"].append(int((~row[ro]).sum()))
                            D["source_only"].append(int(not row[m_.from_square]))
                            D["destination_only"].append(int(not row[m_.to_square]))
                            D["move_touched"].append(
                                int(not (row[m_.from_square] and row[m_.to_square])))
                            D["y"].append(0 if legal else 1)
                            D["game"].append(int(gi))
                            D["bucket"].append(bi)
                    board.push_uci(itos[int(toks[gi, t + 1])])

    y = np.array(D["y"]); g = np.array(D["game"]); b = np.array(D["bucket"])
    out = {"name": args.name, "n": int(len(y)), "buckets": BUCKETS,
           "pooled": {}, "per_bucket": {}}
    print(f"[{args.name}] n={len(y)}  AUC for predicting an illegal move, "
          f"with game-clustered 95% CI")
    for k in keys:
        sc = np.array(D[k], float)
        a_ = auc(sc, y)
        lo, hi = boot_ci(sc, y, g)
        out["pooled"][k] = {"auc": a_, "ci_lo": lo, "ci_hi": hi}
        print(f"  {k:22s} {a_:.3f}  [{lo:.3f}, {hi:.3f}]")
    for bi, (lo_, hi_) in enumerate(BUCKETS):
        m = b == bi
        if m.sum() < 200:
            continue
        out["per_bucket"][f"{lo_}-{hi_}"] = {
            k: auc(np.array(D[k], float)[m], y[m]) for k in keys}
    json.dump(out, open(os.path.join(RES, f"{args.name}_sqcontrol.json"), "w"),
              indent=1)


if __name__ == "__main__":
    main()
