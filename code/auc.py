"""Aggregate versus action-relevant state error, compared like for like.

The paper contrasted a within-bucket point-biserial correlation (aggregate) with
a conditional probability (action-relevant). Those are different statistics and
the comparison was not fair. Here both predictors are scored the same way, as
predictors of the same binary outcome, with the same samples:

  AUC of each predictor for "the top-1 move is illegal", within depth bucket
  a logistic fit with both predictors together, giving standardised coefficients

Predictors:
  aggregate       number of misremembered squares on the whole board
  action-relevant whether the probe is wrong on either square the move touches
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
    """Rank-based AUC, ties handled by average rank."""
    s = np.asarray(scores, float); y = np.asarray(labels, int)
    n1, n0 = int(y.sum()), int((1 - y).sum())
    if n1 == 0 or n0 == 0:
        return None
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), float)
    ranks[order] = np.arange(1, len(s) + 1)
    # average ranks within ties
    uniq, inv, cnt = np.unique(s, return_inverse=True, return_counts=True)
    sums = np.zeros(len(uniq)); np.add.at(sums, inv, ranks)
    ranks = (sums / cnt)[inv]
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def logistic(X, y, iters=300, lr=0.1):
    """Small logistic fit on standardised predictors."""
    X = np.asarray(X, float)
    mu, sd = X.mean(0), X.std(0) + 1e-9
    Z = (X - mu) / sd
    Z = np.hstack([Z, np.ones((len(Z), 1))])
    w = np.zeros(Z.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-Z @ w))
        w -= lr * (Z.T @ (p - y)) / len(y)
    return w[:-1]


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

    rows = {bi: {"agg": [], "act": [], "y": []} for bi in range(len(BUCKETS))}
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
                            touched_wrong = int(not (ok[r, t, m_.from_square]
                                                     and ok[r, t, m_.to_square]))
                        except Exception:
                            m_, legal, touched_wrong = None, False, 1
                        if m_ is not None:
                            rows[bi]["agg"].append(int((~ok[r, t]).sum()))
                            rows[bi]["act"].append(touched_wrong)
                            rows[bi]["y"].append(0 if legal else 1)
                    board.push_uci(itos[int(toks[gi, t + 1])])

    out = {"name": args.name, "buckets": BUCKETS, "per_bucket": {}}
    print(f"[{args.name}] like-for-like: AUC for predicting an illegal top-1 move")
    print(f"{'ply':>8} {'AUC aggregate':>14} {'AUC action-rel':>15} "
          f"{'beta agg':>9} {'beta act':>9} {'n':>7}")
    for bi, (lo, hi) in enumerate(BUCKETS):
        d = rows[bi]
        y = np.array(d["y"])
        if len(y) < 200 or y.sum() == 0 or y.sum() == len(y):
            continue
        a_auc = auc(d["agg"], y)
        c_auc = auc(d["act"], y)
        beta = logistic(np.stack([d["agg"], d["act"]], 1), y)
        out["per_bucket"][f"{lo}-{hi}"] = {
            "auc_aggregate": a_auc, "auc_action_relevant": c_auc,
            "beta_aggregate": float(beta[0]), "beta_action_relevant": float(beta[1]),
            "n": int(len(y)), "illegal_rate": float(y.mean())}
        print(f"{str(lo)+'-'+str(hi):>8} {a_auc:14.3f} {c_auc:15.3f} "
              f"{beta[0]:9.3f} {beta[1]:9.3f} {len(y):7d}")

    # pooled, depth held fixed by including bucket index as a covariate
    allagg = np.concatenate([rows[b]["agg"] for b in rows])
    allact = np.concatenate([rows[b]["act"] for b in rows])
    ally = np.concatenate([rows[b]["y"] for b in rows])
    alldep = np.concatenate([[b] * len(rows[b]["y"]) for b in rows])
    beta = logistic(np.stack([allagg, allact, alldep], 1), ally)
    out["pooled_with_depth_covariate"] = {
        "beta_aggregate": float(beta[0]), "beta_action_relevant": float(beta[1]),
        "beta_depth": float(beta[2]), "n": int(len(ally))}
    print(f"\npooled, depth as covariate: beta_aggregate {beta[0]:+.3f}, "
          f"beta_action_relevant {beta[1]:+.3f}, beta_depth {beta[2]:+.3f}")
    json.dump(out, open(os.path.join(RES, f"{args.name}_auc.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
