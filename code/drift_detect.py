"""If the state cannot be repaired, can its inconsistency at least be detected?

Repair fails for a principled reason. The model at position t already attends to
the whole prefix, so its own earlier state is a lossy function of the same
tokens: rolling that state forward cannot supply information the model does not
already have. Context rot here is not a failure to retain information but a
failure to compute with it, and a small linear operator does not compute better.

Detection is a different question, and a more useful one for deployment. If the
representation has drifted, the model's current state and the state implied by
rolling its earlier state forward should disagree. That disagreement is
computable at inference from activations and tokens alone, with no oracle, and
Paper 1 showed that a signal of this kind has value if it carries information the
model's own confidence does not.

We score three predictors of an illegal next move, identically and on the same
positions, with intervals clustered on games:

  model confidence     the top-1 probability, the model's own estimate
  consistency error    1 minus cosine between the rolled and current states
  combined             both, fitted on training games, evaluated on held-out ones
"""
import os, json, argparse
import numpy as np
import torch
import torch.nn.functional as F
import chess
from probes import load_split, build
from scr import StateConsistency, MultiStepConsistency

DATA = os.path.join(os.path.dirname(__file__), "..", "data", "proc")
RUNS = os.path.join(os.path.dirname(__file__), "..", "runs")
RES = os.path.join(os.path.dirname(__file__), "..", "results")
COVERAGE = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]


def auc(s, y):
    s = np.asarray(s, float); y = np.asarray(y, int)
    n1, n0 = int(y.sum()), int((1 - y).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    o = np.argsort(s, kind="mergesort")
    r = np.empty(len(s), float); r[o] = np.arange(1, len(s) + 1)
    u, inv, cnt = np.unique(s, return_inverse=True, return_counts=True)
    sm = np.zeros(len(u)); np.add.at(sm, inv, r)
    r = (sm / cnt)[inv]
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def cluster_ci(fn, games, n_boot=300, seed=0):
    rng = np.random.default_rng(seed)
    uniq = np.unique(games); by = {g: np.where(games == g)[0] for g in uniq}
    out = []
    for _ in range(n_boot):
        pick = rng.choice(uniq, len(uniq), replace=True)
        sel = np.concatenate([by[g] for g in pick])
        v = fn(sel)
        if np.isfinite(v):
            out.append(v)
    return (float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))) \
        if out else (None, None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="attn_8L256_s0")
    ap.add_argument("--trans", default="_trans12")
    ap.add_argument("--games", type=int, default=1200)
    ap.add_argument("--lookback", type=int, default=12)
    ap.add_argument("--minply", type=int, default=20)
    args = ap.parse_args()

    model, a, ck = build(os.path.join(RUNS, f"{args.name}.pt"))
    stoi = json.load(open(os.path.join(DATA, "vocab.json")))
    itos = {v: k for k, v in stoi.items()}
    tr = torch.load(os.path.join(RUNS, f"{args.name}{args.trans}.pt"),
                    map_location="cuda", weights_only=False)
    dim, layer = tr["args"]["dim"], tr["args"]["layer"]
    ms = tr["args"].get("multistep", 0)
    cls = MultiStepConsistency if ms > 0 else StateConsistency
    kw = {"max_k": ms} if ms > 0 else {}
    scr = cls(a["width"], d_state=dim, vocab=len(stoi), **kw).cuda()
    scr.load_state_dict(tr["scr"]); scr.eval()

    toks, lens, occ = load_split("eval", args.games)
    conf, cerr, y, game = [], [], [], []

    with torch.no_grad():
        for gi in range(len(toks)):
            T = int(lens[gi])
            if T < 30:
                continue
            seq = torch.tensor([int(v) for v in toks[gi, :T]],
                               dtype=torch.long).cuda()
            pos = [t for t in range(T - 1) if t >= max(args.minply,
                                                       args.lookback + 2)]
            if not pos:
                continue
            logits, hs, _ = model(seq[None], return_hidden=True)
            h = hs[layer].float()[0]
            lg = logits.float()[0]
            pt = torch.tensor(pos, device="cuda")
            t0 = pt - args.lookback
            z_now = scr.project(h[None])[0][pt]
            z = scr.project(h[None])[0][t0]
            for step in range(1, args.lookback + 1):
                mv = scr.move(seq[t0 + step])
                z = F.normalize(scr.trans(torch.cat([z, mv], dim=-1)), dim=-1)
            incons = (1.0 - (z * z_now).sum(-1)).cpu().numpy()
            pr = torch.softmax(lg[pt], -1)
            top = pr.max(-1)
            pred = top.indices.cpu().numpy()
            topv = top.values.cpu().numpy()

            board = chess.Board()
            ti = 0
            for t in range(T - 1):
                if ti < len(pos) and pos[ti] == t:
                    mv = itos.get(int(pred[ti]), "")
                    try:
                        legal = chess.Move.from_uci(mv) in board.legal_moves
                    except Exception:
                        legal = False
                    conf.append(float(topv[ti])); cerr.append(float(incons[ti]))
                    y.append(0 if legal else 1); game.append(gi)
                    ti += 1
                board.push_uci(itos[int(seq[t + 1])])
            if len(y) > 60000:
                break

    conf = np.array(conf); cerr = np.array(cerr)
    y = np.array(y); game = np.array(game)

    out = {"name": args.name, "trans": args.trans, "n": int(len(y)),
           "n_games": int(len(np.unique(game))), "base_rate": float(y.mean()),
           "auc": {}, "coverage": {}}
    for nm, sc in (("model_confidence", -conf), ("consistency_error", cerr)):
        a_ = auc(sc, y)
        lo, hi = cluster_ci(lambda sel, sc=sc: auc(sc[sel], y[sel]), game)
        out["auc"][nm] = {"auc": a_, "ci_lo": lo, "ci_hi": hi}

    # combined, fitted on training games and evaluated on held-out games
    uniq = np.unique(game); rng = np.random.default_rng(0); rng.shuffle(uniq)
    tr_g = set(uniq[:int(0.7 * len(uniq))])
    trm = np.array([g in tr_g for g in game]); te = ~trm
    X = np.column_stack([-conf, cerr, np.ones(len(y))])
    mu, sd = X[trm].mean(0), X[trm].std(0) + 1e-9
    mu[-1], sd[-1] = 0.0, 1.0
    Z = (X - mu) / sd
    w = np.zeros(3)
    for _ in range(800):
        p = 1 / (1 + np.exp(-Z[trm] @ w))
        w -= 0.2 * (Z[trm].T @ (p - y[trm])) / trm.sum()
    score = Z @ w
    out["combined"] = {"beta_confidence": float(w[0]),
                       "beta_consistency": float(w[1]),
                       "auc_heldout": auc(score[te], y[te]),
                       "auc_conf_heldout": auc(-conf[te], y[te]),
                       "auc_cons_heldout": auc(cerr[te], y[te])}
    for cov in COVERAGE:
        k = int(te.sum() * cov)
        kc = np.argsort(-conf[te], kind="mergesort")[:k]
        ke = np.argsort(cerr[te], kind="mergesort")[:k]
        kk = np.argsort(score[te], kind="mergesort")[:k]
        out["coverage"][f"{cov:.1f}"] = {
            "model_confidence": float(y[te][kc].mean()),
            "consistency_error": float(y[te][ke].mean()),
            "combined": float(y[te][kk].mean())}
    json.dump(out, open(os.path.join(RES, f"{args.name}_drift_detect.json"), "w"),
              indent=1)

    print(f"[{args.name}] n={out['n']} from {out['n_games']} games, "
          f"illegal rate {out['base_rate']:.4f}")
    for nm, v in out["auc"].items():
        print(f"  AUC {nm:20s} {v['auc']:.4f} [{v['ci_lo']:.4f}, {v['ci_hi']:.4f}]")
    c = out["combined"]
    print(f"  held-out AUC: confidence {c['auc_conf_heldout']:.4f}, "
          f"consistency {c['auc_cons_heldout']:.4f}, combined {c['auc_heldout']:.4f}")
    print(f"  betas: confidence {c['beta_confidence']:+.3f}, "
          f"consistency {c['beta_consistency']:+.3f}")
    print(f"\n{'coverage':>9} {'confidence':>11} {'consistency':>12} {'combined':>10}")
    for cov in COVERAGE:
        r = out["coverage"][f"{cov:.1f}"]
        print(f"{cov:9.1f} {r['model_confidence']:11.4f} "
              f"{r['consistency_error']:12.4f} {r['combined']:10.4f}")


if __name__ == "__main__":
    main()
