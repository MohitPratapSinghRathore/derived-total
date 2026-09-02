"""Is action-relevant fidelity useful, or only diagnostic?

The paper establishes that state loss explains behavioural failure. That is a
claim about mechanism. This asks whether the same measurement is worth anything
operationally, and against the only baseline that matters: the model's own
confidence.

A system that wants to avoid acting on a decayed state can abstain. The question
is what to abstain on. Three signals are compared, each used to rank positions
for abstention:

  model confidence   the top-1 probability the model assigns (its own estimate)
  model entropy      the entropy of its next-move distribution
  probe uncertainty  how confident the probe is about the squares the chosen
                     move touches, computed from activations alone

An earlier version of this experiment used whether the probe was WRONG on those
squares. That is not deployable: establishing it requires the oracle, which is
the very thing an abstention rule exists to do without. The signal here uses only
the probe's own certainty, so it can be computed at inference from activations,
having fitted the probe once against an oracle offline. We report the
oracle-dependent version too, as an upper bound on what a perfect state monitor
could achieve.

If the probe signal only recovers what the model already knows about its own
uncertainty, it is diagnostically interesting and operationally redundant. If it
carries information the model's confidence does not, an external monitor can
catch failures the model cannot self-report, which is a practical claim.

Reported: illegal-move rate among retained positions at each coverage level, and
the incremental value of the probe signal in a joint logistic fit with confidence.
"""
import os, json, argparse
import numpy as np
import torch
import chess
from probes import load_split, build

DATA = os.path.join(os.path.dirname(__file__), "..", "data", "proc")
RUNS = os.path.join(os.path.dirname(__file__), "..", "runs")
RES = os.path.join(os.path.dirname(__file__), "..", "results")
COVERAGE = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]
MINPLY = 20            # abstention only matters where failures actually occur


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


def logistic(X, y, iters=400, lr=0.15):
    X = np.asarray(X, float)
    Z = (X - X.mean(0)) / (X.std(0) + 1e-9)
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

    conf, ent, act, illegal, punc = [], [], [], [], []
    with torch.no_grad():
        for i in range(0, len(toks), 32):
            idx = np.arange(i, min(i + 32, len(toks)))
            x = torch.from_numpy(toks[idx]).cuda()
            s = torch.from_numpy(np.asarray(occ[idx]).astype(np.int64)).cuda()
            with torch.amp.autocast("cuda", dtype=torch.float16):
                logits, hs, _ = model(x[:, :-1], return_hidden=True)
            lp = logits.float()
            pr = torch.softmax(lp, -1)
            top = pr.max(-1)
            H = -(pr * torch.log(pr.clamp_min(1e-9))).sum(-1)
            pred = lp.argmax(-1).cpu().numpy()
            topv = top.values.cpu().numpy(); Hv = H.cpu().numpy()
            plog = probe(hs[best].float()).reshape(len(idx), -1, 64, 13)
            pprob = torch.softmax(plog, -1)
            dec = plog.argmax(-1)
            # probe certainty per square: the mass on its own top class.
            # Needs no oracle, so it is available at inference.
            pmax = pprob.max(-1).values.cpu().numpy()
            ok = (dec == s[:, :dec.shape[1]]).cpu().numpy()
            for r, gi in enumerate(idx):
                board = chess.Board()
                T = int(lens[gi])
                for t in range(min(T - 1, ok.shape[1])):
                    if t >= MINPLY:
                        mv = itos.get(int(pred[r, t]), "")
                        try:
                            m_ = chess.Move.from_uci(mv)
                            leg = m_ in board.legal_moves
                            tw = int(not (ok[r, t, m_.from_square]
                                          and ok[r, t, m_.to_square]))
                        except Exception:
                            m_, leg, tw = None, False, 1
                        if m_ is not None:
                            punc.append(float(min(pmax[r, t, m_.from_square],
                                                  pmax[r, t, m_.to_square])))
                            conf.append(float(topv[r, t]))
                            ent.append(float(Hv[r, t]))
                            act.append(tw)
                            illegal.append(0 if leg else 1)
                    board.push_uci(itos[int(toks[gi, t + 1])])

    conf = np.array(conf); ent = np.array(ent)
    act = np.array(act); y = np.array(illegal); punc = np.array(punc)
    n = len(y)
    base = float(y.mean())

    out = {"name": args.name, "n": int(n), "base_illegal_rate": base,
           "auc": {"model_confidence": auc(-conf, y), "model_entropy": auc(ent, y),
                   "probe_uncertainty": auc(-punc, y),
                   "action_relevant_oracle": auc(act, y)},
           "coverage": {}}

    # risk at coverage: keep the fraction of positions each signal ranks safest
    # combined ranker: standardise both, weight by the joint fit
    _b = logistic(np.stack([-conf, -punc], 1), y)
    zc = (-conf - (-conf).mean()) / ((-conf).std() + 1e-9)
    zp = (-punc - (-punc).mean()) / ((-punc).std() + 1e-9)
    signals = {"model_confidence": -conf, "model_entropy": ent,
               "probe_uncertainty": -punc,
               "combined": _b[0] * zc + _b[1] * zp,
               "action_relevant_oracle": act.astype(float) + 1e-6 * ent}
    for cov in COVERAGE:
        k = int(n * cov)
        row = {}
        for nm, sc in signals.items():
            keep = np.argsort(sc, kind="mergesort")[:k]
            row[nm] = float(y[keep].mean())
        out["coverage"][f"{cov:.1f}"] = row

    beta = logistic(np.stack([-conf, -punc], 1), y)
    out["joint_fit"] = {"beta_model_confidence": float(beta[0]),
                        "beta_probe_uncertainty": float(beta[1])}
    json.dump(out, open(os.path.join(RES, f"{args.name}_abstain.json"), "w"),
              indent=1)

    print(f"[{args.name}] n={n}, illegal rate at full coverage {base:.4f}")
    print(f"  AUC  model confidence {out['auc']['model_confidence']:.3f}   "
          f"entropy {out['auc']['model_entropy']:.3f}   "
          f"action-relevant {out['auc']['action_relevant']:.3f}")
    print(f"\n{'coverage':>9} {'by confidence':>14} {'by entropy':>11} "
          f"{'by action-rel':>14}")
    for cov in COVERAGE:
        r = out["coverage"][f"{cov:.1f}"]
        print(f"{cov:9.1f} {r['model_confidence']:14.4f} "
              f"{r['model_entropy']:11.4f} {r['action_relevant']:14.4f}")
    print(f"\njoint fit: beta_confidence {beta[0]:+.3f}, "
          f"beta_action_relevant {beta[1]:+.3f}")


if __name__ == "__main__":
    main()
