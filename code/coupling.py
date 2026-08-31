"""Does internal state loss actually EXPLAIN behavioural failure?

The paper so far shows two curves that fall together: probe-decodable state
declines with depth, illegal moves rise with depth. Two curves moving together
is not evidence that one causes the other, and the whole decomposition rests on
the claim that they are the same phenomenon seen from inside and outside.

This measures the link directly, per token, three ways.

(A) COUPLING. At each position, is the model's top-1 move illegal, and is the
    probe's decoded state wrong? If state loss drives behavioural failure, the
    illegal rate should be far higher where the state is wrong, WITHIN a fixed
    depth bucket (so the correlation is not just both tracking depth).

(B) LOCALITY. An illegal move has a source square and a destination square. If
    the state representation is what the policy reads, illegality should track
    error on the SPECIFIC squares that move touches, not just global state error.

(C) RECENCY. For each square, how long since its occupancy last changed? If the
    model re-derives state by composing recent updates, error should concentrate
    on squares whose last change is far back. This distinguishes "forgets the
    distant past" from "fails to integrate recent updates".
"""
import os, json, argparse
import numpy as np
import torch
import chess
from probes import load_split, build

DATA = os.path.join(os.path.dirname(__file__), "..", "data", "proc")
RUNS = os.path.join(os.path.dirname(__file__), "..", "runs")
RES = os.path.join(os.path.dirname(__file__), "..", "results")
BUCKETS = [(0, 10), (10, 20), (20, 30), (30, 40), (40, 50),
           (50, 60), (60, 80), (80, 120), (120, 161)]
AGE_BINS = [(0, 1), (1, 3), (3, 6), (6, 12), (12, 25), (25, 50), (50, 200)]


def bucket_of(t):
    for i, (lo, hi) in enumerate(BUCKETS):
        if lo <= t < hi:
            return i
    return None


def age_bin(a):
    for i, (lo, hi) in enumerate(AGE_BINS):
        if lo <= a < hi:
            return i
    return len(AGE_BINS) - 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--games", type=int, default=2000)
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

    nb = len(BUCKETS)
    # (A) coupling: illegal rate split by whether the probe state is exact
    ill_bad = np.zeros(nb); n_bad = np.zeros(nb)
    ill_good = np.zeros(nb); n_good = np.zeros(nb)
    # coarser: number of wrong squares vs illegality (for a rank correlation)
    wrong_counts, illegal_flags, depth_flags = [], [], []
    # (B) locality: is the error on the squares this move touches?
    loc_hit = np.zeros(4)   # [illegal&srcwrong, illegal&srcok, legal&srcwrong, legal&srcok]
    # (C) recency
    age_err = np.zeros(len(AGE_BINS)); age_n = np.zeros(len(AGE_BINS))

    bs = 32
    with torch.no_grad():
        for i in range(0, len(toks), bs):
            idx = np.arange(i, min(i + bs, len(toks)))
            x = torch.from_numpy(toks[idx]).cuda()
            s = torch.from_numpy(np.asarray(occ[idx]).astype(np.int64)).cuda()
            with torch.amp.autocast("cuda", dtype=torch.float16):
                logits, hs, _ = model(x[:, :-1], return_hidden=True)
            pred_tok = logits.float().argmax(-1).cpu().numpy()
            pstate = probe(hs[best].float()).reshape(len(idx), -1, 64, 13).argmax(-1)
            correct = (pstate == s[:, :pstate.shape[1]]).cpu().numpy()   # (B,T,64)

            occ_np = np.asarray(occ[idx]).astype(np.int64)
            for r, gi in enumerate(idx):
                T = int(lens[gi])
                board = chess.Board()
                # last-change ply per square
                last_change = np.zeros(64, dtype=np.int64)
                for t in range(min(T - 1, correct.shape[1])):
                    b = bucket_of(t)
                    if b is None:
                        continue
                    mv = itos.get(int(pred_tok[r, t]), "")
                    try:
                        m_ = chess.Move.from_uci(mv)
                        legal = m_ in board.legal_moves
                    except Exception:
                        m_, legal = None, False
                    cs = correct[r, t]
                    exact = bool(cs.all())
                    nwrong = int((~cs).sum())

                    if exact:
                        ill_good[b] += (not legal); n_good[b] += 1
                    else:
                        ill_bad[b] += (not legal); n_bad[b] += 1
                    wrong_counts.append(nwrong)
                    illegal_flags.append(0 if legal else 1)
                    depth_flags.append(b)

                    # (B) locality on the squares the predicted move touches
                    if m_ is not None:
                        touched_ok = bool(cs[m_.from_square] and cs[m_.to_square])
                        loc_hit[(0 if not legal else 2) + (0 if not touched_ok else 1)] += 1

                    # (C) recency, sampled to keep this affordable
                    if t % 7 == 0:
                        for sq in range(0, 64, 3):
                            age_bin_i = age_bin(t - last_change[sq])
                            age_err[age_bin_i] += (not cs[sq])
                            age_n[age_bin_i] += 1

                    nxt = itos[int(toks[gi, t + 1])]
                    prev = occ_np[r, t].copy()
                    board.push_uci(nxt)
                    if t + 1 < occ_np.shape[1]:
                        changed = np.nonzero(occ_np[r, t + 1] != prev)[0]
                        last_change[changed] = t + 1

    wrong_counts = np.array(wrong_counts)
    illegal_flags = np.array(illegal_flags)
    depth_flags = np.array(depth_flags)

    # within-bucket point-biserial correlation between #wrong squares and illegality
    corr_by_bucket = []
    for b in range(nb):
        m = depth_flags == b
        if m.sum() > 50 and illegal_flags[m].std() > 0 and wrong_counts[m].std() > 0:
            corr_by_bucket.append(float(np.corrcoef(wrong_counts[m], illegal_flags[m])[0, 1]))
        else:
            corr_by_bucket.append(None)

    out = {
        "name": args.name, "buckets": BUCKETS, "age_bins": AGE_BINS,
        "illegal_given_state_wrong": (ill_bad / np.maximum(n_bad, 1)).tolist(),
        "illegal_given_state_exact": (ill_good / np.maximum(n_good, 1)).tolist(),
        "n_state_wrong": n_bad.tolist(), "n_state_exact": n_good.tolist(),
        "within_bucket_corr_wrongsquares_illegal": corr_by_bucket,
        "locality": {"illegal_touched_wrong": float(loc_hit[0]),
                     "illegal_touched_ok": float(loc_hit[1]),
                     "legal_touched_wrong": float(loc_hit[2]),
                     "legal_touched_ok": float(loc_hit[3])},
        "error_by_age": (age_err / np.maximum(age_n, 1)).tolist(),
        "n_by_age": age_n.tolist(),
    }
    json.dump(out, open(os.path.join(RES, f"{args.name}_coupling.json"), "w"), indent=1)

    print(f"[{args.name}] coupling")
    print(f"{'ply':>9} {'ill|wrong':>10} {'ill|exact':>10} {'n_wrong':>9} {'n_exact':>9}")
    for i, (lo, hi) in enumerate(BUCKETS):
        print(f"{str(lo)+'-'+str(hi):>9} "
              f"{out['illegal_given_state_wrong'][i]:10.4f} "
              f"{out['illegal_given_state_exact'][i]:10.4f} "
              f"{n_bad[i]:9.0f} {n_good[i]:9.0f}")
    print("within-bucket corr(#wrong squares, illegal):",
          [None if c is None else round(c, 3) for c in corr_by_bucket])
    L = out["locality"]
    ill_tot = L["illegal_touched_wrong"] + L["illegal_touched_ok"]
    leg_tot = L["legal_touched_wrong"] + L["legal_touched_ok"]
    print(f"locality: P(touched squares wrong | illegal) = "
          f"{L['illegal_touched_wrong']/max(ill_tot,1):.4f}   "
          f"P(touched squares wrong | legal) = {L['legal_touched_wrong']/max(leg_tot,1):.4f}")
    print("error by age:", np.round(out["error_by_age"], 4).tolist())


if __name__ == "__main__":
    main()
