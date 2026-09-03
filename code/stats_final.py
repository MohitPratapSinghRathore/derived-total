"""Journal-level statistics for every headline result.

Positions within a game are not independent, so intervals computed over positions
understate uncertainty. Everything here clusters on the game:

  * game-clustered bootstrap intervals for belief consistency and its controls
  * game-clustered intervals for both AUCs and, paired, for their difference
  * a population-averaged logistic fit (GEE, exchangeable working correlation,
    game as the cluster) giving cluster-robust coefficients
  * nested models compared by held-out log loss on a split by game, so the
    incremental value of each predictor is measured out of sample rather than
    read off an in-sample coefficient
  * the combined abstention rule refitted on training games and evaluated on
    held-out games, since fitting and evaluating the same weights on the same
    positions overstates what it would deliver
"""
import os, json, argparse, warnings
import numpy as np
import torch
import chess
from probes import load_split, build

warnings.filterwarnings("ignore")
DATA = os.path.join(os.path.dirname(__file__), "..", "data", "proc")
RUNS = os.path.join(os.path.dirname(__file__), "..", "runs")
RES = os.path.join(os.path.dirname(__file__), "..", "results")
BUCKETS = [(10, 20), (20, 30), (30, 40), (40, 50), (50, 60), (60, 80)]

CLS2PIECE = {}
for ci, col in enumerate([chess.WHITE, chess.BLACK]):
    for pi, pt in enumerate([chess.PAWN, chess.KNIGHT, chess.BISHOP,
                             chess.ROOK, chess.QUEEN, chess.KING]):
        CLS2PIECE[1 + ci * 6 + pi] = (pt, col)


def auc(s, y):
    s = np.asarray(s, float); y = np.asarray(y, int)
    n1, n0 = int(y.sum()), int((1 - y).sum())
    if n1 == 0 or n0 == 0:
        return np.nan
    order = np.argsort(s, kind="mergesort")
    r = np.empty(len(s), float); r[order] = np.arange(1, len(s) + 1)
    u, inv, cnt = np.unique(s, return_inverse=True, return_counts=True)
    sm = np.zeros(len(u)); np.add.at(sm, inv, r)
    r = (sm / cnt)[inv]
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def clustered_boot(fn, games, n_boot=400, seed=0):
    """Resample whole games, recompute the statistic."""
    rng = np.random.default_rng(seed)
    uniq = np.unique(games)
    by = {g: np.where(games == g)[0] for g in uniq}
    out = []
    for _ in range(n_boot):
        pick = rng.choice(uniq, len(uniq), replace=True)
        sel = np.concatenate([by[g] for g in pick])
        v = fn(sel)
        if v is not None and np.isfinite(v):
            out.append(v)
    if not out:
        return None, None
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def believed_board(row, turn_white):
    b = chess.Board(None)
    for sq in range(64):
        c = int(row[sq])
        if c:
            pt, col = CLS2PIECE[c]
            b.set_piece_at(sq, chess.Piece(pt, col))
    b.turn = chess.WHITE if turn_white else chess.BLACK
    b.castling_rights = 0
    return b


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
    all_moves = [m for m in stoi if m not in ("<pad>", "<bos>")]
    rng = np.random.default_rng(0)

    F = {k: [] for k in ("agg", "touch", "conf", "y", "game", "depth",
                         "bel", "bel_mis", "bel_rand", "is_ill", "pcert")}
    bank = {b: [] for b in range(len(BUCKETS))}

    with torch.no_grad():
        for i in range(0, len(toks), 32):
            idx = np.arange(i, min(i + 32, len(toks)))
            x = torch.from_numpy(toks[idx]).cuda()
            s = torch.from_numpy(np.asarray(occ[idx]).astype(np.int64)).cuda()
            with torch.amp.autocast("cuda", dtype=torch.float16):
                logits, hs, _ = model(x[:, :-1], return_hidden=True)
            lp = logits.float()
            pr = torch.softmax(lp, -1)
            topv = pr.max(-1).values.cpu().numpy()
            pred = lp.argmax(-1).cpu().numpy()
            plog = probe(hs[best].float()).reshape(len(idx), -1, 64, 13)
            dec = plog.argmax(-1)
            # probe certainty needs no oracle and is available at inference
            pmax = torch.softmax(plog, -1).max(-1).values.cpu().numpy()
            ok = (dec == s[:, :dec.shape[1]]).cpu().numpy()
            decn = dec.cpu().numpy()
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
                            F["agg"].append(int((~row).sum()))
                            F["touch"].append(int(not (row[m_.from_square]
                                                       and row[m_.to_square])))
                            F["conf"].append(float(topv[r, t]))
                            F["pcert"].append(float(min(pmax[r, t, m_.from_square],
                                                        pmax[r, t, m_.to_square])))
                            F["y"].append(0 if legal else 1)
                            F["game"].append(int(gi))
                            F["depth"].append(bi)
                            if not legal:
                                bb = believed_board(decn[r, t],
                                                    board.turn == chess.WHITE)
                                F["is_ill"].append(1)
                                F["bel"].append(int(m_ in bb.pseudo_legal_moves))
                                if bank[bi]:
                                    ob = believed_board(
                                        bank[bi][int(rng.integers(len(bank[bi])))],
                                        board.turn == chess.WHITE)
                                    F["bel_mis"].append(int(m_ in ob.pseudo_legal_moves))
                                else:
                                    F["bel_mis"].append(0)
                                cand = None
                                for _ in range(6):
                                    cd = chess.Move.from_uci(
                                        all_moves[int(rng.integers(len(all_moves)))])
                                    if cd not in board.legal_moves:
                                        cand = cd; break
                                F["bel_rand"].append(
                                    int(cand in bb.pseudo_legal_moves) if cand else 0)
                            if len(bank[bi]) < 400:
                                bank[bi].append(decn[r, t].copy())
                    board.push_uci(itos[int(toks[gi, t + 1])])

    agg = np.array(F["agg"], float); touch = np.array(F["touch"], float)
    conf = np.array(F["conf"], float); y = np.array(F["y"], int)
    pcert = np.array(F["pcert"], float)
    game = np.array(F["game"]); depth = np.array(F["depth"], float)
    ill_mask = y == 1
    bel = np.array(F["bel"], float); bel_mis = np.array(F["bel_mis"], float)
    bel_rand = np.array(F["bel_rand"], float)
    ill_games = game[ill_mask]

    out = {"name": args.name, "n": int(len(y)), "n_illegal": int(ill_mask.sum()),
           "n_games": int(len(np.unique(game)))}

    # ---- belief consistency and controls, game-clustered
    for nm, v in (("belief", bel), ("belief_mismatched", bel_mis),
                  ("belief_random", bel_rand)):
        lo, hi = clustered_boot(lambda sel, v=v: v[sel].mean(), ill_games)
        out[nm] = {"mean": float(v.mean()), "ci_lo": lo, "ci_hi": hi,
                   "n": int(len(v))}

    # ---- AUCs and their paired difference, game-clustered
    for nm, sc in (("auc_aggregate", agg), ("auc_touched", touch)):
        lo, hi = clustered_boot(lambda sel, sc=sc: auc(sc[sel], y[sel]), game)
        out[nm] = {"auc": auc(sc, y), "ci_lo": lo, "ci_hi": hi}
    lo, hi = clustered_boot(
        lambda sel: auc(touch[sel], y[sel]) - auc(agg[sel], y[sel]), game)
    out["auc_difference"] = {"diff": out["auc_touched"]["auc"]
                             - out["auc_aggregate"]["auc"],
                             "ci_lo": lo, "ci_hi": hi}

    # ---- population-averaged logistic with game clusters
    try:
        import statsmodels.api as sm
        X = np.column_stack([
            (agg - agg.mean()) / agg.std(),
            (touch - touch.mean()) / touch.std(),
            (-conf - (-conf).mean()) / conf.std(),
            (depth - depth.mean()) / depth.std(),
            np.ones(len(y))])
        gee = sm.GEE(y, X, groups=game, family=sm.families.Binomial(),
                     cov_struct=sm.cov_struct.Exchangeable()).fit(maxiter=30)
        names = ["aggregate", "move_touched", "low_confidence", "depth", "const"]
        out["gee"] = {n: {"coef": float(gee.params[k]),
                          "robust_se": float(gee.bse[k]),
                          "z": float(gee.tvalues[k]),
                          "p": float(gee.pvalues[k])}
                      for k, n in enumerate(names)}
    except Exception as e:
        out["gee"] = {"error": str(e)}

    # ---- nested models, held-out log loss, split by game
    uniq = np.unique(game)
    rs = np.random.default_rng(0); rs.shuffle(uniq)
    tr_g = set(uniq[:int(0.7 * len(uniq))])
    tr = np.array([g in tr_g for g in game]); te = ~tr

    def fit_ll(cols):
        Xtr = np.column_stack(cols + [np.ones(len(y))])[tr]
        Xte = np.column_stack(cols + [np.ones(len(y))])[te]
        mu = Xtr.mean(0); sd = Xtr.std(0) + 1e-9
        mu[-1], sd[-1] = 0.0, 1.0
        Ztr, Zte = (Xtr - mu) / sd, (Xte - mu) / sd
        w = np.zeros(Ztr.shape[1])
        for _ in range(600):
            p = 1 / (1 + np.exp(-Ztr @ w))
            w -= 0.2 * (Ztr.T @ (p - y[tr])) / tr.sum()
        p = np.clip(1 / (1 + np.exp(-Zte @ w)), 1e-9, 1 - 1e-9)
        return float(-(y[te] * np.log(p) + (1 - y[te]) * np.log(1 - p)).mean())

    base = fit_ll([depth])
    out["heldout_logloss"] = {
        "depth_only": base,
        "plus_aggregate": fit_ll([depth, agg]),
        "plus_touched": fit_ll([depth, touch]),
        "plus_both": fit_ll([depth, agg, touch]),
        "plus_both_and_confidence": fit_ll([depth, agg, touch, -conf]),
        "n_train_games": int(len(tr_g)), "n_test_positions": int(te.sum())}

    # ---- abstention, weights fitted on training games only.
    # The deployable rule may use probe CERTAINTY, which needs no oracle. The
    # oracle-correctness version is reported separately as an upper bound, never
    # as a method: establishing it at inference would require the ground truth
    # the rule exists to do without.
    def fit_score(cols):
        Xc = np.column_stack(cols + [np.ones(len(y))])
        mu = Xc[tr].mean(0); sd = Xc[tr].std(0) + 1e-9
        mu[-1], sd[-1] = 0.0, 1.0
        Z = (Xc - mu) / sd
        w = np.zeros(Z.shape[1])
        for _ in range(600):
            pz = 1 / (1 + np.exp(-Z[tr] @ w))
            w -= 0.2 * (Z[tr].T @ (pz - y[tr])) / tr.sum()
        return Z @ w

    sc_comb = fit_score([-conf, -pcert])          # deployable
    sc_bound = fit_score([-conf, touch])          # oracle bound
    cov_rows = {}
    for cov in (1.0, 0.9, 0.8, 0.7, 0.6, 0.5):
        k = int(te.sum() * cov)
        kc = np.argsort(-conf[te], kind="mergesort")[:k]
        kp = np.argsort(-pcert[te], kind="mergesort")[:k]
        kk = np.argsort(sc_comb[te], kind="mergesort")[:k]
        kb = np.argsort(sc_bound[te], kind="mergesort")[:k]
        cov_rows[f"{cov:.1f}"] = {
            "model_confidence": float(y[te][kc].mean()),
            "probe_certainty": float(y[te][kp].mean()),
            "combined_deployable": float(y[te][kk].mean()),
            "oracle_bound": float(y[te][kb].mean())}
    out["abstain_heldout"] = cov_rows

    json.dump(out, open(os.path.join(RES, f"{args.name}_stats.json"), "w"), indent=1)

    print(f"[{args.name}] n={out['n']} positions, {out['n_games']} games, "
          f"{out['n_illegal']} illegal")
    for k in ("belief", "belief_mismatched", "belief_random"):
        v = out[k]
        print(f"  {k:20s} {v['mean']:.4f}  [{v['ci_lo']:.4f}, {v['ci_hi']:.4f}]")
    for k in ("auc_aggregate", "auc_touched"):
        v = out[k]
        print(f"  {k:20s} {v['auc']:.4f}  [{v['ci_lo']:.4f}, {v['ci_hi']:.4f}]")
    d = out["auc_difference"]
    print(f"  {'auc difference':20s} {d['diff']:+.4f}  "
          f"[{d['ci_lo']:+.4f}, {d['ci_hi']:+.4f}]")
    if "error" not in out["gee"]:
        print("  GEE, game-clustered:")
        for n, v in out["gee"].items():
            if n != "const":
                print(f"    {n:16s} {v['coef']:+.4f} (se {v['robust_se']:.4f}, "
                      f"p {v['p']:.2g})")
    print("  held-out log loss:")
    for k, v in out["heldout_logloss"].items():
        if isinstance(v, float):
            print(f"    {k:26s} {v:.5f}")


if __name__ == "__main__":
    main()
