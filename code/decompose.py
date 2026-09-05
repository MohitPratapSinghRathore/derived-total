"""S2: four-way decomposition of long-horizon failure.

Pre-registered in PREREGISTRATION.md (amended, sha256 067e024e...). Thresholds
are frozen there and are not tuned here.

Every illegal top-1 move at ply >= 20 is assigned to exactly one bucket:

  memory      correct repair makes the top-1 move legal, and a reference engine
              scores the repaired move within 100 centipawns of its own best.
              The state was wrong and the policy was fine.
  mixed       repair makes the top-1 legal but the move is weak. The state was
              wrong and fixing it is not sufficient.
  policy      belief was already correct at both squares the move touches, so
              repair has nothing to install. The state was fine and the policy
              chose an illegal move anyway.
  unresolved  belief was wrong at the action squares and repair did not make the
              top-1 legal.

The frozen prior is that memory plus mixed lands near 0.34, from belief
consistency, and policy near 0.20, from the week-0 count. A split far outside
that envelope is treated as a threshold or implementation error before it is
treated as a finding.

The engine is used only to grade repaired moves, never to select them, so it
cannot leak into the classification of which failures repair fixes.
"""
import os, json, argparse
import numpy as np
import torch
import chess
import chess.engine
from probes import load_split, build

DATA = os.path.join(os.path.dirname(__file__), "..", "data", "proc")
RUNS = os.path.join(os.path.dirname(__file__), "..", "runs")
RES = os.path.join(os.path.dirname(__file__), "..", "results")
SF = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "sf",
                                  "stockfish",
                                  "stockfish-windows-x86-64-avx2.exe"))
BUCKETS = ["memory", "mixed", "policy", "unresolved"]
BLUNDER_CP = 100      # frozen in PREREGISTRATION.md section 2
DEPTH = 12            # frozen
MOVETIME = 0.10       # frozen


def boot_shares(labels, games, n_boot=500, seed=0):
    rng = np.random.default_rng(seed)
    labels = np.asarray(labels)
    games = np.asarray(games)
    uniq = np.unique(games)
    by = {g: np.where(games == g)[0] for g in uniq}
    acc = {b: [] for b in BUCKETS}
    for _ in range(n_boot):
        sel = np.concatenate([by[g] for g in
                              rng.choice(uniq, len(uniq), replace=True)])
        l = labels[sel]
        for b in BUCKETS:
            acc[b].append(float((l == b).mean()))
    return {b: (float(np.mean(acc[b])), float(np.percentile(acc[b], 2.5)),
                float(np.percentile(acc[b], 97.5))) for b in BUCKETS}


def cp_loss(eng, board, move, lim):
    """Centipawn loss of `move` against the engine's best, side-to-move view."""
    info = eng.analyse(board, lim)
    best = info["score"].pov(board.turn).score(mate_score=10000)
    b2 = board.copy()
    b2.push(move)
    if b2.is_game_over():
        r = b2.result()
        after = 10000 if r == ("1-0" if board.turn else "0-1") else \
            (-10000 if r != "1/2-1/2" else 0)
    else:
        i2 = eng.analyse(b2, lim)
        after = i2["score"].pov(board.turn).score(mate_score=10000)
    if best is None or after is None:
        return None
    return float(best - after)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="attn_8L256_s0")
    ap.add_argument("--n", type=int, default=700)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--minply", type=int, default=20)
    ap.add_argument("--games", type=int, default=3000)
    ap.add_argument("--repair_all", action="store_true",
                    help="install true occupancy at EVERY divergent square, not "
                         "only the two the move touches. Separates a genuine "
                         "unresolved bucket from an artefact of partial repair.")
    args = ap.parse_args()

    model, a, ck = build(os.path.join(RUNS, f"{args.name}.pt"))
    res = json.load(open(os.path.join(RES, f"{args.name}.json")))
    best_l = res["best_layer"]
    pls = torch.load(os.path.join(RES, f"{args.name}_probes.pt"),
                     weights_only=False)
    probe = torch.nn.Linear(a["width"], 64 * 13).cuda()
    probe.load_state_dict(pls[best_l])
    probe.eval()
    W = probe.weight.reshape(64, 13, a["width"]).float()

    stoi = json.load(open(os.path.join(DATA, "vocab.json")))
    itos = {v: k for k, v in stoi.items()}
    toks, lens, occ = load_split("eval", args.games)

    eng = chess.engine.SimpleEngine.popen_uci(SF)
    eng.configure({"Threads": 1})
    lim = chess.engine.Limit(depth=DEPTH, time=MOVETIME)

    def forward(x, edit):
        store = {}

        def hook(mod, inp, out):
            o = out
            if edit is not None:
                o = o.clone()
                o[:, -1] = o[:, -1] + edit
            store["h"] = o
            return o

        hd = model.blocks[best_l].register_forward_hook(hook)
        with torch.amp.autocast("cuda", dtype=torch.float16):
            lg, _, _ = model(x)
        hd.remove()
        h = store["h"][0, -1].float()
        return torch.softmax(lg.float()[0, -1], -1), \
            probe(h).reshape(64, 13).argmax(-1)

    labels, games, cps, plies = [], [], [], []
    n_done = 0
    with torch.no_grad():
        for gi in range(len(toks)):
            if n_done >= args.n:
                break
            T = int(lens[gi])
            if T < args.minply + 4:
                continue
            board = chess.Board()
            for t in range(T - 1):
                if n_done >= args.n:
                    break
                if t >= args.minply:
                    x = torch.from_numpy(
                        toks[gi:gi + 1, :t + 1].astype(np.int64)).cuda()
                    p0, dec0 = forward(x, None)
                    top1 = int(p0.argmax())
                    try:
                        m_ill = chess.Move.from_uci(itos.get(top1, ""))
                        illegal = m_ill not in board.legal_moves
                    except Exception:
                        m_ill, illegal = None, False
                    if m_ill is not None and illegal:
                        truth = np.asarray(occ[gi, t]).astype(int)
                        bel = dec0.cpu().numpy()
                        act = [m_ill.from_square, m_ill.to_square]
                        div = [s for s in act if bel[s] != truth[s]]
                        if args.repair_all and div:
                            div = [s for s in range(64) if bel[s] != truth[s]]
                        if not div:
                            labels.append("policy")
                            games.append(gi)
                            cps.append(np.nan)
                            plies.append(t)
                            n_done += 1
                        else:
                            with torch.amp.autocast("cuda",
                                                    dtype=torch.float16):
                                _, hs0, _ = model(x, return_hidden=True)
                            rms = float(hs0[best_l][0, -1].float()
                                        .pow(2).mean().sqrt())
                            scale = args.alpha * rms * np.sqrt(a["width"])
                            d = torch.zeros(a["width"], device="cuda")
                            for s in div:
                                v = W[s, int(truth[s])] - W[s, int(bel[s])]
                                if float(v.norm()) > 1e-6:
                                    d = d + v / v.norm()
                            if float(d.norm()) > 1e-6:
                                d = d / d.norm() * scale
                            p1, _ = forward(x, d)
                            nt = int(p1.argmax())
                            try:
                                mv2 = chess.Move.from_uci(itos.get(nt, ""))
                                ok = mv2 in board.legal_moves
                            except Exception:
                                mv2, ok = None, False
                            if ok:
                                c = cp_loss(eng, board, mv2, lim)
                                labels.append(
                                    "memory" if (c is not None and
                                                 c <= BLUNDER_CP) else "mixed")
                                cps.append(np.nan if c is None else c)
                            else:
                                labels.append("unresolved")
                                cps.append(np.nan)
                            games.append(gi)
                            plies.append(t)
                            n_done += 1
                try:
                    board.push_uci(itos[int(toks[gi, t + 1])])
                except Exception:
                    break
    eng.quit()

    labels = np.array(labels)
    games = np.array(games)
    plies = np.array(plies)
    sh = boot_shares(labels, games)
    out = {"name": args.name, "repair_all": bool(args.repair_all),
           "n": int(len(labels)),
           "n_games": int(len(np.unique(games))),
           "alpha": args.alpha, "blunder_cp": BLUNDER_CP, "depth": DEPTH,
           "shares": {b: {"mean": sh[b][0], "ci_lo": sh[b][1],
                          "ci_hi": sh[b][2]} for b in BUCKETS},
           "prior": {"memory_plus_mixed": 0.34, "policy": 0.20}}
    mm = float(((labels == "memory") | (labels == "mixed")).mean())
    out["memory_plus_mixed"] = mm
    cpa = np.array(cps, float)
    fin = np.isfinite(cpa)
    out["median_cp_loss_repaired"] = (float(np.median(cpa[fin]))
                                      if fin.sum() else None)

    print(f"\n[{args.name}] S2 decomposition, n={len(labels)} illegal top-1 "
          f"moves from {len(np.unique(games))} games")
    print(f"{'bucket':>12} {'share':>8}  {'95% CI (game-clustered)':>26}")
    for b in BUCKETS:
        m, lo, hi = sh[b]
        print(f"{b:>12} {m:8.4f}  [{lo:.4f}, {hi:.4f}]")
    print(f"\n  memory + mixed = {mm:.4f}   (frozen prior 0.34)")
    print(f"  policy         = {float((labels=='policy').mean()):.4f}"
          f"   (frozen prior 0.20)")
    if out["median_cp_loss_repaired"] is not None:
        print(f"  median centipawn loss of repaired moves = "
              f"{out['median_cp_loss_repaired']:.0f}")

    print(f"\nby depth:")
    print(f"{'ply':>9} {'n':>6} " + " ".join(f"{b:>11}" for b in BUCKETS))
    for lo_, hi_ in [(20, 40), (40, 60), (60, 120)]:
        m = (plies >= lo_) & (plies < hi_)
        if m.sum() < 25:
            continue
        out.setdefault("strata", {})[f"{lo_}-{hi_}"] = {
            "n": int(m.sum()),
            **{b: float((labels[m] == b).mean()) for b in BUCKETS}}
        print(f"{str(lo_)+'-'+str(hi_):>9} {m.sum():6,} "
              + " ".join(f"{float((labels[m]==b).mean()):11.4f}"
                         for b in BUCKETS))

    suf = "_decompose_all" if args.repair_all else "_decompose"
    json.dump(out, open(os.path.join(RES, f"{args.name}{suf}.json"), "w"),
              indent=1)


if __name__ == "__main__":
    main()
