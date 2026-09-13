"""S1: oracle-identified repair of the model's belief at action-relevant squares.

Pre-registered in PREREGISTRATION.md, sha256 916c89bd..., frozen before any edit
below was run. Read that document rather than this docstring for the binding
definitions; what follows is why the code looks the way it does.

The primary observable is a forced-choice preference, not probability mass on the
legal move:

    r = P(m*) / (P(m*) + P(m_ill))

where m_ill is the model's original illegal top-1 and m* is its
highest-probability move that is legal on the true board. Repair is a
perturbation and can flatten the distribution. Raw mass on m* rises under uniform
flattening whenever P(m*) < P(m_ill), which is the common case because m_ill is
the argmax, so a flattened distribution fakes success on it. That is the artefact
that made a Paper 2 ablation look like it worked. A ratio is invariant to uniform
rescaling and cannot be produced by hedging alone.

Controls, at the calibrated strength. The claim requires beating wrong-target,
not no-op:

  correct      install true occupancy at divergent action-relevant squares
  irrelevant   same edit budget spent on divergent squares the move does not
               touch. Natural divergence is diffuse, a median of 14 squares of
               64, so these are plentiful, and this is the control that tests
               whether coupling is local. It is the sharpest of the four.
  wrong_target install the action square's true occupancy at a different square
  random       norm-matched random direction
  null         re-install occupancy already believed correct, a zero direction by
               construction. Reports exactly zero unless the pipeline is broken.

Every condition reports the change in entropy it produced, and the contrast
against each control is additionally computed on the positions where the two
edits flattened the distribution comparably. That matters: the raw advantage of
correct repair over wrong-target is +0.081, and matched it is +0.031, so most of
the apparent effect was differential hedging rather than preference.

Note for anyone extending this. Two earlier versions of the matching were no-ops
that silently reported a column identical to the raw mean, because stratifying
and then weighting each stratum by its own count reproduces the overall mean
whatever the strata are. Matching must change which positions are counted.
"""
import os, json, argparse
import numpy as np
import torch
import chess
from probes import load_split, build

DATA = os.path.join(os.path.dirname(__file__), "..", "data", "proc")
RUNS = os.path.join(os.path.dirname(__file__), "..", "runs")
RES = os.path.join(os.path.dirname(__file__), "..", "results")
COND = ["correct", "irrelevant", "wrong_target", "random", "null"]


def boot_ci(vals, games, n_boot=500, seed=0):
    rng = np.random.default_rng(seed)
    vals = np.asarray(vals, float)
    games = np.asarray(games)
    uniq = np.unique(games)
    by = {g: np.where(games == g)[0] for g in uniq}
    out = []
    for _ in range(n_boot):
        pick = rng.choice(uniq, len(uniq), replace=True)
        sel = np.concatenate([by[g] for g in pick])
        out.append(vals[sel].mean())
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def subset_mean(vals, mask, games, n_boot=500, seed=0):
    """Effect restricted to positions the edit did not flatten."""
    vals = np.asarray(vals, float)[mask]
    g = np.asarray(games)[mask]
    if len(vals) < 20:
        return float("nan"), float("nan"), float("nan"), int(len(vals))
    lo, hi = boot_ci(vals, g, n_boot=n_boot, seed=seed)
    return float(vals.mean()), lo, hi, int(len(vals))


def strat_diff(a_vals, b_vals, strat_by, games, nq=10, n_boot=500, seed=0):
    """Paired effect of one condition over another, matched on how much each
    edit flattened the move distribution.

    Two earlier versions of this were no-ops and both shipped a column identical
    to the raw mean. Stratifying and then weighting each stratum by its own
    count reconstructs the overall mean whatever the strata are, so neither
    pre-edit entropy nor change in entropy controlled for anything. Matching has
    to change which positions are counted, not merely how they are grouped.

    strat_by is the pair (delta entropy of a, delta entropy of b). We keep the
    positions where the two edits flattened comparably, so a difference in
    preference there cannot be explained by one edit hedging more.
    """
    a_vals = np.asarray(a_vals, float)
    b_vals = np.asarray(b_vals, float)
    ea = np.asarray(strat_by[0], float)
    eb = np.asarray(strat_by[1], float)
    games = np.asarray(games)

    # Restrict to positions where the two edits flattened the distribution to a
    # comparable degree. Within that subset a difference in preference cannot be
    # attributed to one edit having hedged more than the other.
    gap = np.abs(ea - eb)
    thr = float(np.median(gap))
    m = gap <= thr
    d = (a_vals - b_vals)[m]
    g = games[m]
    if len(d) < 20:
        return float("nan"), float("nan"), float("nan"), 0
    lo, hi = boot_ci(d, g, n_boot=n_boot, seed=seed)
    return float(d.mean()), lo, hi, int(len(d))


def strat_mean(vals, ent, games, nq=10, n_boot=500, seed=0):
    """Retained only for reference; see strat_diff for the real control."""
    vals = np.asarray(vals, float)
    ent = np.asarray(ent, float)
    games = np.asarray(games)
    edges = np.quantile(ent, np.linspace(0, 1, nq + 1))
    edges[-1] += 1e-9
    b = np.clip(np.digitize(ent, edges[1:-1]), 0, nq - 1)

    def agg(sel):
        tot = n = 0.0
        for q in range(nq):
            m = b[sel] == q
            if m.sum():
                tot += vals[sel][m].mean() * m.sum()
                n += m.sum()
        return tot / max(n, 1e-9)

    point = agg(np.arange(len(vals)))
    rng = np.random.default_rng(seed)
    uniq = np.unique(games)
    by = {g: np.where(games == g)[0] for g in uniq}
    bs = []
    for _ in range(n_boot):
        pick = rng.choice(uniq, len(uniq), replace=True)
        bs.append(agg(np.concatenate([by[g] for g in pick])))
    return float(point), float(np.percentile(bs, 2.5)), \
        float(np.percentile(bs, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="attn_8L256_s0")
    ap.add_argument("--n", type=int, default=900)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--mults", type=float, nargs="+",
                    default=[0.25, 0.5, 1.0, 2.0, 4.0])
    ap.add_argument("--minply", type=int, default=20)
    ap.add_argument("--games", type=int, default=2000)
    args = ap.parse_args()

    model, a, ck = build(os.path.join(RUNS, f"{args.name}.pt"))
    res = json.load(open(os.path.join(RES, f"{args.name}.json")))
    best = res["best_layer"]
    pls = torch.load(os.path.join(RES, f"{args.name}_probes.pt"),
                     weights_only=False)
    probe = torch.nn.Linear(a["width"], 64 * 13).cuda()
    probe.load_state_dict(pls[best])
    probe.eval()
    W = probe.weight.reshape(64, 13, a["width"]).float()

    stoi = json.load(open(os.path.join(DATA, "vocab.json")))
    itos = {v: k for k, v in stoi.items()}
    toks, lens, occ = load_split("eval", args.games)
    rng = np.random.default_rng(0)

    rec = {c: {"dr": [], "dpstar": [], "dpill": [], "dent": [],
               "now_legal": [], "flip": []} for c in COND}
    sweep = {m: {"dr": [], "dent": [], "game": []} for m in args.mults}
    ent0, games, base_pre = [], [], []
    n_done = 0

    def forward(x, edit):
        store = {}

        def hook(mod, inp, out):
            o = out
            if edit is not None:
                o = o.clone()
                o[:, -1] = o[:, -1] + edit
            store["h"] = o
            return o

        hd = model.blocks[best].register_forward_hook(hook)
        with torch.amp.autocast("cuda", dtype=torch.float16):
            lg, _, _ = model(x)
        hd.remove()
        h = store["h"][0, -1].float()
        dec = probe(h).reshape(64, 13).argmax(-1)
        return torch.softmax(lg.float()[0, -1], -1), dec

    def direction(pairs, scale):
        """pairs: list of (square, from_class, to_class)."""
        d = torch.zeros(a["width"], device="cuda")
        for sq, c0, c1 in pairs:
            v = (W[sq, c1] - W[sq, c0])
            n = v.norm()
            if float(n) > 1e-6:
                d = d + v / n
        n = d.norm()
        if float(n) < 1e-6:
            return torch.zeros(a["width"], device="cuda")
        return d / n * scale

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
                    uci = itos.get(top1, "")
                    try:
                        m_ill = chess.Move.from_uci(uci)
                        illegal = m_ill not in board.legal_moves
                    except Exception:
                        m_ill, illegal = None, False
                    if m_ill is not None and illegal:
                        truth = np.asarray(occ[gi, t]).astype(int)
                        bel = dec0.cpu().numpy()
                        act = [m_ill.from_square, m_ill.to_square]
                        div_act = [s for s in act if bel[s] != truth[s]]
                        if div_act:
                            lids = [stoi[m.uci()] for m in board.legal_moves
                                    if m.uci() in stoi]
                            if lids:
                                li = torch.tensor(lids, device="cuda")
                                mstar = int(li[p0[li].argmax()])
                                pre_s, pre_i = float(p0[mstar]), float(p0[top1])
                                r_pre = pre_s / (pre_s + pre_i + 1e-12)
                                e_pre = float(-(p0 * (p0 + 1e-12).log()).sum())
                                rms = float(torch.linalg.vector_norm(
                                    probe.weight.new_zeros(1)) * 0 + 1)
                                # residual scale at the edited position
                                with torch.amp.autocast("cuda",
                                                        dtype=torch.float16):
                                    _, hs0, _ = model(x, return_hidden=True)
                                rms = float(hs0[best][0, -1].float()
                                            .pow(2).mean().sqrt())
                                scale = args.alpha * rms * np.sqrt(a["width"])

                                div_all = [s for s in range(64)
                                           if bel[s] != truth[s]]
                                div_irr = [s for s in div_all if s not in act]
                                pairs = {
                                    "correct": [(s, int(bel[s]), int(truth[s]))
                                                for s in div_act],
                                    "irrelevant": [
                                        (s, int(bel[s]), int(truth[s]))
                                        for s in rng.choice(
                                            div_irr,
                                            min(len(div_act), len(div_irr)),
                                            replace=False)]
                                    if div_irr else [],
                                    "wrong_target": [],
                                    "null": [(s, int(bel[s]), int(bel[s]))
                                             for s in act if bel[s] == truth[s]],
                                }
                                others = [s for s in range(64) if s not in act]
                                for s in div_act:
                                    q = int(rng.choice(others))
                                    pairs["wrong_target"].append(
                                        (q, int(bel[q]), int(truth[s])))

                                for c in COND:
                                    if c == "random":
                                        d = direction(pairs["correct"], scale)
                                        if float(d.norm()) > 1e-6:
                                            r_ = torch.randn_like(d)
                                            d = r_ / r_.norm() * d.norm()
                                    else:
                                        d = direction(pairs[c], scale)
                                    p1, dec1 = forward(
                                        x, d if float(d.norm()) > 1e-6 else None)
                                    po_s, po_i = float(p1[mstar]), float(p1[top1])
                                    r_post = po_s / (po_s + po_i + 1e-12)
                                    e_post = float(
                                        -(p1 * (p1 + 1e-12).log()).sum())
                                    nt = int(p1.argmax())
                                    try:
                                        nl = chess.Move.from_uci(
                                            itos.get(nt, "")) in board.legal_moves
                                    except Exception:
                                        nl = False
                                    rec[c]["dr"].append(r_post - r_pre)
                                    rec[c]["dpstar"].append(po_s - pre_s)
                                    rec[c]["dpill"].append(po_i - pre_i)
                                    rec[c]["dent"].append(e_post - e_pre)
                                    rec[c]["now_legal"].append(int(nl))
                                    d1 = dec1.cpu().numpy()
                                    rec[c]["flip"].append(
                                        float(np.mean([d1[s] == truth[s]
                                                       for s in div_act])))

                                for mlt in args.mults:
                                    d = direction(pairs["correct"],
                                                  scale * mlt)
                                    p1, _ = forward(x, d)
                                    po_s, po_i = float(p1[mstar]), float(p1[top1])
                                    sweep[mlt]["dr"].append(
                                        po_s / (po_s + po_i + 1e-12) - r_pre)
                                    sweep[mlt]["dent"].append(
                                        float(-(p1 * (p1 + 1e-12).log()).sum())
                                        - e_pre)
                                    sweep[mlt]["game"].append(gi)

                                ent0.append(e_pre)
                                games.append(gi)
                                base_pre.append(r_pre)
                                n_done += 1
                try:
                    board.push_uci(itos[int(toks[gi, t + 1])])
                except Exception:
                    break

    out = {"name": args.name, "n": n_done, "alpha": args.alpha,
           "n_games": int(len(np.unique(games))),
           "ceiling_illegal_rate": 0.0888,
           "mean_r_pre": float(np.mean(base_pre)), "conditions": {}, "sweep": {}}

    print(f"\n[{args.name}] S1 repair, n={n_done} positions from "
          f"{len(np.unique(games))} games, alpha={args.alpha}")
    print(f"pre-edit forced-choice preference r = {np.mean(base_pre):.4f}\n")
    print(f"{'condition':>13} {'dr (all positions)':>24} "
          f"{'dr where edit did not flatten':>34} "
          f"{'d entropy':>10} {'now legal':>10} {'flip':>6}")
    for c in COND:
        v = rec[c]["dr"]
        lo, hi = boot_ci(v, games)
        noflat = np.asarray(rec[c]["dent"]) <= 0
        nf, nflo, nfhi, nfn = subset_mean(v, noflat, games)
        out["conditions"][c] = {
            "dr_mean": float(np.mean(v)), "dr_lo": lo, "dr_hi": hi,
            "dr_noflatten": nf, "dr_noflatten_lo": nflo,
            "dr_noflatten_hi": nfhi, "n_noflatten": nfn,
            "dpstar": float(np.mean(rec[c]["dpstar"])),
            "dpill": float(np.mean(rec[c]["dpill"])),
            "dentropy": float(np.mean(rec[c]["dent"])),
            "now_legal": float(np.mean(rec[c]["now_legal"])),
            "manipulation_flip": float(np.mean(rec[c]["flip"]))}
        o = out["conditions"][c]
        nfs = (f"{nf:+10.4f} [{nflo:+.4f},{nfhi:+.4f}] n={nfn:<4d}"
               if nfn >= 20 else f"{'too few':>32}")
        print(f"{c:>13} {o['dr_mean']:+8.4f} [{lo:+.4f},{hi:+.4f}] "
              f"{nfs} "
              f"{o['dentropy']:+10.4f} {o['now_legal']:10.3f} "
              f"{o['manipulation_flip']:6.3f}")

    print(f"\nmagnitude sweep, correct repair:")
    print(f"{'mult':>6} {'dr':>22} {'d entropy':>11}")
    for mlt in args.mults:
        v = sweep[mlt]["dr"]
        lo, hi = boot_ci(v, sweep[mlt]["game"])
        out["sweep"][str(mlt)] = {"dr": float(np.mean(v)), "lo": lo, "hi": hi,
                                  "dentropy": float(np.mean(sweep[mlt]["dent"]))}
        print(f"{mlt:6.2f} {np.mean(v):+8.4f} [{lo:+.4f},{hi:+.4f}] "
              f"{np.mean(sweep[mlt]['dent']):+11.4f}")

    cc = out["conditions"]["correct"]
    wt = out["conditions"]["wrong_target"]
    d = np.array(rec["correct"]["dr"]) - np.array(rec["wrong_target"]["dr"])
    lo, hi = boot_ci(d, games)
    out["correct_minus_wrong_target"] = {"mean": float(d.mean()),
                                         "ci_lo": lo, "ci_hi": hi}
    di = np.array(rec["correct"]["dr"]) - np.array(rec["irrelevant"]["dr"])
    lo2, hi2 = boot_ci(di, games)
    out["correct_minus_irrelevant"] = {"mean": float(di.mean()),
                                       "ci_lo": lo2, "ci_hi": hi2}
    # entropy-matched contrast: strata defined by the change in entropy the
    # correct edit produced, contrast taken within stratum.
    #
    # Everything belonging in the saved record must be computed BEFORE the dump
    # below. An earlier version dumped at this point, so the matched contrasts
    # went to stdout and were never persisted. That is invisible when a driver
    # captures and discards stdout, and it would have produced eighteen ladder
    # files missing the comparison the argument rests on.
    sd, sdlo, sdhi, sdn = strat_diff(
        rec["correct"]["dr"], rec["wrong_target"]["dr"],
        (rec["correct"]["dent"], rec["wrong_target"]["dent"]), games)
    out["correct_minus_wrong_target_entropy_matched"] = {
        "mean": sd, "ci_lo": sdlo, "ci_hi": sdhi, "n": sdn}
    sdi, sdilo, sdihi, sdin = strat_diff(
        rec["correct"]["dr"], rec["irrelevant"]["dr"],
        (rec["correct"]["dent"], rec["irrelevant"]["dent"]), games)
    out["correct_minus_irrelevant_entropy_matched"] = {
        "mean": sdi, "ci_lo": sdilo, "ci_hi": sdihi, "n": sdin}
    out["raw"] = {c: {k: list(map(float, rec[c][k]))
                      for k in ("dr", "dent", "now_legal")} for c in COND}
    out["raw"]["game"] = list(map(int, games))
    json.dump(out, open(os.path.join(RES, f"{args.name}_repair.json"), "w"),
              indent=1)

    print(f"\nKILL CRITERION 1  correct - wrong_target = {d.mean():+.4f} "
          f"[{lo:+.4f}, {hi:+.4f}]  -> "
          f"{'PASS' if lo > 0 else 'FAIL'}")
    print(f"                  entropy-matched          {sd:+.4f} "
          f"[{sdlo:+.4f}, {sdhi:+.4f}] n={sdn}  -> "
          f"{'PASS' if sdlo > 0 else 'FAIL'}")
    print(f"KILL CRITERION 2  correct, no flattening = {cc['dr_noflatten']:+.4f} "
          f"[{cc['dr_noflatten_lo']:+.4f}, {cc['dr_noflatten_hi']:+.4f}] "
          f"n={cc['n_noflatten']}  -> "
          f"{'PASS' if cc['dr_noflatten_lo'] > 0 else 'FAIL'}")
    print(f"LOCALITY (S3)     correct - irrelevant  = {di.mean():+.4f} "
          f"[{lo2:+.4f}, {hi2:+.4f}]")
    print(f"                  entropy-matched          {sdi:+.4f} "
          f"[{sdilo:+.4f}, {sdihi:+.4f}] n={sdin}")


if __name__ == "__main__":
    main()
