"""Re-run every adapter check and record it, so the paper can cite a results file.

Each external model reaches our measurements through code whose failure mode is
silent: a wrong vocabulary, an off-by-one in move alignment, or a missed weight
transpose all produce plausible output rather than an error. These checks were
run during development and printed to the console; this script repeats them and
writes results/adapter_checks.json so the numbers in the methods text have
provenance like every other number in the paper.
"""
import os, json, sys, time
import numpy as np
import torch
import chess

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
RES = os.path.join(HERE, "..", "results")
EXT = os.path.join(HERE, "..", "data", "ext")


def main():
    out = {}

    # 1. character vocabulary: reconstructed, so it must play legal chess
    from chessgpt import load, sanity_check, encode, ITOS, STOI
    model, cfg, _ = load()
    res = sanity_check(model)
    out["vocab_openings_legal"] = sum(1 for *_, ok in res if ok)
    out["vocab_openings_total"] = len(res)
    out["vocab_size"] = int(cfg.vocab_size)
    tied = model.lm_head.weight.data_ptr() == model.transformer.wte.weight.data_ptr()
    out["weights_tied"] = bool(tied)

    # 2. move-boundary alignment, three independent checks per site
    from chessgpt_data import build, verify, load_games
    games = load_games("eval", 60)
    sites = probs = 0
    for g in games:
        text, s = build(g, max_plies=80)
        probs += len(verify(g, text, s))
        sites += len(s)
    out["align_sites"] = sites
    out["align_problems"] = probs

    # 3. nanoGPT conversion: legal free play for trained, not for random-init
    from chessgpt_nano import convert, free_play
    for f, tag in (("lichess_6layers_ckpt_no_optimizer.pt", "nano6"),
                   ("lichess_8layers_ckpt_no_optimizer.pt", "nano8"),
                   ("lichess_16layers_ckpt_no_optimizer.pt", "nano16"),
                   ("randominit_8layers_ckpt.pt", "nanoRandom")):
        p = os.path.join(EXT, f)
        if not os.path.exists(p):
            continue
        m, c, _ = convert(p)
        legal, total, _ = free_play(m, plies=20)
        out[f"{tag}_legal"] = legal
        out[f"{tag}_total"] = total
        del m
        torch.cuda.empty_cache()

    # 4. the cached decoder must reproduce the uncached one exactly
    from chessgpt_structure import decode_game

    @torch.no_grad()
    def slow(m, ids, upto, max_chars=8):
        ctx = torch.tensor([ids[:upto + 1][-1000:]], dtype=torch.long, device="cuda")
        o = m(ctx, use_cache=True)
        past, nxt, ch = o.past_key_values, int(o.logits[0, -1].argmax()), []
        for _ in range(max_chars):
            c = ITOS.get(nxt, "")
            if c in (" ", ";"):
                break
            ch.append(c)
            st = m(torch.tensor([[nxt]], device="cuda"), past_key_values=past,
                   use_cache=True)
            past, nxt = st.past_key_values, int(st.logits[0, -1].argmax())
        return "".join(ch)

    n = mism = 0
    for g in load_games("eval", 8):
        text, s = build(g, max_plies=120)
        ids = encode(text)
        want = [x["char_index"] for x in s if x["ply"] >= 20]
        fast = decode_game(model, ids, want)
        for w in want:
            n += 1
            mism += int(fast[w] != slow(model, ids, w))
    out["cache_sites"] = n
    out["cache_mismatches"] = mism

    # 5. SAN classifier self-test
    import io, contextlib
    from san_classify import selftest
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        ok = selftest()
    lines = [l for l in buf.getvalue().splitlines() if l.strip().startswith(("ok", "BAD"))]
    out["san_cases"] = len(lines)
    out["san_correct"] = sum(1 for l in lines if l.strip().startswith("ok"))

    # 6. OthelloGPT forward and engine
    from othello import load as oload, verify as overify, start_board, legal_moves, BLACK
    out["othello_opening_ok"] = sorted(legal_moves(start_board(), BLACK)) == [19, 26, 37, 44]
    om = oload()
    ok_, tot_ = overify(om)
    out["othello_legal"] = ok_
    out["othello_total"] = tot_

    out["timestamp"] = time.strftime("%Y-%m-%d %H:%M")
    json.dump(out, open(os.path.join(RES, "adapter_checks.json"), "w"), indent=1)
    for k, v in out.items():
        print(f"  {k:22s} {v}")


if __name__ == "__main__":
    main()
