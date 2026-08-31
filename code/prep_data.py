"""Stage 1: PGN -> per-game UCI token sequences, split by game.

Splits are by GAME, never by position, so no prefix leaks between the
probe-fitting split and the eval split (protocol/experiments.md sec.1).
"""
import io, os, re, sys, json, glob, hashlib, random
import zstandard as zstd
import chess, chess.pgn
import numpy as np

DATA = os.path.join(os.path.dirname(__file__), "..", "data")
ELO_FLOOR = 1600
MIN_PLIES = 30
MAX_PLIES = 160
MAX_GAMES = int(sys.argv[1]) if len(sys.argv) > 1 else 60000

hdr_re = re.compile(r'\[(\w+) "([^"]*)"\]')


def games(path):
    dctx = zstd.ZstdDecompressor()
    with open(path, "rb") as fh:
        stream = dctx.stream_reader(fh)
        text = io.TextIOWrapper(stream, encoding="utf-8", errors="ignore")
        while True:
            g = chess.pgn.read_game(text)
            if g is None:
                return
            yield g


def main():
    srcs = sorted(glob.glob(os.path.join(DATA, "*.pgn.zst")))
    print("sources:", [os.path.basename(x) for x in srcs])
    seen = set()
    out = []
    kept = scanned = 0
    def all_games():
        for s_ in srcs:
            for g_ in games(s_):
                yield g_

    for g in all_games():
        scanned += 1
        if scanned % 20000 == 0:
            print(f"  scanned {scanned} kept {kept}", flush=True)
        h = g.headers
        if h.get("Variant", "Standard") != "Standard":
            continue
        try:
            we, be = int(h.get("WhiteElo", 0)), int(h.get("BlackElo", 0))
        except ValueError:
            continue
        if we < ELO_FLOOR or be < ELO_FLOOR:
            continue
        board = g.board()
        if board.fen() != chess.STARTING_FEN:
            continue
        ucis = []
        for mv in g.mainline_moves():
            ucis.append(mv.uci())
            board.push(mv)
            if len(ucis) >= MAX_PLIES:
                break
        if len(ucis) < MIN_PLIES:
            continue
        key = hashlib.md5(" ".join(ucis).encode()).hexdigest()
        if key in seen:            # dedup by move-sequence hash
            continue
        seen.add(key)
        out.append(ucis)
        kept += 1
        if kept >= MAX_GAMES:
            break

    print(f"scanned {scanned}, kept {kept}")

    # vocab over observed UCI moves
    vocab = sorted({m for g in out for m in g})
    stoi = {"<pad>": 0, "<bos>": 1}
    for m in vocab:
        stoi[m] = len(stoi)
    print("vocab", len(stoi))

    rng = random.Random(0)
    rng.shuffle(out)
    n = len(out)
    # disjoint game-level splits
    splits = {
        "train_lm":    out[: int(0.80 * n)],
        "train_probe": out[int(0.80 * n): int(0.90 * n)],
        "eval":        out[int(0.90 * n):],
    }
    os.makedirs(os.path.join(DATA, "proc"), exist_ok=True)
    json.dump(stoi, open(os.path.join(DATA, "proc", "vocab.json"), "w"))
    for name, gs in splits.items():
        arr = np.zeros((len(gs), MAX_PLIES + 1), dtype=np.int16)
        lens = np.zeros(len(gs), dtype=np.int16)
        for i, g in enumerate(gs):
            ids = [stoi["<bos>"]] + [stoi[m] for m in g]
            arr[i, : len(ids)] = ids
            lens[i] = len(ids)
        np.savez_compressed(os.path.join(DATA, "proc", f"{name}.npz"), toks=arr, lens=lens)
        print(name, arr.shape)


if __name__ == "__main__":
    main()
