"""Domain 2: long-horizon variable-state tracking with interleaved queries.

A program updates NVAR integer variables modulo MOD.  Every few steps it is
interrupted by a query asking for (v_a + v_b) mod MOD, answerable only by having
maintained the values.  Queries are interleaved rather than placed once at the
end, for a reason worth recording:

An earlier version put a single query at the end.  The program steps are random
by construction, so nothing about them is predictable, and the single answer
token carried under one percent of the gradient.  Training loss sat at the
token-type entropy of the generator, about 1.44, and no state was ever learned.
That is a defect of the task, not of the model: a task whose learnable signal is
one token in 133 measures nothing.  Interleaving queries gives dense supervision
and, usefully, several depth measurements per sequence.

Layout:  [BOS] (tgt src op const){r}  (QUERY va vb ans)  (tgt src op const){r} ...

Exact per-token oracle: the NVAR variable values after every token.
"""
import numpy as np

NVAR = 4
MOD = 10
MAX_STEPS = 32
MIN_GAP, MAX_GAP = 3, 5      # program steps between successive queries
OPS = ["+", "-"]             # additive updates keep the task learnable while
                             # still requiring exact long-horizon tracking

TOK = {"<pad>": 0, "<bos>": 1, "<query>": 2, "<queryread>": 3}
for i in range(NVAR):
    TOK[f"v{i}"] = len(TOK)
for o in range(len(OPS)):
    TOK[f"op{o}"] = len(TOK)
for c in range(10):
    TOK[f"c{c}"] = len(TOK)
for v in range(MOD):
    TOK[f"val{v}"] = len(TOK)
VOCAB = len(TOK)

MAX_QUERIES = MAX_STEPS // MIN_GAP + 1
SEQ = 1 + 4 * MAX_STEPS + 4 * MAX_QUERIES


def gen(n, seed=0):
    """tokens, lens, state, per-query (answer pos, depth, va, vb), query kind."""
    rng = np.random.default_rng(seed)
    toks = np.zeros((n, SEQ), dtype=np.int16)
    state = np.zeros((n, SEQ, NVAR), dtype=np.uint8)
    lens = np.zeros(n, dtype=np.int32)
    qpos = np.full((n, MAX_QUERIES), -1, dtype=np.int32)
    qdepth = np.full((n, MAX_QUERIES), -1, dtype=np.int32)
    qvars = np.full((n, MAX_QUERIES, 2), -1, dtype=np.int8)
    qkind = np.zeros((n, MAX_QUERIES), dtype=np.int8)   # 0 read, 1 sum

    for i in range(n):
        val = rng.integers(0, MOD, NVAR).astype(np.int64)
        p = 0
        toks[i, p] = TOK["<bos>"]; state[i, p] = val; p += 1
        steps = 0
        qi = 0
        while steps < MAX_STEPS:
            gap = min(int(rng.integers(MIN_GAP, MAX_GAP + 1)), MAX_STEPS - steps)
            for _ in range(gap):
                tgt = int(rng.integers(0, NVAR))
                src = int(rng.integers(0, NVAR))
                op = int(rng.integers(0, len(OPS)))
                c = int(rng.integers(0, 10))
                toks[i, p:p + 4] = [TOK[f"v{tgt}"], TOK[f"v{src}"],
                                    TOK[f"op{op}"], TOK[f"c{c}"]]
                state[i, p:p + 3] = val          # unchanged while the step is read
                nv = (val[src] + c) % MOD if op == 0 else (val[src] - c) % MOD
                val = val.copy(); val[tgt] = nv
                state[i, p + 3] = val            # updates on the step's last token
                p += 4
                steps += 1
            va, vb = int(rng.integers(0, NVAR)), int(rng.integers(0, NVAR))
            # two query kinds. READ asks for a stored value and isolates state
            # tracking; SUM asks for a computation over two stored values and so
            # keeps a planning step that can fail with the state intact.
            if rng.random() < 0.5:
                kind = 0                      # READ
                ans = int(val[va] % MOD)
                toks[i, p] = TOK["<queryread>"]
                toks[i, p + 1] = TOK[f"v{va}"]
                toks[i, p + 2] = TOK[f"v{va}"]
                toks[i, p + 3] = TOK[f"val{ans}"]
            else:
                kind = 1                      # SUM
                ans = int((val[va] + val[vb]) % MOD)
                toks[i, p] = TOK["<query>"]
                toks[i, p + 1] = TOK[f"v{va}"]
                toks[i, p + 2] = TOK[f"v{vb}"]
                toks[i, p + 3] = TOK[f"val{ans}"]
            qkind[i, qi] = kind
            state[i, p:p + 4] = val
            qpos[i, qi] = p + 3
            qdepth[i, qi] = steps
            qvars[i, qi] = (va, vb)
            qi += 1
            p += 4
        lens[i] = p
    return toks, lens, state, qpos, qdepth, qvars, qkind


if __name__ == "__main__":
    t, l, s, qp, qd, qv, qk = gen(2, seed=0)
    inv = {v: k for k, v in TOK.items()}
    print("vocab", VOCAB, "seq", SEQ, "max queries", MAX_QUERIES,
          "chance answer", 1 / MOD)
    print("len", l[0])
    print(" ".join(inv[x] for x in t[0][:44]))
    print("query depths", qd[0][qd[0] > 0])
    for k in range(3):
        pos = qp[0, k]; va, vb = qv[0, k]
        got = inv[t[0, pos]]
        want = (int(s[0, pos, va]) + int(s[0, pos, vb])) % MOD
        print(f"  q{k}: v{va}={s[0,pos,va]} v{vb}={s[0,pos,vb]} -> {got} (expect val{want})")
