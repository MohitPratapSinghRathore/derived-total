"""Domain 2: long-horizon variable-state tracking (a stand-in for multi-step maths).

A program assigns and updates 8 integer variables (mod 100) over many steps,
then asks for a value that requires COMBINING two of them:

    [BOS] (tgt src op const)*n  QUERY va vb  <answer>
    answer = (val[va] + val[vb]) mod 100

Why this domain earns its place next to chess:
  * exact state oracle at every token (the 8 variable values), as in chess;
  * but the state is arithmetic, not spatial, and updates are non-local in a
    different way -- so it is a genuinely different task, not chess relabelled;
  * the answer needs a computation ON TOP of the retrieved state, so
    E_state (did it keep the values?) and E_plan (did it combine them right?)
    stay separable exactly as in the chess arm;
  * depth is the number of steps, a clean horizon axis.

This is NOT the GSM8K transfer experiment of the paper's H5, which needs
general-corpus pretraining well beyond this hardware.  It tests the weaker,
still-meaningful claim: does the architectural fix generalise beyond chess to
another long-horizon state-tracking domain?
"""
import os, json
import numpy as np

NVAR = 4
MOD = 20
MAX_STEPS = 32
OPS = ["+", "-"]          # multiplication mod 100 makes values jump discontinuously
                          # and pins small models at the floor; additive updates
                          # keep the task learnable while still requiring exact
                          # long-horizon tracking

# vocab: BOS, QUERY, VAR_0..7 (as targets/sources), OP_0..2, CONST_0..9,
#        VAL_0..99 (answers), PAD
TOK = {"<pad>": 0, "<bos>": 1, "<query>": 2}
for i in range(NVAR):
    TOK[f"v{i}"] = len(TOK)
for o in range(len(OPS)):
    TOK[f"op{o}"] = len(TOK)
for c in range(10):
    TOK[f"c{c}"] = len(TOK)
for v in range(MOD):
    TOK[f"val{v}"] = len(TOK)
VOCAB = len(TOK)
SEQ = 1 + 4 * MAX_STEPS + 4          # bos + steps + query va vb ans


def gen(n, seed=0, min_steps=5):
    """Returns tokens (n,SEQ), lens, state (n,SEQ,NVAR), anspos, answers."""
    rng = np.random.default_rng(seed)
    toks = np.zeros((n, SEQ), dtype=np.int16)
    state = np.zeros((n, SEQ, NVAR), dtype=np.uint8)
    lens = np.zeros(n, dtype=np.int32)
    anspos = np.zeros(n, dtype=np.int32)
    for i in range(n):
        nsteps = int(rng.integers(min_steps, MAX_STEPS + 1))
        val = np.zeros(NVAR, dtype=np.int64)
        # seed the variables so early steps are not all zeros
        for v in range(NVAR):
            val[v] = int(rng.integers(0, MOD))
        p = 0
        toks[i, p] = TOK["<bos>"]; state[i, p] = val; p += 1
        for _ in range(nsteps):
            tgt = int(rng.integers(0, NVAR))
            src = int(rng.integers(0, NVAR))
            op = int(rng.integers(0, len(OPS)))
            c = int(rng.integers(0, 10))
            toks[i, p:p + 4] = [TOK[f"v{tgt}"], TOK[f"v{src}"],
                                TOK[f"op{op}"], TOK[f"c{c}"]]
            # state is unchanged while the step is being read ...
            state[i, p:p + 3] = val
            if op == 0:
                nv = (val[src] + c) % MOD
            elif op == 1:
                nv = (val[src] - c) % MOD
            else:
                nv = (val[src] * c) % MOD
            val = val.copy(); val[tgt] = nv
            state[i, p + 3] = val          # ... and updates on the step's last token
            p += 4
        va, vb = int(rng.integers(0, NVAR)), int(rng.integers(0, NVAR))
        ans = int((val[va] + val[vb]) % MOD)
        toks[i, p] = TOK["<query>"]; state[i, p] = val
        toks[i, p + 1] = TOK[f"v{va}"]; state[i, p + 1] = val
        toks[i, p + 2] = TOK[f"v{vb}"]; state[i, p + 2] = val
        toks[i, p + 3] = TOK[f"val{ans}"]; state[i, p + 3] = val
        anspos[i] = p + 3
        lens[i] = p + 4
    return toks, lens, state, anspos


def steps_of(pos):
    """token position -> number of completed steps (the depth axis)"""
    return np.maximum(0, (pos - 1) // 4)


if __name__ == "__main__":
    t, l, s, a = gen(3, seed=0)
    inv = {v: k for k, v in TOK.items()}
    print("vocab", VOCAB, "seq", SEQ)
    print(" ".join(inv[x] for x in t[0][:l[0]]))
    print("state@end", s[0, l[0] - 1])
    print("answer tok", inv[t[0, a[0]]])
