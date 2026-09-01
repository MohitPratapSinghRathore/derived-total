"""Domain 2: permutation tracking over a SOLVABLE group, with mixed operations.

Design history, recorded because it is the useful part. Four arithmetic designs
floored because they required modular arithmetic before any state tracking became
visible. A fifth, permutation tracking over five objects, also floored, and that
failure was more informative: with five objects the state is an element of S5,
the smallest non-solvable symmetric group, which is exactly the case the theory
of bounded-depth sequence models says cannot be tracked. We had chosen the
hardest possible instance.

Three principled changes follow.

  FOUR OBJECTS. S4 is solvable, with the series S4 > A4 > V4 > 1, so bounded-depth
  composition is tractable in principle rather than excluded by theory.

  MIXED OPERATIONS. A SWAP names two boxes and moves their contents without
  naming the objects, so the affected object's new location cannot be retrieved
  by attending to its last mention: it must be maintained. A MOVE names the
  object, giving an anchor that makes the task learnable. Mixing the two means
  the task rewards maintenance while remaining learnable, and the two operation
  types can be analysed separately.

  ANSWER WEIGHTING. Query answers are a small share of tokens, so the quantity of
  interest contributes little gradient. The training loss upweights answer
  positions. This changes the optimisation, not the task.

  [BOS] (box_i obj_j){N}
        ( SWAP box_a box_b | MOVE obj_x box_y ){r}
        (QUERY obj_x box_y)
        ...

Exact per-token oracle: the box of every object after every token.
"""
import numpy as np

NOBJ = 4                      # S4 is solvable; S5 is not
MAX_OPS = 24
# Query after EVERY operation. Six earlier designs floored, and the common cause
# was that all of them were all-or-nothing: with uniformly random updates and
# sparse queries, a model either tracks the state exactly or sits at chance, the
# same structure as parity, which is hard to learn from endpoint supervision.
# Chess trains because it gives partial credit: a plausible move needs only
# partial state, so there is a gradient of partial competence to climb. Querying
# after every operation supplies the same thing here. Depth-one queries are
# answerable directly, which gives the model a foothold from which to extend.
MIN_GAP, MAX_GAP = 1, 1
P_SWAP = 0.5                  # the rest are anchored MOVEs

TOK = {"<pad>": 0, "<bos>": 1, "<swap>": 2, "<move>": 3, "<query>": 4}
for i in range(NOBJ):
    TOK[f"box{i}"] = len(TOK)
for i in range(NOBJ):
    TOK[f"obj{i}"] = len(TOK)
VOCAB = len(TOK)
BOX0 = TOK["box0"]
OBJ0 = TOK["obj0"]

MAX_QUERIES = MAX_OPS // MIN_GAP + 1
SEQ = 1 + 2 * NOBJ + 3 * MAX_OPS + 3 * MAX_QUERIES


def gen(n, seed=0):
    """tokens, lens, state[obj]->box, per-query (answer pos, depth, object),
    and per-token answer weights for the training loss."""
    rng = np.random.default_rng(seed)
    toks = np.zeros((n, SEQ), dtype=np.int16)
    state = np.zeros((n, SEQ, NOBJ), dtype=np.uint8)
    lens = np.zeros(n, dtype=np.int32)
    qpos = np.full((n, MAX_QUERIES), -1, dtype=np.int32)
    qdepth = np.full((n, MAX_QUERIES), -1, dtype=np.int32)
    qobj = np.full((n, MAX_QUERIES), -1, dtype=np.int8)

    for i in range(n):
        contents = rng.permutation(NOBJ)          # contents[box] = object
        where = np.zeros(NOBJ, dtype=np.int64)
        for b, o in enumerate(contents):
            where[o] = b

        p = 0
        toks[i, p] = TOK["<bos>"]; state[i, p] = where; p += 1
        for b in range(NOBJ):
            toks[i, p] = BOX0 + b
            toks[i, p + 1] = OBJ0 + int(contents[b])
            state[i, p:p + 2] = where
            p += 2

        ops = 0
        qi = 0
        while ops < MAX_OPS:
            gap = min(int(rng.integers(MIN_GAP, MAX_GAP + 1)), MAX_OPS - ops)
            for _ in range(gap):
                if rng.random() < P_SWAP:
                    a, b = rng.choice(NOBJ, size=2, replace=False)
                    toks[i, p] = TOK["<swap>"]
                    toks[i, p + 1] = BOX0 + int(a)
                    toks[i, p + 2] = BOX0 + int(b)
                else:
                    x = int(rng.integers(NOBJ))
                    a = int(where[x])
                    b = int(rng.choice([c for c in range(NOBJ) if c != a]))
                    toks[i, p] = TOK["<move>"]
                    toks[i, p + 1] = OBJ0 + x
                    toks[i, p + 2] = BOX0 + b
                state[i, p:p + 2] = where
                oa, ob = contents[a], contents[b]
                contents[a], contents[b] = ob, oa
                where = where.copy()
                where[oa], where[ob] = b, a
                state[i, p + 2] = where
                p += 3
                ops += 1
            x = int(rng.integers(NOBJ))
            toks[i, p] = TOK["<query>"]
            toks[i, p + 1] = OBJ0 + x
            toks[i, p + 2] = BOX0 + int(where[x])
            state[i, p:p + 3] = where
            qpos[i, qi] = p + 2
            qdepth[i, qi] = ops
            qobj[i, qi] = x
            qi += 1
            p += 3
        lens[i] = p
    return toks, lens, state, qpos, qdepth, qobj


def answer_weights(toks, lens, qpos, weight=5.0):
    """Per-token loss weight: answer positions carry the signal of interest."""
    w = np.ones(toks.shape, dtype=np.float32)
    for i in range(len(toks)):
        for q in qpos[i]:
            if q >= 0:
                w[i, int(q)] = weight
    return w


if __name__ == "__main__":
    t, l, s, qp, qd, qo = gen(3, seed=0)
    inv = {v: k for k, v in TOK.items()}
    print("vocab", VOCAB, "seq", SEQ, "chance", 1 / NOBJ, "len", l[0])
    print(" ".join(inv[x] for x in t[0][:32]))
    print("query depths", qd[0][qd[0] > 0])
    ok = True
    for k in range(int((qd[0] > 0).sum())):
        pos, x = int(qp[0, k]), int(qo[0, k])
        ok &= (int(t[0, pos]) - BOX0) == int(s[0, pos, x])
    print("all queries consistent with oracle:", bool(ok))
    # a bijection must be preserved at every step
    valid = all(sorted(s[0, u]) == list(range(NOBJ)) for u in range(int(l[0])))
    print("state is a bijection at every token:", valid)
