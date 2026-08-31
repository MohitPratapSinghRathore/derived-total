"""One-off: shrink synthetic-domain memory (int64 -> uint8/int16).

state is (N, SEQ, NVAR) with values in [0,100) and toks has a 124-token vocab;
holding both as int64 costs ~8x more RAM than needed and does not fit the full
200k-sequence training set alongside the chess grid.
"""
import os, re

d = os.path.dirname(__file__)

p = os.path.join(d, "synth.py")
s = open(p, encoding="utf-8").read()
s = s.replace("toks = np.zeros((n, SEQ), dtype=np.int64)",
              "toks = np.zeros((n, SEQ), dtype=np.int16)")
s = s.replace("state = np.zeros((n, SEQ, NVAR), dtype=np.int64)",
              "state = np.zeros((n, SEQ, NVAR), dtype=np.uint8)")
s = s.replace("lens = np.zeros(n, dtype=np.int64)",
              "lens = np.zeros(n, dtype=np.int32)")
s = s.replace("anspos = np.zeros(n, dtype=np.int64)",
              "anspos = np.zeros(n, dtype=np.int32)")
open(p, "w", encoding="utf-8").write(s)
print("synth.py: compact dtypes")

p = os.path.join(d, "synth_run.py")
s = open(p, encoding="utf-8").read()
# every tensor handoff must widen back to int64 for torch indexing / CE targets
s = s.replace("x = torch.from_numpy(toks[idx]).to(dev)",
              "x = torch.from_numpy(toks[idx].astype(np.int64)).to(dev)")
s = s.replace("s = torch.from_numpy(state[idx]).to(dev)[:, :-1]",
              "s = torch.from_numpy(state[idx].astype(np.int64)).to(dev)[:, :-1]")
s = s.replace("s = torch.from_numpy(state[idx]).to(dev)",
              "s = torch.from_numpy(state[idx].astype(np.int64)).to(dev)")
s = s.replace("pad = torch.from_numpy(np.arange(SEQ)[None] >= lens[idx][:, None]).to(dev)",
              "pad = torch.from_numpy(np.arange(SEQ)[None] >= lens[idx][:, None].astype(np.int64)).to(dev)")
s = s.replace("v = torch.from_numpy(np.arange(SEQ)[None] < lens[idx][:, None]).to(dev)",
              "v = torch.from_numpy(np.arange(SEQ)[None] < lens[idx][:, None].astype(np.int64)).to(dev)")
# anspos arithmetic must not overflow int16 comparisons
s = s.replace("ap = anspos[idx]", "ap = anspos[idx].astype(np.int64)")
s = s.replace("ap = anspos[idx]; qpos = ap - 3",
              "ap = anspos[idx].astype(np.int64); qpos = ap - 3")
s = s.replace("true_ans = toks[idx, ap]", "true_ans = toks[idx, ap].astype(np.int64)")
open(p, "w", encoding="utf-8").write(s)
print("synth_run.py: widen at tensor boundaries")
