"""Revision eight: the drift control, and the language it licenses."""
import os

p = os.path.join(os.path.dirname(__file__), "..", "paper", "main.tex")
s = open(p, encoding="utf-8").read()


def once(old, new, label=""):
    global s
    assert old in s, "NOT FOUND " + label + ": " + old[:70]
    s = s.replace(old, new, 1)


SEC = r"""\subsection{Lost, or merely moved? Depth-local probes}
\label{sec:drift}

Every fidelity number so far comes from one decoder fitted across all depths, and
a falling number has been read as the state becoming less available. A single
global decoder cannot distinguish that from an alternative: the state might stay
equally present while its linear basis moves with depth, so that one fixed
decoder simply transfers badly. The reversible-cycle experiment holds the
position fixed but still uses the global probe, so it inherits the same
ambiguity.

We fit a separate linear probe inside each depth bucket and evaluate every probe
on every bucket.

\input{tab_drift}

The diagonal answers the question. A probe fitted for the deepest bucket and
tested there reaches \result{drDiagLast}, against \result{drDiagFirst} for the
shallowest bucket tested on itself. Depth-local probes collapse in almost the
same way the global probe does, \result{drPoolFirst} to \result{drPoolLast}, so
the decline is not an artifact of a fixed decoder failing to keep up. If the
representation merely rotated, a probe fitted at the relevant depth would recover
what a global probe misses, and it does not.

Basis change is nonetheless present in the off-diagonal. A probe fitted on the
shallowest positions reaches only \result{drFarOff} on the deepest ones, far
below what a probe fitted there achieves, so the encoding does move with depth.
It moves and it also thins, and the thinning is what the paper's curves track.

One caveat cuts in our favour rather than against. Each depth-local probe sees
roughly a sixth of the data the pooled probe sees, which is why the pooled probe
edges the diagonal at depth. More data would help the local probes, so this is a
conservative test of the drift account: it would have to overturn a gap that
already runs the wrong way for it.

"""

once(r"""\subsection{Which squares, not how many}""",
     SEC + r"""\subsection{Which squares, not how many}""", "drift section")

open(p, "w", encoding="utf-8").write(s)
print("patched: drift section and decay language")
