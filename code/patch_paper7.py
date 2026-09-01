"""Write in the planning-error decomposition and the domain-2 outcome."""
import os

p = os.path.join(os.path.dirname(__file__), "..", "paper", "main.tex")
s = open(p, encoding="utf-8").read()


def once(old, new):
    global s
    assert old in s, old[:70]
    s = s.replace(old, new, 1)


PLAN = r"""\subsection{How much of the planning error is really memory}
\label{sec:plan}

The other half of the decomposition needs care, because the conditioning that
would isolate it cleanly is unattainable. Requiring the probe to recover the
exact board leaves no samples past the opening, for the reason already given.
Requiring only the squares the move touches leaves samples everywhere but does
not constrain the rest of the board, so state loss elsewhere is still counted as
a failure of judgement.

We therefore vary the strictness and read the trend (Table~\ref{tab:strict}).
Tightening the conditioning lowers the loss substantially. At ply 30 to 40 the
median centipawn loss falls from \result{epsMed30to40tol64} with the board
unconstrained to \result{epsMed30to40tol4} when at most four other squares may be
wrong, and the blunder rate from \result{epsBl30to40tol64} to
\result{epsBl30to40tol4}. Roughly half of what a naive reading would call
degraded judgement is misattributed memory failure.

The trend does not go to zero. Holding the conditioning fixed at four wrong
squares, the median loss still grows with depth, from
\result{epsMed20to30tol4} at ply 20 to 30 to \result{epsMed30to40tol4} at ply 30
to 40. Judgement given intact state is not depth-invariant.

We take the honest reading. Memory is the dominant component and the better
characterised one, and it is not the only one. Our original framing of H4, a
clean separation in which an intervention moves one term and leaves the other
untouched, was too tidy for what the measurement supports. The decomposition
still earns its place, because it converts one uninterpretable curve into two
quantities with different remedies, but a practitioner should expect the terms to
be entangled rather than orthogonal, and should report the conditioning they used
when they quote either one.
"""

BOXES = r"""
The outcome is not the clean replication we hoped for, and it is more
informative than one would have been (Table~\ref{tab:boxes}).

With the supervised recurrent channel the task is close to solved at every depth,
\result{ansDeepBoxAlsb} at the deepest bucket against a chance rate of
\result{chanceBoxAlsb}, the probe recovers the queried object at
\result{probeDeepBoxAlsb}, and belief consistency is \result{beliefBoxAlsb}
against a mismatched control of \result{beliefMisBoxAlsb}. The mechanism
reproduces.

The attention-only model behaves differently. It solves the task well above
chance, \result{ansDeepBoxAttn} at depth, while the probe recovers its state at
\result{probeDeepBoxAttn}, which is chance. Belief consistency is
\result{beliefBoxAttn} against a control of \result{beliefMisBoxAttn}, which is
no signal at all. The same probe procedure, on the same positions, recovers state
at \result{probeDeepBoxAlsb} in the other model, so this is a fact about the
model and not a failure of the probe.

The reading we prefer is that this task does not force the model to maintain
state. Only one object is asked about at a time, so the answer can be resolved on
demand by tracing that object's history, and an explicit assignment carried
forward is unnecessary. Chess allows no such shortcut: predicting a legal move
requires the whole position, so the state must be maintained eagerly.

That gives a fourth lesson for anyone building these diagnostics, and it is the
one we would put first. A task can carry an exact oracle, be learnable, and still
fail to measure state maintenance, because it never obliges the model to
maintain anything. Belief consistency is only defined where a belief exists to
read, and whether one exists is a property of the task, not only of the
architecture.
"""

once(r"""\subsection{A degenerate metric, declared}""",
     PLAN + "\n" + r"""\subsection{A degenerate metric, declared}""")

once(r"""We report this in
place of a cross-domain result, which we do not have.""",
     r"""We report this in
place of an easy cross-domain result.
""" + BOXES)

open(p, "w", encoding="utf-8").write(s)
print("patched: planning decomposition and domain 2 outcome")
