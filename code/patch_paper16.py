"""Revision three: the two real concrete errors, and the square controls."""
import os

p = os.path.join(os.path.dirname(__file__), "..", "paper", "main.tex")
s = open(p, encoding="utf-8").read()


def once(old, new, label=""):
    global s
    assert old in s, "NOT FOUND " + label + ": " + old[:60]
    s = s.replace(old, new, 1)


# ------------------------------- the "roughly ply 0" absurdity in the metric note
once(r"""Our horizon was pre-registered on exact-position fidelity. That quantity reaches
zero by roughly ply \result{exactPly} in every condition, so it cannot
discriminate between them, and conditioning on it is impossible because exact
states essentially do not occur past the opening.""",
     r"""Our horizon was pre-registered on exact-position fidelity, the probability that
all sixty-four squares are simultaneously recovered. That criterion is already
below its threshold in the very first bucket, so the pre-registered horizon
evaluates to zero ply for every condition and cannot discriminate between them.
The underlying quantity falls to \result{exactFirst} in the opening and to
\result{exactMid} by ply 20 to 30, after which exact states essentially do not
occur, which is also why conditioning on them yields no samples at depth.""",
     "exact ply")

# ------------------------------------------- the square controls, as a subsection
once(r"""\subsection{The errors are coherent}""",
     r"""\subsection{Which squares, not how many}
\label{sec:sqcontrol}

Move-touched fidelity scores two squares where the aggregate averages
sixty-four, and a statistic over two concentrated variables can look sharper than
one over sixty-four diluted ones for reasons unrelated to relevance. The control
that separates those explanations is a random pair.

\input{tab_sqcontrol}

Two randomly chosen squares predict an illegal move at an AUC of
\result{sqTwoRandom}, well below the whole board at \result{sqWholeBoard}. Two
squares drawn at random from those actually \emph{occupied}, which matches the
occupancy profile of a move's origin and destination, reach
\result{sqTwoRandomOcc}, statistically indistinguishable from the whole board.
The squares a move touches reach \result{sqMoveTouched}, with a game-clustered
interval that does not overlap either control. The advantage is therefore about
which squares are read, not how many, and it is not an artifact of averaging
fewer variables.

Splitting the pair, the origin square carries more of the signal than the
destination, \result{sqSourceOnly} against \result{sqDestOnly}. That ordering is
what one would expect if the dominant failure is proposing a move from a square
the model has misremembered as occupied.

\subsection{The errors are coherent}""", "square control section")

open(p, "w", encoding="utf-8").write(s)
print("patched: exact-ply wording, square-control section")
