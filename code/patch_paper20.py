"""Revision seven: bring the conclusion and contribution 4 in line with the
revised formalisation, and fix the occupancy-matching claim."""
import os

p = os.path.join(os.path.dirname(__file__), "..", "paper", "main.tex")
s = open(p, encoding="utf-8").read()


def once(old, new, label=""):
    global s
    assert old in s, "NOT FOUND " + label + ": " + old[:70]
    s = s.replace(old, new, 1)


# ------------------------------------------------------------ the conclusion
i0 = s.index(r"\section{Conclusion}")
i1 = s.index(r"\section*{Declarations}")
CONC = r"""\section{Conclusion}

Long-horizon behavioural failure can arise from degradation of the task state,
from poor action selection given the state that is available, or from failures
that remain unresolved under a lossy decoder. These are not cleanly separable
with the instruments we have, and we have not separated them.

What we can report is narrower and, we think, still worth having. In search-free
chess models, whole-board probe fidelity is a weaker predictor of an invalid next
action than fidelity on the endpoints of the move actually selected, and the gap
widens with depth. A substantial identified subset of invalid actions is
admissible under the board that model appears to hold at that position, at many
times the rate of a mismatched-belief control and of an arbitrary invalid move.
Targeted suppression and transfer edits show that the corresponding occupancy
representation influences the policy rather than merely accompanying it. The
identified share grows across the range of model sizes we can train, and survives
a shift to uniformly random legal play.

These measurements identify state-consistent failure. They do not partition all
failures, and the errors they leave unidentified are unresolved rather than
attributed to reasoning. The practical consequence is a change in what to
measure: a world model reported as an aggregate accuracy tells a practitioner
less than the same representation scored on the state the next action will
consume, and the difference matters most exactly where long-horizon behaviour is
hardest to check.

"""
s = s[:i0] + CONC + s[i1:]

# ---------------------------------------------------------- contribution 4
once(r"""\item \textbf{A per-token decomposition} of long-horizon error into
state-tracking and planning components, against an exact oracle rather than a
learned verifier, with the conditioning analysis needed to keep the two apart.""",
     r"""\item \textbf{A per-token diagnostic stratification} into state mismatch,
belief-consistent error, conditional planning loss, and unresolved failure,
measured against an exact oracle rather than a learned verifier, with the
conditioning analysis that shows how far the strata can be told apart.""",
     "contribution 4")

# ------------------------------------- the memory-versus-reasoning sentence
once(r"""It is competent play in a world that has drifted,
which is a claim about memory rather than about reasoning, and it explains why
fluency outlives correctness: the model does the right thing, in the wrong place.""",
     r"""A substantial share of it is action admissible under a
world that has drifted, which is a claim about state rather than about reasoning
for that share, and it suggests why fluency outlives correctness: the model acts
coherently with respect to a position it no longer holds.""", "memory sentence")

open(p, "w", encoding="utf-8").write(s)
print("patched: conclusion, contribution 4, memory sentence")
