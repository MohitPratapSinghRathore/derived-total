"""Restructure for an ML audience and rebalance the claims.

  1. title and abstract scoped to chess; long-horizon becomes the motivation
  2. design-science scaffolding removed: the DP table and the RQ/RO block go,
     replaced by prose. What design science actually contributed, the
     pre-registration, the stated verdicts and the reported failures, stays.
  3. contributions reordered so belief consistency and the causal edit lead
  4. the verdict paragraph no longer calls the aggregate contrast the finding we
     would keep above all others, since the like-for-like comparison shrank it
  5. a horizon defined on action-relevant fidelity, beside the behavioural one
  6. the implications section folds into one paragraph of the discussion
"""
import os

p = os.path.join(os.path.dirname(__file__), "..", "paper", "main.tex")
s = open(p, encoding="utf-8").read()


def once(old, new):
    global s
    assert old in s, "NOT FOUND: " + old[:70]
    s = s.replace(old, new, 1)


def excise(start_marker, end_marker):
    """Remove from start_marker up to (not including) end_marker."""
    global s
    i = s.index(start_marker)
    j = s.index(end_marker, i)
    s = s[:i] + s[j:]


# ------------------------------------------------------------------- 1. title
once(r"\title{\bf Coherent Errors: Long-Horizon Failure as Action\\on Misremembered State}",
     r"\title{\bf Coherent Errors: Chess Transformers Play Correctly\\on Misremembered Boards}")

# ------------------------------------------------- 2. design-science scaffolding
excise("\\begin{table}[t]\n\\centering\\small\n\\caption{Design principles and their realisation.}",
       "\\paragraph{In scope.}")

once(r"""because the measurement does not distinguish the causes. The artifact is a
measurement procedure, and Table~\ref{tab:dp} states the principles it
embodies.""",
     r"""because the measurement does not distinguish the causes.

The procedure rests on four commitments, each ruling out a way of fooling
oneself. The diagnostic is grounded in an external oracle rather than model
self-report. The two failure modes are separated before any effect is attributed.
Every decodability number is reported against trivial predictors that could
produce it without a world model. And the representation is tested by
intervention, not only by correlation.""")

excise(r"\paragraph{Research questions and objectives.}", r"\paragraph{Contributions.}")

once(r"""We present this as design-science research in the sense of
\citet{hevner2004design} and \citet{peffers2007design}, where the contribution is
an artifact together with the design knowledge it carries
\citep{gregor2013positioning}.

""", "")

# --------------------------------------------- 3. contributions, resequenced
i0 = s.index("\\paragraph{Contributions.}")
i1 = s.index("\\section{Related Work}")
CONTRIB = """\\paragraph{Contributions.}
\\begin{enumerate}[nosep]
\\item \\textbf{Long-horizon errors are coherent.} An invalid action is usually a
valid action in the position the model believes it is in, established against
mismatched-belief and random-move controls, and strengthening across the range of
model sizes we can train.
\\item \\textbf{The state representation is read, not merely present.} Inducing a
false belief by editing a single probe direction changes the action accordingly,
at a calibrated edit strength and against a norm-matched control.
\\item \\textbf{Where a world model is measured matters.} Whole-board probe
accuracy predicts whether the next action will be valid, and the same measurement
restricted to the state that action consumes predicts it better, with the gap
widening as reasoning deepens. We define a horizon on that fidelity directly,
which avoids a failure mode of behavioural horizons that we document.
\\item \\textbf{A per-token decomposition} of long-horizon error into
state-tracking and planning components, against an exact oracle rather than a
learned verifier, with the conditioning analysis needed to keep the two apart.
\\item \\textbf{An account of what failed}, including a degenerate metric, a
prediction that ran backwards, an intervention that improved the measurement more
than the behaviour, and seven task designs before a second domain became
measurable. Each changed a decision and none is usually reported.
\\end{enumerate}

"""
s = s[:i0] + CONTRIB + s[i1:]

# ------------------------------------------- 6. implications folded into discussion
i0 = s.index("\\section{Implications for Individuals, Organisations, and Society}")
i1 = s.index("\\section{Limitations}")
FOLD = """\\paragraph{What this implies for practice.}
The decomposition tells a team which remedy fits the failure they have. Where
state-tracking error dominates, an inference-time verifier is an expensive way to
buy back something the architecture gave away, and it is a recurring per-query
cost rather than a one-off. Where planning error dominates, a verifier is exactly
right. Reporting only the sum leaves a practitioner unable to tell which world
they are in, and the two worlds recommend opposite purchases. The measurement
needed to tell them apart is a rules engine and a linear probe, not privileged
access to a model at scale, so it can be run in-house on a system a team already
depends on. For the reader of a long generated answer, who is rarely in a
position to audit step twenty-eight, the practical value is a stated depth beyond
which the output should not be trusted without checking.

"""
s = s[:i0] + FOLD + s[i1:]

open(p, "w", encoding="utf-8").write(s)
print("patched: title, scaffolding removed, contributions resequenced, "
      "implications folded")
