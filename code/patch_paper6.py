"""Resolve every pre-registered prediction explicitly, and fix dangling labels."""
import os

p = os.path.join(os.path.dirname(__file__), "..", "paper", "main.tex")
s = open(p, encoding="utf-8").read()


def once(old, new):
    global s
    assert old in s, old[:70]
    s = s.replace(old, new, 1)


SUMMARY = r"""\subsection{What the predictions came to}
\label{sec:verdict}

Pre-registration is only worth the words if the verdicts are stated, including
the ones that went against us.

\begin{description}[nosep]
\item[H1, supported.] State fidelity declines with depth, from
\result{occFirst} to \result{occLast} occupied-square accuracy, clear of a
majority predictor, a label-permutation control, and a randomised-weight model
(Table~\ref{tab:controls}, Figure~\ref{fig:decay}). The behavioural horizon,
\result{hintAttn} ply, sits well inside the context window.
\item[H2, supported.] The horizon grows slowly with scale, from
\result{hintAttn} ply at \result{paramsAttn} parameters to
\result{hintAttnWide} at \result{paramsAttnWide} (Section~\ref{sec:scale}).
\item[H3, supported only weakly.] The supervised recurrent channel raises state
decodability substantially and the horizon by \result{hintAlsbOne} against
\result{hintAttn} ply. The architecture without supervision changes neither.
\item[H4, not supported as predicted.] We expected an intervention that improved
state tracking to improve behaviour through it. Instead the intervention improved
the measurement far more than its consequences, matching a model twice its size
on decodability while capturing a fraction of the behavioural gain
(Section~\ref{sec:alsb}). We designed H4 to be able to fail and it failed, in a
direction that supports the paper's main claim rather than undermining it.
\item[H5, partially addressed.] The natural-language transfer test our protocol
specifies was not run and is absent rather than approximated
(Section~\ref{sec:limits}). A second oracle-bearing domain, reached only after
several failed designs, tests the weaker claim
(Section~\ref{sec:synthfail}).
\end{description}

Two further results were not predicted at all. The layer profile ran opposite to
our expectation (Section~\ref{sec:bottleneck}). And the quantity this literature
reports as evidence of a world model turned out to carry almost no information
about behaviour, which is the finding we would keep if we could keep only one
(Section~\ref{sec:coupling}).

"""

once(r"""\subsection{A degenerate metric, declared}""",
     SUMMARY + r"""\subsection{A degenerate metric, declared}""")

# reference the two orphaned labels
once(r"""and supported causally by an intervention that induces a false belief and changes
the action accordingly, with a calibrated edit strength and a norm-matched
control.""",
     r"""and supported causally by an intervention that induces a false belief and changes
the action accordingly (Section~\ref{sec:causal}), with a calibrated edit
strength and a norm-matched control.""")

once(r"""measurement procedure.""",
     r"""measurement procedure, and Table~\ref{tab:dp} states the principles it
embodies.""")

open(p, "w", encoding="utf-8").write(s)
print("patched: predictions resolved, labels referenced")
