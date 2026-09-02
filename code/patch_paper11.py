"""Lead with the findings that carry the paper, and add the fidelity horizon."""
import os

p = os.path.join(os.path.dirname(__file__), "..", "paper", "main.tex")
s = open(p, encoding="utf-8").read()


def once(old, new):
    global s
    assert old in s, "NOT FOUND: " + old[:70]
    s = s.replace(old, new, 1)


# ------------------------------------------- resequence the introduction's list
i0 = s.index("\\begin{enumerate}[nosep]\n\\item \\textbf{Aggregate state fidelity")
i1 = s.index("\\end{enumerate}", i0) + len("\\end{enumerate}")
FIND = """\\begin{enumerate}[nosep]
\\item \\textbf{The errors are coherent.} We reconstruct the board the model
appears to believe and ask whether its illegal move is legal there. It is, for
\\result{belIllAttn} of illegal moves, with a bootstrap interval of
[\\result{belIllLoAttn}, \\result{belIllHiAttn}]. The same move scored against
another position's believed board at equal depth is legal \\result{belMisAttn} of
the time, and an arbitrary illegal move scored against the model's own believed
board is legal \\result{belRandAttn} of the time. Lossy decoding caps what is
detectable at \\result{belLegAttn}, the rate for moves that were genuinely legal.
\\item \\textbf{The representation is read, not merely present.} Editing the probe
direction that encodes an occupied square removes \\result{patchCutAttn} of the
probability mass on moves from that square, while a norm-matched random edit
removes \\result{patchRandCutAttn} and shifts the mass at chance. The edit
strength is calibrated rather than chosen.
\\item \\textbf{Where the world model is measured matters, increasingly with
depth.} Whole-board probe accuracy predicts an illegal next move with an AUC of
\\result{aucAggMin} to \\result{aucAggMax}. The same representation restricted to
the squares the move uses reaches \\result{aucActMin} to \\result{aucActMax}. In
the opening the two are almost indistinguishable; the gap opens as reasoning
deepens, and in a joint fit the action-relevant term carries a standardised
weight of \\result{betaAct} against \\result{betaAgg}.
\\end{enumerate}"""
s = s[:i0] + FIND + s[i1:]

once(r"""The third result is the one we would defend hardest. Long-horizon failure in
these models is not noise breaking through as the context grows. It is competent
play in a world that has drifted, which is a claim about memory rather than about
reasoning, and it explains why fluency survives correctness: the model is doing
the right thing, in the wrong place.""",
     r"""The first two are the ones we would defend hardest, and they belong together:
the errors are coherent with a decayed state, and that state is causally
load-bearing rather than an epiphenomenon. Failure here is not noise breaking
through as the context grows. It is competent play in a world that has drifted,
which is a claim about memory rather than about reasoning, and it explains why
fluency outlives correctness: the model does the right thing, in the wrong place.""")

# ------------------------------------------------------------------- keywords
once(r"""\noindent\textbf{Keywords:} long-horizon reasoning, state tracking, world models,
probing, model diagnostics, design science""",
     r"""\noindent\textbf{Keywords:} state tracking, world models, probing, model
diagnostics, chess, long-horizon reasoning""")

# ------------------------------------------- the verdict paragraph, rebalanced
once(r"""Two further results were not predicted at all. The layer profile ran opposite to
our expectation (Section~\ref{sec:bottleneck}). And the quantity this literature
reports as evidence of a world model turned out to carry almost no information
about behaviour, which is the finding we would keep if we could keep only one
(Section~\ref{sec:coupling}).""",
     r"""Three results were not predicted at all. The layer profile ran opposite to our
expectation (Section~\ref{sec:bottleneck}). Ablating the recurrent read-out
improved behaviour while worsening prediction, which turned out to say something
about our own observable rather than about the artifact
(Section~\ref{sec:alsb}). And the standard whole-board measure of a world model
proved a weaker predictor of behaviour than the same measure restricted to the
state an action consumes, by a margin that grows with depth
(Section~\ref{sec:coupling}).

If we could keep only one result it would be belief consistency together with the
causal edit, which say that these models act coherently on a state they are
losing. The measurement-placement result is a practical corollary: it tells you
where to look, and it matters most exactly where long-horizon behaviour is
hardest.""")

# --------------------------------------- a horizon on action-relevant fidelity
once(r"""\subsection{A degenerate metric, declared}""",
     r"""\subsection{A horizon that does not depend on behaviour}
\label{sec:hfid}

Both horizons used so far are read off behaviour, and the read-out ablation
showed why that is fragile: a model can lower its illegal-move rate by hedging
toward the legal-move prior, without knowing the position any better. A horizon
defined on the representation avoids this, because it never consults the output
distribution.

We define $H_{\mathrm{fid}}$ as the depth at which the probability that the probe
is wrong on the squares the model's chosen move touches first crosses one half.
It uses the same probe and the same activations as everything else, and it is
immune to hedging: a model that spreads mass over legal moves gains nothing,
since the measure asks what it knows, not what it emits.

Read this way the ladder is orderly. The smallest model reaches
\result{hfidRung1} ply, the baseline \result{hfidAttn}, and the largest
\result{hfidRung6}, against behavioural horizons of \result{hintAttn} and
\result{hfidRung6} respectively for the same two models. Action-relevant error at
the shallowest depth runs from \result{actErrFirstRung1} in the smallest model to
\result{actErrFirstRung6} in the largest. We report both horizons throughout, and
prefer this one where the comparison is between architectures rather than between
depths, since that is where hedging is most likely to confound a behavioural
reading.

\subsection{A degenerate metric, declared}""")

open(p, "w", encoding="utf-8").write(s)
print("patched: findings resequenced, verdict rebalanced, fidelity horizon added")
