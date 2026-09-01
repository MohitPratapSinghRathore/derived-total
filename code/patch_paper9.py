"""Corrections arising from review.

  1. the aggregate versus action-relevant claim was overstated, because the two
     were measured with incommensurable statistics
  2. the read-out ablation anomaly is explained, having been measured
  3. L2 contradicted Section 7.9 about the second domain
  4. the scale claim is stated as a ratio against each model's own ceiling
  5. Karvonen is cited, being the closest prior work
  6. the abstract states the scale actually studied
"""
import os

p = os.path.join(os.path.dirname(__file__), "..", "paper", "main.tex")
s = open(p, encoding="utf-8").read()


def once(old, new):
    global s
    assert old in s, "NOT FOUND: " + old[:70]
    s = s.replace(old, new, 1)


# ---------------------------------------------------------------- 1. the claim
once(r"""\item \textbf{Aggregate state fidelity is a poor predictor of behaviour.} Probe
accuracy over the whole board, which is the standard reported quantity, correlates
with illegal-move production at only \result{corrGlobalMinAttn} to
\result{corrGlobalMaxAttn} within a fixed depth. A model can have lost a great
deal of the board and still play legally, and can have most of it and still fail.""",
     r"""\item \textbf{Aggregate state fidelity is the weaker predictor of behaviour.}
Scored as a predictor of an illegal next move within a fixed depth, whole-board
probe accuracy reaches an AUC of \result{aucAggMin} to \result{aucAggMax}. It
carries information, and it is the weaker signal: the same representation
restricted to the squares the move uses reaches \result{aucActMin} to
\result{aucActMax}, and the gap widens with depth.""")

once(r"""\item \textbf{Action-relevant state fidelity is a strong predictor.} Restricting
attention to the squares a move actually touches, the probe is wrong for
\result{locIllAttn} of illegal moves and \result{locLegAttn} of legal ones. What
matters is not how much of the world has been lost but whether the part about to
be used has been lost.""",
     r"""\item \textbf{Action-relevant state fidelity is the stronger predictor.} The
probe is wrong on the squares a move touches for \result{locIllAttn} of illegal
moves against \result{locLegAttn} of legal ones, and in a joint logistic fit with
depth held fixed it carries a standardised weight of \result{betaAct} against
\result{betaAgg} for the aggregate. What matters is less how much of the world
has been lost than whether the part about to be used has been lost.""")

# ------------------------------------------------------- the results subsection
once(r"""\subsection{Aggregate fidelity does not explain behaviour}""",
     r"""\subsection{Aggregate fidelity is the weaker signal}""")

once(r"""The two curves fall together, and it would be easy to stop there and call them
one phenomenon. They are not, at least not in the form usually reported. Within a
fixed depth bucket, so that both quantities cannot simply be tracking depth, the
correlation between the number of misremembered squares and whether the model
plays an illegal move ranges from \result{corrGlobalMinAttn} to
\result{corrGlobalMaxAttn}. Aggregate board fidelity, the number this literature
reports as evidence of a world model, carries almost no information about whether
the next action will be valid.""",
     r"""The two curves fall together, and it would be easy to stop there and call them
one phenomenon. The relationship is real but weaker than that, and measuring it
carelessly overstates it.

An earlier version of this analysis compared a point-biserial correlation for the
aggregate predictor against a conditional probability for the action-relevant
one. Those are different statistics, and the contrast flattered the conclusion: a
correlation against a heavily skewed binary outcome is attenuated, so the
aggregate measure looked uninformative when it was merely differently scaled. We
now score both the same way, as predictors of the same binary outcome on the same
samples, within depth so that neither can simply be tracking depth.

Aggregate board fidelity predicts an illegal next move with an AUC between
\result{aucAggMin} and \result{aucAggMax}. That is above chance and not
negligible. The same representation restricted to the two squares the move
touches reaches \result{aucActMin} to \result{aucActMax}, and the two are
closest in the opening and furthest apart at depth: at ply 40 to 50 the aggregate
reaches \result{aucAgg40to50} against \result{aucAct40to50}. Fitted jointly
with depth as a covariate, the action-relevant term takes a standardised weight
of \result{betaAct} against \result{betaAgg} for the aggregate.""")

once(r"""\caption{(a) Within a fixed depth, the correlation between the number of
misremembered squares and playing an illegal move is close to zero. (b) The same
representation, scored only on the squares the move actually uses, separates
illegal from legal moves sharply.}""",
     r"""\caption{(a) Within a fixed depth, the correlation between the number of
misremembered squares and playing an illegal move is small. (b) The same
representation, scored only on the squares the move actually uses, separates
illegal from legal moves sharply. Table~\ref{tab:auc} compares the two on equal
footing, as predictors of the same outcome.}""")

once(r"""We think this deserves emphasis because it is a negative result about a standard
practice. Reporting that a probe recovers a world model at some accuracy says
little about whether that model is load-bearing.""",
     r"""The claim we can defend is therefore narrower than the one we first wrote, and it
is still a claim about a standard practice. Whole-board probe accuracy is a
usable but weak proxy for whether a world model is load-bearing, it degrades as a
proxy exactly where long-horizon behaviour matters most, and a measurement
restricted to the state an action consumes is several times more informative from
the same probe and the same activations.""")

# --------------------------------------------------------- 2. ablation anomaly
once(r"""We therefore report H3 as supported only in its weakest form""",
     r"""\paragraph{An ablation that improves behaviour.}
Zeroing the read-out at test time lowers the illegal-move rate, from
\result{ill40AlsbOne} to \result{ill40Zab}, and lengthens the horizon, from
\result{hintAlsbOne} to \result{hintZab} ply. Removing a component the model was
trained with should not help, so we measured what else it does. Language-model
loss rises when the read-out is removed, from \result{lmIntact} to
\result{lmAblated} across seeds, and the read-out is not vestigial: it injects a
vector of \result{injRatio} of the residual stream's own norm. The ablated model
is worse at predicting the continuation and simultaneously better at staying
legal, and the reason is visible in the distribution. It places more mass on
legal moves than the intact model, \result{massAblated} against
\result{massIntact}, while committing less confidently to any particular one.

The lesson is about our own observable rather than about the artifact. A model
can lower its illegal-move rate by hedging toward the legal-move prior, so the
illegal rate is not a monotone proxy for the quality of state maintenance, and
should be read alongside a likelihood. This does not disturb the coupling,
belief-consistency, or causal results, which are computed per position rather
than from a rate, but it does qualify the intervention comparison and the horizon
metric, and we flag it rather than leaving a reader to find it.

We therefore report H3 as supported only in its weakest form""")

# -------------------------------------------------------------- 3. stale L2
once(r"""unrepresentative of open-ended reasoning. Our attempt at a second domain did not
reach a measurable regime (Section~\ref{sec:synthfail}), so the findings rest on
one instrument and should be read that way.""",
     r"""unrepresentative of open-ended reasoning. The second domain we built does reach a
measurable regime (Section~\ref{sec:synthfail}), but only its supervised
condition maintains a decodable state, so the attention-only replication we
wanted is not available there. The central findings rest on one instrument and
should be read that way.""")

# ------------------------------------------------------------ 4. scale as ratio
once(r"""Figure~\ref{fig:scale}(b) reports belief consistency and its matched control at
every rung. If the coherence of errors is stable across the ladder while the
horizon shifts, then what we have characterised is a property of how these models
maintain state rather than an artifact of the smallest ones.""",
     r"""Figure~\ref{fig:scale}(b) reports belief consistency at every rung. The raw rate
rises with size, but so does each model's decoding ceiling, so the honest
quantity is the ratio of the two. Normalised that way the coherence of errors
still rises, from \result{belRatioSmall} at \result{paramsSmall} parameters to
\result{belRatioLarge} at \result{paramsLarge}, while the mismatched control stays
flat. What we have characterised strengthens with scale over the range we can
reach rather than washing out, which is evidence against it being an artifact of
the smallest models. Every rung is a single seed, so we read the direction of the
trend and not its precise slope.""")

# ----------------------------------------------------------- 5. closest prior work
once(r"""Linear probes recovering board state from game transcripts
\citep{li2023emergent}, the linear structure of those representations
\citep{nanda2023emergent}, chess as a state-tracking testbed
\citep{toshniwal2022chess}, and causal editing of internal associations
\citep{meng2022locating} together establish that sequence models form internal
world models and that intervening on them changes behaviour.""",
     r"""Linear probes recovering board state from game transcripts
\citep{li2023emergent}, the linear structure of those representations
\citep{nanda2023emergent}, chess as a state-tracking testbed
\citep{toshniwal2022chess}, and causal editing of internal associations
\citep{meng2022locating} together establish that sequence models form internal
world models and that intervening on them changes behaviour. Closest to our
setting, \citet{karvonen2024emergent} probes board state and estimated player
skill in a chess-playing language model and intervenes on those representations,
and the same board-state probes are later used as ground truth for evaluating
dictionary learning \citep{karvonen2024measuring}. That work establishes the
representation exists and is manipulable in this exact domain, which is our
premise rather than our result.""")

# --------------------------------------------- 6. abstract states the scale studied
once(r"""them where an exact state oracle exists, chess from move notation with no search
at inference, reading linear probes against the true position at every token.""",
     r"""them where an exact state oracle exists, chess from move notation with no search
at inference, in models of \result{paramsSmall} to \result{paramsLarge}
parameters, reading linear probes against the true position at every token.""")

open(p, "w", encoding="utf-8").write(s)
print("patched: six review corrections")
