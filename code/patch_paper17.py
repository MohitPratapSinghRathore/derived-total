"""Revision four: the three new experiments, and the appendices."""
import os

p = os.path.join(os.path.dirname(__file__), "..", "paper", "main.tex")
s = open(p, encoding="utf-8").read()


def once(old, new, label=""):
    global s
    assert old in s, "NOT FOUND " + label + ": " + old[:60]
    s = s.replace(old, new, 1)


NEW = r"""\subsection{A constructive edit: moving a piece in the model's belief}
\label{sec:transfer}

Suppressing a square's occupancy removes the mass on moves from it, which is
consistent with a causal belief state and equally consistent with generic damage
to source-square features. A constructive edit separates the two. We take a
square $A$ the mover genuinely occupies and an empty square $B$, and edit toward
``$A$ is empty'' and ``$B$ holds the piece that was on $A$'' together. Generic
damage predicts that mass leaves $A$; only a belief the policy reads predicts
that mass arrives at $B$.

Both happen. Across \result{trN} positions at the calibrated strength, mass on
moves out of $A$ changes by \result{trOutA} [\result{trOutALo},
\result{trOutAHi}] while a norm-matched random edit gives \result{trOutARand}
[\result{trOutARandLo}, \result{trOutARandHi}]. Mass on moves out of $B$, which
are illegal in the true position, changes by \result{trOutB} [\result{trOutBLo},
\result{trOutBHi}] against \result{trOutBRand} [\result{trOutBRandLo},
\result{trOutBRandHi}] for the random edit. Intervals are bootstrapped over
games.

Splitting the edit budget across two directions makes the manipulation less
reliable than the single-square version: $A$ decodes as empty in
\result{trFlipA} of cases and $B$ as the transferred piece in \result{trFlipB}.
Restricting to the \result{trBothN} cases where both flips demonstrably
succeeded, the constructive effect strengthens, to \result{trOutBBoth}
[\result{trOutBBothLo}, \result{trOutBBothHi}], while the random control on $A$
becomes indistinguishable from zero. We read this as the policy consulting a
joined-up occupancy representation rather than a single feature, while noting it
establishes this for occupancy and not for the whole position.

\subsection{Off distribution: uniformly random legal play}
\label{sec:randomplay}

Every measurement so far uses held-out human games, which are in distribution.
\citet{walker2026chess} report that a uniformly-random-legal-play split stays
discriminative where in-distribution performance saturates, so it is a fair test
of whether our findings lean on familiar opening theory and typical structure.

They do not. On \result{rpGames} games of uniformly random legal play,
occupied-square accuracy falls from \result{rpOccFirst} to \result{rpOccLast} and
the illegal rate rises from \result{rpIllFirst} to \result{rpIllLast}, both
steeper than on human games as expected for an unfamiliar distribution. Belief
consistency is \result{rpBelief} over \result{rpBelN} illegal moves, against a
random-move control of \result{rpRand}. That is slightly higher than the
in-distribution rate, not lower. Coherent error is not an artifact of the model
being on familiar ground.

\subsection{Transcript length against position difficulty}
\label{sec:cycles}

The decay curves are read throughout as loss caused by composing more updates.
Depth is confounded with position difficulty: a position at ply 100 differs from
one at ply 10 in material, phase, and legal-move count. The curves alone do not
license the causal reading, and this test removes the confound.

We insert a reversible cycle, a knight out and back for each side, which returns
the position exactly while adding four plies of transcript. State is then read on
an identical position at several transcript lengths.

\input{tab_cycles}

Fidelity falls monotonically, from \result{cycAcc0} [\result{cycLo0},
\result{cycHi0}] with no insertion to \result{cycAcc12} [\result{cycLo12},
\result{cycHi12}] after twelve added plies, on \result{cycN} positions whose
state is unchanged. The intervals at the two ends do not overlap. Transcript
length alone therefore causes measurable state loss.

The size matters as much as the sign. Twelve plies of pure length cost about four
points of accuracy, where the natural depth curve falls far further over its
range. Both factors contribute, and we do not attribute the whole curve to
composition depth. One caveat we cannot remove: the inserted moves are legal but
unlike training data, so part of the drop may be distribution shift rather than
length as such. The random-play result of Section~\ref{sec:randomplay} bounds
that concern without eliminating it.

"""

once(r"""\subsection{A policy-conditioned state-fidelity horizon}""",
     NEW + r"""\subsection{A policy-conditioned state-fidelity horizon}""",
     "new sections")

# ------------------------------------------------------------- appendices
APP = r"""
\appendix

\section{Experimental detail}
\label{app:methods}

\paragraph{Data.} Public Lichess archives, standard chess only, both players rated
at least 1600, games between 30 and 160 ply, deduplicated by a hash of the move
sequence. Moves are tokenised one per move in UCI form, giving a vocabulary of
\result{nVocab} including padding and a start symbol; there is no sub-move
tokenisation and no board rendering in the input. Sequences are padded to 161
tokens and truncated at 160 ply. Splits are by game: \result{ntrainlm} for
language-model training, \result{ntrainprobe} for probe fitting,
\result{neval} held out for evaluation.

\paragraph{Oracle.} Every game is replayed with a rules engine, recording after
each token the occupancy of all 64 squares as one of 13 classes, the side to
move, and castling rights. Replay completeness is verified before any model is
measured.

\paragraph{Models.} Pre-norm decoder-only Transformers with learned positional
embeddings, GELU feed-forward of width four times the model dimension, and no
dropout. The grid spans 6 to 12 layers and widths 192 to 512, with 6 or 8 heads.
Training uses AdamW, $\beta = (0.9, 0.95)$, weight decay 0.01, a one-cycle
schedule peaking at $3\times10^{-4}$ with 5 percent warmup, batch size 32, 12000
steps, and mixed precision with gradient clipping at 1.0. The final checkpoint is
used; no checkpoint selection is performed. Seeds 0, 1 and 2 control
initialisation, batch order, and any auxiliary sampling.

\paragraph{Probes.} One linear head per layer maps the residual stream to 64
squares by 13 classes, trained with cross-entropy on the probe split only, AdamW
at $10^{-3}$, two epochs, batch size 48. The reported layer is the one with the
highest occupied-square accuracy, selected on the probe split rather than on the
evaluation split. Controls are a per-square per-depth majority predictor, a
label-permutation control pairing activations with another game's state at equal
depth, and a randomised-weight model of identical architecture whose probes are
fitted the same way.

\paragraph{Believed boards.} A decoded board is built by placing the argmax class
at each square, setting side to move from ply parity, and clearing castling
rights so that castling cannot be licensed by an assumption we did not decode.
Legality is tested as pseudo-legality, since a decoded board may lack a king and
make check tests ill-defined. Moves that fail to parse are counted as illegal.

\paragraph{Engine scoring.} Stockfish 17 at fixed depth 10. Loss is the drop from
the best available score to the score after the played move, both from the
mover's point of view, with mate scores mapped to 10000. The blunder threshold is
100 centipawns, fixed in advance.

\paragraph{Interventions.} Edits are added to the residual stream at the output of
the probed block, at the final position only. The direction is the difference of
probe decoder rows for the target and current classes, normalised and scaled to
$\alpha$ times the residual stream's own root-mean-square magnitude times the
square root of the model dimension. The reported $\alpha$ is \result{patchAlphaAttn},
chosen from the sweep in Table~\ref{tab:patchcal} as the smallest value that
reliably flips the decoded belief while leaving a norm-matched random edit at
chance. Random controls are drawn isotropically and rescaled to the same norm.

\paragraph{Recurrent channel.} A gated diagonal recurrence of width 64 inserted
after every fourth block, with the read-out matrix zero-initialised so the module
begins as the identity. The auxiliary objective, when active, is cross-entropy
from the channel to the oracle state with weight 1.0. The parameter-matched
control widens the feed-forward multiplier to 4.156 to absorb the same parameter
count.

\paragraph{Second domain.} Four objects in four boxes, an explicit initial
assignment, then 24 operations drawn evenly between a swap of two named boxes and
a move of a named object to a named box, with a query after every operation
asking where one object is. Answers are weighted five times in the loss. Details
of the six designs that failed before this one are in Section~\ref{sec:synthfail}.

\section{Deviations from the pre-registration}
\label{app:deviations}

We record every deviation rather than presenting the final protocol as the
original one.

\begin{enumerate}[nosep]
\item The horizon was pre-registered on exact-position fidelity, which proved
degenerate. Horizons on occupied-square accuracy and on the illegal-move rate
were added after inspecting baseline data and before evaluating any intervention,
and applied unchanged to all conditions.
\item The bucketed horizon was replaced by an interpolated crossing point,
because the bucketed read-off returned the same bucket for every condition.
\item Planning error was pre-registered as conditional on exact state recovery,
which yields no samples past the opening. It is reported conditional on
move-touched recovery, with a strictness ladder showing what the choice costs.
\item The aggregate and move-touched comparison was first made with
incommensurable statistics, a correlation against a conditional probability. Both
are now scored as predictors of the same outcome.
\item The layer-profile prediction was falsified and is retained as such.
\item The second domain required seven designs. All are reported.
\end{enumerate}

\section{Research questions, design principles, and implications}
\label{app:framing}

This work began as a design-science exercise, and the framing is recorded here
rather than in the body, where an empirical narrative reads better without it.

\paragraph{Research questions and objectives.}
\textbf{RQ1}, can long-horizon accuracy loss be separated into state tracking and
planning, is addressed by \textbf{RO1}, defining both and giving an estimator for
each (Sections~\ref{sec:formal} and \ref{sec:diag}). \textbf{RQ2}, does internal
state loss explain behavioural failure or merely accompany it, is addressed by
\textbf{RO2}, testing the link per token at the level of the state an action
consumes (Sections~\ref{sec:coupling} and \ref{sec:belief}). \textbf{RQ3}, how far
does a search-free model reason before its state stops being recoverable and how
does that respond to scale, is addressed by \textbf{RO3}, measuring horizons
across a seeded ladder against trivial baselines (Section~\ref{sec:scale}).
\textbf{RQ4}, does the diagnostic discriminate between architectural
interventions, is addressed by \textbf{RO4}, evaluating a recurrent state channel
with parameter-matched and ablated controls (Section~\ref{sec:alsb}).

\paragraph{Design principles.}
\textbf{DP1}, ground the diagnostic in an external oracle rather than model
self-report, realised by rules-engine labels at every token. \textbf{DP2},
separate the failure modes before attributing any effect, realised by scoring
planning only where the move is legal and the relevant state is independently
verified. \textbf{DP3}, establish what a trivial predictor achieves first,
realised by majority, label-permutation and randomised-weight controls on every
decodability number. \textbf{DP4}, measure the state the action depends on rather
than the state in aggregate, realised by move-touched fidelity and its
random-pair controls. \textbf{DP5}, test coherence by intervention and not only
by correlation, realised by the calibrated suppression and transfer edits.

\paragraph{Implications for individuals, organisations, and society.}
For the reader of a long generated answer, who is rarely placed to audit step
twenty-eight, the practical value is a stated depth beyond which output should
not be trusted unchecked, and the uncomfortable finding that failure arrives
fluent rather than obviously broken. For organisations, the decomposition
indicates which remedy fits: a recurring per-query verifier cost suits a policy
problem, a one-off structural change suits a memory problem, and reporting only
the sum leaves a team unable to tell which they have. Strategically, the
measurement needs a rules engine and a linear probe rather than privileged access
at scale, so it can be owned and audited in-house rather than taken on trust.
For society, diagnosing a failure is cheaper than paying inference-time compute
to mask it, and remedies that cost a small fraction of parameters travel to
on-device and public-sector deployments that cannot buy deliberation per request.
"""

once(r"\bibliographystyle{apalike}", APP + "\n" + r"\bibliographystyle{apalike}",
     "appendices")

open(p, "w", encoding="utf-8").write(s)
print("patched: three new sections, methods, deviations, framing appendices")
