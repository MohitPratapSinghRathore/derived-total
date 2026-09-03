"""Revision one: the formal definitions, the novelty position, and the terminology.

  1. Definition 3 presented an additive identity that does not hold. A state
     mismatch need not produce a behavioural error, and our own numbers show it
     often does not. Replaced by an observational stratification plus a named,
     identified subset, with the remainder called unresolved rather than
     attributed to planning.
  2. The novelty claim was false: the illegal-moves-legal-on-decoded-board
     measurement is reported in ICLR 2026 work. Repositioned around what is
     actually different, and the contrast with their causal finding is now one of
     the more interesting parts of the related work.
  3. Action-relevant fidelity renamed move-touched fidelity, since it scores only
     source and destination and not the full legality dependency.
"""
import os

p = os.path.join(os.path.dirname(__file__), "..", "paper", "main.tex")
s = open(p, encoding="utf-8").read()


def once(old, new, label=""):
    global s
    assert old in s, "NOT FOUND " + label + ": " + old[:60]
    s = s.replace(old, new, 1)


# ---------------------------------------------- 1. the definitions, rewritten
i0 = s.index(r"\begin{definition}[Error decomposition]")
i1 = s.index(r"\end{definition}", i0) + len(r"\end{definition}")
NEWDEF = r"""\begin{definition}[Behavioural error and its stratification]
Let $Y_t = 1$ when the action selected at depth $t$ is behaviourally invalid, and
let $M_t = 1$ when the decoded state disagrees with the oracle on the components
the selected action reads. By the law of total probability,
\[
\Pr(Y_t{=}1) = \Pr(M_t{=}1)\Pr(Y_t{=}1 \mid M_t{=}1)
             + \Pr(M_t{=}0)\Pr(Y_t{=}1 \mid M_t{=}0).
\]
This is an observational stratification and not a causal decomposition. The first
term is not by itself memory-caused error and the second is not by itself
policy-caused error.
\end{definition}

\begin{definition}[Belief-consistent error]
Write $\mathcal{A}(s)$ for the actions valid in state $s$ and
$\widetilde{\mathcal{A}}(\hat{s})$ for those admitted by the decoded state. An
error is \emph{belief-consistent} when
$B_t = \mathbf{1}[\,a_t \notin \mathcal{A}(s_t) \wedge a_t \in
\widetilde{\mathcal{A}}(\hat{s}_t)\,]$, and we report
$\Pr(B_t{=}1 \mid a_t \notin \mathcal{A}(s_t))$.
\end{definition}

Two consequences of this framing are worth stating plainly, because an earlier
version of this paper got them wrong. A state mismatch need not produce a
behavioural error: our own measurements show the probe wrong on the squares a
move touches for a substantial fraction of \emph{legal} moves, so
$M_t{=}1$ does not imply $Y_t{=}1$. And because the probe is an incomplete and
lossy read of the state, belief consistency identifies a subset of errors rather
than partitioning all of them. The errors it does not identify are
\emph{unresolved}: they may be genuine policy failures, state errors the probe
did not capture, information the probe cannot read linearly, or reconstruction
failures. We do not call them planning errors."""
s = s[:i0] + NEWDEF + s[i1:]

OLD_BC = r"""\begin{definition}[Belief consistency]
An action is \emph{belief-consistent} when it is invalid in $s_t$ but valid in
$\hat{s}_t$. The rate of belief consistency among invalid actions distinguishes
coherent action on a wrong state from incoherent action.
\end{definition}

"""
assert OLD_BC in s, "old belief-consistency definition not found"
s = s.replace(OLD_BC, "", 1)

# ------------------------------------------------- 2. novelty, repositioned
once(r"""Probe design follows the control-task discipline of
\citet{hewitt2019designing}. To our knowledge the belief-consistency test of
Section~\ref{sec:belief}, which asks whether a wrong action is right in the
model's own decoded world, has not been reported.""",
     r"""Probe design
follows the control-task discipline of \citet{hewitt2019designing}.

\paragraph{Concurrent work, and what is actually new here.}
The measurement at the centre of this paper is not new.
\citet{balogh2026verification} report the fraction of illegal chess moves that
are legal on a probe-reconstructed board, and reach a conclusion close to the
opposite of ours: for models trained on large datasets that fraction is small, and
the gradient of their board-state probe is nearly orthogonal to that of the
next-token head, which they read as the probe lacking a causal role in
generation. \citet{li2026tracking} separate state tracking from decision quality
in searchless chess transformers, treating illegal moves as direct evidence of
tracking failure. \citet{walker2026chess} benchmark exact state tracking across
architectures over a parameter range close to ours, and
\citet{pereira2026transformers} show world-state representations decaying during
generation in frontier reasoning models, with a causal restoration experiment.
\citet{harang2025tracking} document loss of coherent internal state over long
sequences. State decay with depth, chess state tracking, and scaling this
measurement are therefore not novel contributions of ours, and we do not claim
them.

What differs is the conditions and the conclusion. Their illegal moves are
adversarially induced; ours arise on ordinary held-out trajectories, which is a
different population and may have a different mechanism. We add same-depth
mismatched-belief and random-move controls, which bound how much of the effect a
merely plausible board would produce. We compare aggregate against
action-conditioned predictors on equal footing rather than assuming the aggregate
is the right one. And where their gradient-orthogonality analysis finds the probe
weakly coupled to generation, our calibrated residual-stream intervention finds a
strong effect on the action. Those two results are not strictly contradictory,
since a direction can be causally load-bearing at one layer while its gradient is
near-orthogonal to the output head, but the tension is real and we flag it as the
most useful open question this paper raises.""", "novelty")

# ------------------------------------ 3. rename action-relevant -> move-touched
once(r"""\begin{definition}[Action-relevant fidelity]
For an action $a$ touching a set of state components $R(a)$, the probability that
$\hat{s}_t$ and $s_t$ agree on $R(a)$. For a chess move this is the origin and
destination squares.
\end{definition}""",
     r"""\begin{definition}[Move-touched fidelity]
The probability that $\hat{s}_t$ and $s_t$ agree on the origin and destination
squares of the selected move. We name it for what it measures rather than calling
it action-relevant, because it is a deliberately local proxy. The state an action
actually depends on is larger: legality can turn on intermediate path squares for
sliding pieces, the king's square and post-move attack status, pins and
discovered attacks, castling rights and transit squares, the en-passant target,
side to move, and promotion. A fuller treatment would define
$R_{\mathrm{legal}}(a, s)$, the complete set of components legality depends on,
and report a progression from whole-board through move-touched to path-aware and
fully legality-aware fidelity. We report the local proxy and the controls that
bound it, and leave that progression to future work.
\end{definition}""", "rename definition")

for old, new in [
    ("action-relevant fidelity", "move-touched fidelity"),
    ("Action-relevant state fidelity", "Move-touched fidelity"),
    ("action-relevant state fidelity", "move-touched fidelity"),
    ("action-relevant state error", "move-touched state error"),
    ("Action-relevant error", "Move-touched error"),
    ("action-relevant term", "move-touched term"),
    ("the action-relevant measure", "the move-touched measure"),
    ("action-relevant state accuracy", "move-touched accuracy"),
    ("action-relevant state", "move-touched state"),
    ("Action-relevant", "Move-touched"),
    ("action-relevant", "move-touched"),
]:
    s = s.replace(old, new)

open(p, "w", encoding="utf-8").write(s)
print("patched: definitions, novelty position, move-touched terminology")
