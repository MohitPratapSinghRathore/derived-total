"""Revision six, from review: duplicated tables, contradictory numbers, the
ceiling language, the conclusion, the formal state, and two overstatements."""
import os

p = os.path.join(os.path.dirname(__file__), "..", "paper", "main.tex")
s = open(p, encoding="utf-8").read()


def once(old, new, label=""):
    global s
    assert old in s, "NOT FOUND " + label + ": " + old[:60]
    s = s.replace(old, new, 1)


# ---- 1. table duplication: place each table once, drop two superseded ones
once(r"\input{tables}",
     "\n".join([
         r"\input{tab_controls}",
         r"\input{tab_auc}",
         r"\input{tab_mechanism}",
         r"\input{tab_patch}",
         r"\input{tab_horizon}",
         r"\input{tab_scale}",
         r"\input{tab_strictness}",
         r"\input{tab_boxes}"]), "table block")

# ---- 2. the ceiling contradiction and the terminology, everywhere
once(r"""Against that ceiling, roughly half of the model's illegal moves are coherent
actions in a misremembered world, and the enrichment over the matched control is
a factor of \result{belRatioMisAttn}. This is what we mean by coherent error, and
it is the clearest statement we can make that long-horizon degradation here is a
memory failure rather than a reasoning failure.""",
     r"""We do not divide by that reference. Doing so would estimate an unobserved true
rate only under an assumption we have not tested, that the probe is equally
sensitive on legal and illegal cases. What the comparison licenses is narrower
and still substantial: a large identified subset of illegal moves is admissible
under the board this model appears to hold at this position, at
\result{belRatioMisAttn} times the rate of the matched control. It identifies
state-consistent failure. It does not assign the remainder.""", "ceiling contra")

for old, new in [
    ("decoding ceiling", "legal-move recovery reference"),
    ("the ceiling this test can detect", "the recovery reference for this test"),
    ("its own decoding ceiling", "its own legal-move recovery reference"),
    ("each model's decoding ceiling", "each model's recovery reference"),
    ("each model's own decoding ceiling", "each model's own recovery reference"),
]:
    s = s.replace(old, new)

# ---- 3. the scale section copy error, and the ratio framing
once(r"""while the mismatched control stays flat near
\result{beliefMisBoxAttn}.""",
     r"""while the mismatched control stays flat near
\result{pcBelMis} and the random-move control near \result{pcBelRand}. We show
the ratio as a descriptive normalisation, since the reference itself rises with
size, and not as an estimate of a hidden true proportion.""", "scale control")

# ---- 4. belief section reports the clustered interval on its own population
once(r"""Of the illegal moves the model plays, \result{belIllAttn} are legal in its own
believed position, with a bootstrap interval of [\result{belIllLoAttn},
\result{belIllHiAttn}].""",
     r"""Of the illegal moves the model plays, \result{pcBel} are legal in its own
believed position, with a game-clustered interval of [\result{pcBelLo},
\result{pcBelHi}] over \result{pcN} illegal moves from \result{pcGames} games.
Positions within a game are not independent, so the interval resamples whole
games.""", "belief clustered")

once(r"""The same move scored
against a believed position decoded elsewhere at equal depth is legal
\result{belMisAttn} of the time, so the effect is specific to this belief at this
position rather than to plausible boards generally. An arbitrary illegal move
scored against the model's own believed position is legal \result{belRandAttn} of
the time, so the believed board is not permissive.""",
     r"""The same move scored
against a believed position decoded elsewhere at equal depth is legal
\result{pcBelMis} [\result{pcBelMisLo}, \result{pcBelMisHi}] of the time, so the
effect is specific to this belief at this position rather than to plausible
boards generally. An arbitrary illegal move scored against the model's own
believed position is legal \result{pcBelRand} [\result{pcBelRandLo},
\result{pcBelRandHi}] of the time, so the believed board is not permissive.""",
     "belief controls clustered")

# ---- 5. the clustered section is a restricted sensitivity analysis
once(r"""The analysis covers \result{stN} positions from \result{stGames} games, of which
\result{stIll} carry an illegal top-1 move, over ply 10 to 80 where the depth
buckets are defined.""",
     r"""Section~\ref{sec:belief} already reports game-clustered intervals for belief
consistency on the full evaluation population. This section is a restricted
sensitivity analysis on a different population, \result{stN} positions from
\result{stGames} games of which \result{stIll} carry an illegal top-1 move, over
ply 10 to 80 where the depth buckets used by the predictor comparison are
defined. Estimates therefore differ slightly from the primary ones, and both are
reported rather than reconciled by choosing whichever is larger.""",
     "stats population")

# ---- 6. formal state versus decoded state
once(r"""Let a task be a token sequence $x_{1:L}$ generated left to right, and
$s_t = \Phi(x_{1:t})$ the exact state after the prefix, from an external oracle
$\Phi$. For chess this is piece placement, side to move, castling rights, and the
en-passant target.""",
     r"""Let a task be a token sequence $x_{1:L}$ generated left to right, and
$s_t = \Phi(x_{1:t})$ the exact state after the prefix, from an external oracle
$\Phi$. For chess $s_t = (b_t, \tau_t, c_t, e_t)$: piece placement, side to move,
castling rights, and the en-passant target.

Our probe decodes only the first component. Write
$\hat{b}_t^{(\ell)} = g^{(\ell)}(h_t^{(\ell)})$ for decoded piece placement, and
let the pseudo-state used by the belief test be
$\tilde{s}_t = (\hat{b}_t, \tau_t, \varnothing, \varnothing)$, with side to move
taken from ply parity and castling and en-passant disabled so that nothing is
licensed by an assumption we did not decode. Claims below concern
$\hat{b}_t$ and $\tilde{s}_t$, not $s_t$.""", "state notation")

once(r"""\begin{definition}[State fidelity]
$F^{(\ell)}(t) = \Pr[\hat{s}_t^{(\ell)} = s_t]$, the probability the exact state
is linearly decodable at depth $t$.
\end{definition}""",
     r"""\begin{definition}[Board-placement fidelity]
$F^{(\ell)}(t) = \Pr[\hat{b}_t^{(\ell)} = b_t]$, the probability that piece
placement is linearly decodable at depth $t$. We name it for placement rather
than for state because side to move, castling rights and the en-passant target
are not decoded.
\end{definition}""", "definition 1")

# ---- 7. the naming slip in definition 2
once(r"""We name it for what it measures rather than calling
it move-touched, because it is a deliberately local proxy.""",
     r"""We name it for what it measures rather than calling it action-relevant, because
it is a deliberately local proxy.""", "definition 2 slip")

# ---- 8. S5 softened
once(r"""That makes the state an element of $S_5$, the
smallest non-solvable symmetric group, which is precisely the case bounded-depth
sequence models are known to be unable to track \citep{merrill2024illusion}; we
had chosen the hardest available instance by accident.""",
     r"""That makes the state an element of $S_5$, a
permutation-composition problem that is theoretically hard for the relevant
fixed-depth model classes under asymptotic expressivity assumptions
\citep{merrill2024illusion}. That may have contributed, but a theorem about
asymptotic expressivity does not establish why one finite model failed to
optimise on one bounded-length dataset, and we do not claim group structure as
the cause.""", "S5")

# ---- 9. the architectural motivation
once(r"""A model that reasons over many steps has to keep track of something. A
Transformer stores nothing between positions.""",
     r"""A model that reasons over many steps has to keep track of something. An
attention-only decoder has no explicit bounded mutable state in which to keep
it.""", "intro opening")

once(r"""For an attention-only decoder, computing $h_t^{(\ell)}$ attends over $t$ prior
positions, so maintaining state costs $O(t)$ reads and $O(t)$ key-value memory
while the state has description length $O(1)$. Nothing amortises it: $s_t$ and
$s_{t+1}$ differ by one move and are computed independently.""",
     r"""Under full attention each new token reads an $O(t)$ key-value history and
retains $O(t)$ cached memory, while the task state has description length
$O(1)$. A key-value cache does amortise the computation of earlier activations,
so the cost is not recomputation from scratch; what the architecture lacks is a
bounded mutable state that could carry $s_t$ forward and be updated in place. We
offer this as a motivating cost mismatch, not as a proof that fidelity must
decay.""", "proposition")

open(p, "w", encoding="utf-8").write(s)
print("patched: tables, ceiling language, populations, state notation, S5, framing")
