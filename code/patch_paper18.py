"""Revision five: clustered inference, out-of-sample model comparison, and a
held-out abstention rule."""
import os

p = os.path.join(os.path.dirname(__file__), "..", "paper", "main.tex")
s = open(p, encoding="utf-8").read()


def once(old, new, label=""):
    global s
    assert old in s, "NOT FOUND " + label + ": " + old[:60]
    s = s.replace(old, new, 1)


SEC = r"""\subsection{Inference that respects the clustering}
\label{sec:stats}

Positions within a game are not independent: they share an opening, a player, and
an accumulating history. Intervals computed over positions therefore understate
uncertainty, and every interval in this section resamples whole games instead.
The analysis covers \result{stN} positions from \result{stGames} games, of which
\result{stIll} carry an illegal top-1 move, over ply 10 to 80 where the depth
buckets are defined.

\input{tab_stats}

Belief consistency is \result{cbBel} [\result{cbBelLo}, \result{cbBelHi}], against
\result{cbBelMis} [\result{cbBelMisLo}, \result{cbBelMisHi}] for a mismatched
belief and \result{cbBelRand} [\result{cbBelRandLo}, \result{cbBelRandHi}] for a
random illegal move. The intervals are far apart, and the separation does not
depend on treating positions as independent.

The two predictors of an illegal move separate as well. Whole-board error reaches
an AUC of \result{cbAucAgg} [\result{cbAucAggLo}, \result{cbAucAggHi}] and
move-touched error \result{cbAucTouch} [\result{cbAucTouchLo},
\result{cbAucTouchHi}]. Bootstrapping the difference as a paired quantity gives
\result{cbAucDiff} [\result{cbAucDiffLo}, \result{cbAucDiffHi}], which excludes
zero.

A population-averaged logistic fit, with game as the cluster and an exchangeable
working correlation, gives cluster-robust coefficients on standardised
predictors: \result{geeTouch} (robust s.e.\ \result{geeTouchSE}) for move-touched
error against \result{geeAgg} (\result{geeAggSE}) for the aggregate, with
\result{geeConf} for low model confidence and \result{geeDepth} for depth. The
ordering survives robust inference.

\paragraph{Out of sample, not only in sample.} Coefficients can be significant and
still add little. Splitting by game and fitting on seventy percent of games, the
held-out log loss falls from \result{lldepthonly} with depth alone to
\result{llplusaggregate} adding whole-board error, and to \result{llplustouched}
adding move-touched error instead. Using both gives \result{llplusboth}, and
adding the model's own confidence \result{llplusbothandconfidence}. The
move-touched predictor buys several times what the aggregate buys out of sample,
and it buys most of it even when the aggregate is already present.

\paragraph{A held-out abstention rule.} Fitting and evaluating a selective
prediction rule on the same positions overstates it. Refitting on training games
and evaluating on held-out games, the illegal rate among retained positions at
half coverage is \result{hoComb05} for a rule combining the model's confidence
with probe certainty, against \result{hoConf05} for confidence alone and
\result{hoCert05} for probe certainty alone, from a base rate of
\result{hoBase}. An oracle-verified state monitor would reach
\result{hoBound05}, which remains the bound rather than a method: establishing it
at inference needs the ground truth the rule exists to avoid.

"""

once(r"""\subsection{A constructive edit: moving a piece in the model's belief}""",
     SEC + r"""\subsection{A constructive edit: moving a piece in the model's belief}""",
     "stats section")

open(p, "w", encoding="utf-8").write(s)
print("patched: clustered inference section")
