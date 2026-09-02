"""One pass over every place that still asserts the retracted claim, plus the
structural leftovers a reader would trip on.

The like-for-like comparison in Section 7.2 showed aggregate fidelity is the
weaker predictor, not an uninformative one. Four passages still said the older,
stronger thing, which reads as not having reread the discussion.
"""
import os

p = os.path.join(os.path.dirname(__file__), "..", "paper", "main.tex")
s = open(p, encoding="utf-8").read()


def once(old, new, label=""):
    global s
    assert old in s, "NOT FOUND " + label + ": " + old[:60]
    s = s.replace(old, new, 1)


# ------------------------------------------------- 1. discussion contradiction
once(r"""The practical content of this paper is a change in what to measure. Reporting
that a probe recovers a world model at some aggregate accuracy is weak evidence
that the model is load-bearing, because aggregate accuracy is nearly
uninformative about the next action. Measuring the same representation on the
components an action depends on is informative.""",
     r"""The practical content of this paper is a change in what to measure. Reporting
that a probe recovers a world model at some aggregate accuracy is weak evidence
that the model is load-bearing: aggregate accuracy predicts the validity of the
next action, and it predicts it poorly, and it gets worse with depth. Measuring
the same representation on the components an action depends on is substantially
more informative from the same activations.""", "discussion")

# ----------------------------------------------- 2. conclusion contradiction
once(r"""says little about what a model will do next. The same measurement, restricted to""",
     r"""is a weak guide to what a model will do next. The same measurement, restricted to""",
     "conclusion")

# ------------------------------------------------------- 3. section 7.8 recap
once(r"""Section~\ref{sec:coupling} aggregate decodability failed to predict behaviour""",
     r"""Section~\ref{sec:coupling} aggregate decodability predicted behaviour weakly""",
     "7.8 recap")

# --------------------------- 5. fold the stub subsection into the section above
once(r"""\subsection{Action-relevant fidelity does}

Restricting the measurement to the squares a move touches changes the picture.
The probe is wrong on those squares for \result{locIllAttn} of illegal moves and
\result{locLegAttn} of legal moves. The relevant question is not how much of the
world the model has lost but whether it has lost the part it is about to use.

""", "", "stub removal")

once(r"""restricted to the state an action consumes is several times more informative from
the same probe and the same activations.""",
     r"""restricted to the state an action consumes is several times more informative from
the same probe and the same activations. Stated as a conditional rather than a
ranking, the probe is wrong on the squares a move touches for
\result{locIllAttn} of illegal moves against \result{locLegAttn} of legal ones.
The relevant question is not how much of the world the model has lost, but
whether it has lost the part it is about to use.""", "fold stub")

# ------------------------------------------------------ 6. duplicate table
once("\\input{tab_abstain}\n\n", "", "duplicate abstain table")

# ------------------------------------------------------------ 7. H2 verdict
once(r"""\item[H2, supported.] The horizon grows slowly with scale, from
\result{hintAttn} ply at \result{paramsAttn} parameters to
\result{hintAttnWide} at \result{paramsAttnWide} (Section~\ref{sec:scale}).""",
     r"""\item[H2, supported.] The horizon grows slowly with scale across six rungs at
three seeds each, from \result{hintRung1} ply at \result{paramsSmall} parameters
to \result{hintRung6} at \result{paramsLarge} (Section~\ref{sec:scale}).""",
     "H2 verdict")

# ------------------------- 8. Scale section keeps the behavioural horizon only
once(r"""Horizon is read off the illegal-move curve at the ply where it first crosses five
percent, interpolated between bucket midpoints. We report the interpolated value
because the bucketed one returns the same bucket for every condition here and so
cannot discriminate between them; the underlying curve is the same either way.

Across the ladder the horizon grows with scale and grows slowly. The baseline at
\result{paramsAttn} parameters reaches \result{hintAttn} ply. Adding depth at
constant width, \result{paramsAttnDeep} parameters, reaches
\result{hintAttnDeep}. Adding width, \result{paramsAttnWide} parameters, reaches
\result{hintAttnWide}. Roughly doubling the parameter count buys a few ply, which""",
     r"""The behavioural horizon is read off the illegal-move curve at the ply where it
first crosses five percent, interpolated between bucket midpoints because the
bucketed read-off returns the same bucket for every condition and so cannot
discriminate between them. A second horizon defined on the representation is
introduced in Section~\ref{sec:hfid}; both are tabulated together there.

Across the ladder the behavioural horizon grows with scale and grows slowly,
from \result{hintRung1} ply at \result{paramsSmall} parameters to
\result{hintRung6} at \result{paramsLarge}, three seeds per rung. Roughly an
order of magnitude in parameters buys about half again as much depth, which""",
     "scale trim")

# ------------------------------------------------------------ 9. soften Hfid
once(r"""defined on the representation avoids this, because it never consults the output
distribution.""",
     r"""defined on the representation largely avoids this, because it does not depend on
the confidence of the output. It is not wholly independent of the model's
behaviour, since the squares it scores are those the chosen move touches, but it
reads no probability mass and so a model that spreads its mass more evenly gains
nothing from doing so.""", "soften hfid")

# -------------------------------------------------------- 10. data availability
once(r"""from which every reported value is generated are available from the authors.""",
     r"""from which every reported value is generated are available at
\url{https://github.com/MohitPratapSinghRathore/content-rot}.""",
     "data availability")

open(p, "w", encoding="utf-8").write(s)
print("patched: four contradictions, stub folded, duplicate table, H2, scale trim, "
      "hfid softened, repo link")
