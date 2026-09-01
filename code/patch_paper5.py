"""Write the scale and intervention findings into the results sections."""
import os

p = os.path.join(os.path.dirname(__file__), "..", "paper", "main.tex")
s = open(p, encoding="utf-8").read()


def once(old, new):
    global s
    assert old in s, old[:70]
    s = s.replace(old, new, 1)


SCALE = r"""
Horizon is read off the illegal-move curve at the ply where it first crosses five
percent, interpolated between bucket midpoints. We report the interpolated value
because the bucketed one returns the same bucket for every condition here and so
cannot discriminate between them; the underlying curve is the same either way.

Across the ladder the horizon grows with scale and grows slowly. The baseline at
\result{paramsAttn} parameters reaches \result{hintAttn} ply. Adding depth at
constant width, \result{paramsAttnDeep} parameters, reaches
\result{hintAttnDeep}. Adding width, \result{paramsAttnWide} parameters, reaches
\result{hintAttnWide}. Roughly doubling the parameter count buys a few ply, which
is consistent with H2 and with the cost mismatch of Section~\ref{sec:bottleneck}:
scaling helps, and it does not buy back a missing mechanism cheaply.

The question that matters more for the reach of this paper is not whether the
horizon moves with scale, which it does, but whether the mechanism does.
Figure~\ref{fig:scale}(b) reports belief consistency and its matched control at
every rung. If the coherence of errors is stable across the ladder while the
horizon shifts, then what we have characterised is a property of how these models
maintain state rather than an artifact of the smallest ones.
"""

INTERVENTION = r"""
The recurrent channel gives a clean and slightly uncomfortable answer.

With the auxiliary objective active, state decodability rises sharply, from
\result{occ40Attn} for the attention-only baseline to \result{occ40AlsbOne} at
matched parameters and matched tokens, a difference many times the seed spread.
The architecture without the objective, \result{occ40AlsbZero}, is
indistinguishable from the baseline, as is the parameter-matched control at
\result{occ40AttnPM}. So the gain comes from supervising the channel, not from
adding it, and not from the parameters it costs.

Behaviour barely follows. The illegal-move rate improves from
\result{ill40Attn} to \result{ill40AlsbOne}, and the horizon from
\result{hintAttn} to \result{hintAlsbOne} ply. Set that beside the scale ladder:
the attention-only model at \result{paramsAttnWide} parameters reaches almost
exactly the same state decodability, \result{occ40AttnWide}, and converts it into
a much larger behavioural gain, \result{ill40AttnWide} and
\result{hintAttnWide} ply.

Two models with the same measured world model, then, and very different
behaviour. This is the paper's thesis arriving from the other direction. In
Section~\ref{sec:coupling} aggregate decodability failed to predict behaviour
across positions within a model; here it fails to predict behaviour across
models. An auxiliary loss that makes the state linearly available does exactly
what it is asked to do, and being available is not the same as being used.

We therefore report H3 as supported only in its weakest form and H4 as not
supported in the form we predicted. We had expected an intervention that improved
state tracking to improve behaviour through it. What we observe is an
intervention that improves the measurement of state tracking far more than it
improves state tracking's consequences. Anyone using auxiliary state supervision
as a training signal, or probe accuracy as a progress metric, should treat that
gap as the default expectation rather than the surprise.
"""

once(r"""\subsection{Scale}
\label{sec:scale}""",
     r"""\subsection{Scale}
\label{sec:scale}
""" + SCALE)

once(r"""attention-only control whose feed-forward block absorbs the same parameter count.""",
     r"""attention-only control whose feed-forward block absorbs the same parameter count.
""" + INTERVENTION)

open(p, "w", encoding="utf-8").write(s)
print("patched: scale and intervention findings written in")
