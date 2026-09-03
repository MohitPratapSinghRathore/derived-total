"""Revision two: narrow every claim the measurements do not license."""
import os

p = os.path.join(os.path.dirname(__file__), "..", "paper", "main.tex")
s = open(p, encoding="utf-8").read()


def once(old, new, label=""):
    global s
    assert old in s, "NOT FOUND " + label + ": " + old[:60]
    s = s.replace(old, new, 1)


# ------------------------------------------------------ title: act, not play
once(r"\title{\bf Coherent Errors: Chess Transformers Play Correctly\\on Misremembered Boards}",
     r"\title{\bf Coherent Errors: Chess Transformers Act\\on Misremembered Boards}", "title")

# ------------------------------------------------- abstract: drop "correctly"
once(r"""representation is read and not merely present. The effect strengthens with scale.
The model is not emitting noise; it plays correctly on a board it misremembers.""",
     r"""representation is read and not merely present. Belief consistency, normalised by
each model's legal-move recovery reference, rises monotonically across six model
sizes; we do not measure the causal edit at every size. A substantial identified
subset of long-horizon errors is therefore coherent with a degraded state
representation that the policy demonstrably reads.""", "abstract")

# ------------------------------------------- contributions: drop "usually"
once(r"""\item \textbf{Long-horizon errors are coherent.} An invalid action is usually a
valid action in the position the model believes it is in, established against
mismatched-belief and random-move controls, and strengthening across the range of
model sizes we can train.""",
     r"""\item \textbf{A substantial identified subset of long-horizon errors is
coherent.} Invalid actions are valid in the position the model appears to believe
far more often than matched controls allow, and the identified share rises across
the range of model sizes we train.""", "contribution 1")

# ------------------------------------- belief section: recovery reference wording
once(r"""The ceiling on what this test
can detect is \result{belLegAttn}, the rate at which genuinely legal moves are
recovered as legal in the believed board, which is below one because probe
decoding is lossy.""",
     r"""For reference, genuinely legal moves are recovered as legal in the believed board
\result{belLegAttn} of the time, which is below one because probe decoding is
lossy. We call this a legal-move recovery reference rather than a decoding
ceiling, and we do not divide by it to estimate an unobserved true rate: that
would assume probe sensitivity is the same on legal and illegal cases, which we
have not established. What the comparison licenses is that a substantial
identified subset of illegal moves is consistent with the decoded state, not that
most of them are.""", "ceiling wording")

# ---------------------------------------------- memory dominates, softened
once(r"""We take the honest reading. Memory is the dominant component and the better
characterised one, and it is not the only one.""",
     r"""We take the honest reading. The identified state-consistent component is
substantial, stricter state conditioning reduces measured planning loss, and a
residual depth-dependent planning component remains. Calling memory dominant
would overstate what a lossy probe and small conditioned samples support.""",
     "memory dominates")

# ------------------------------------------- self-consistency, softened
once(r"""Sampling several chains
and voting will not help if every chain is drawn from the same drifted state.""",
     r"""Sampling several chains
and voting may offer limited benefit when independently sampled continuations
share the same state distortion, though we did not test repeated sampling or how
far drift is correlated across samples.""", "self-consistency")

# -------------------------------- H_fid: policy-conditioned, not independent
once(r"""\subsection{A horizon that does not depend on behaviour}""",
     r"""\subsection{A policy-conditioned state-fidelity horizon}""", "hfid title")

once(r"""defined on the representation largely avoids this, because it does not depend on
the confidence of the output. It is not wholly independent of the model's
behaviour, since the squares it scores are those the chosen move touches, but it
reads no probability mass and so a model that spreads its mass more evenly gains
nothing from doing so.""",
     r"""defined on the representation is less exposed, because it does not read the
confidence of the output: a model that spreads its mass more evenly gains nothing
from doing so. It is not policy-independent, and we do not claim it is. The
squares it scores are those the selected move touches, and that move comes from
the output distribution, so a policy change can alter which squares are scored
even with the state representation unchanged. It is a policy-conditioned
state-fidelity horizon. A genuinely policy-independent comparison would score a
fixed set of components, for instance the recorded continuation move, every
oracle-legal move averaged uniformly, or a candidate set shared across models.""",
     "hfid softened")

# --------------------------- data split: game-disjoint, not prefix-disjoint
for old, new in [
    ("split by \\emph{game} rather than by position, so no\nprefix is shared between the probe-fitting split and the evaluation split",
     "split by \\emph{game}, so no position from a\ngiven game appears in more than one split"),
    ("split by \\emph{game} so no prefix is shared between probe\nfitting and evaluation",
     "split by \\emph{game}, so no position from a given game appears in more\nthan one split. Splitting by game does not make the splits prefix-disjoint,\nsince distinct games share openings, and we do not claim it does"),
]:
    if old in s:
        s = s.replace(old, new, 1)

open(p, "w", encoding="utf-8").write(s)
print("patched: title, abstract, contributions, ceiling, memory, self-consistency, "
      "hfid, split wording")
