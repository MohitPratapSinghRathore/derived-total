"""Add the causal intervention section and the domain-2 negative result."""
import os

p = os.path.join(os.path.dirname(__file__), "..", "paper", "main.tex")
s = open(p, encoding="utf-8").read()

CAUSAL = r"""\subsection{Inducing a false belief changes the action}
\label{sec:causal}

Belief consistency is observational. It shows the model errs coherently with a
decayed state without showing that the state representation drives the action.
We therefore intervene. At a position where the side to move genuinely occupies a
square, we push the residual stream along the probe direction encoding "this
square is empty" and ask whether the model stops proposing moves from it.

Two safeguards decide whether such an experiment means anything, and both are
reported. The manipulation check asks whether the edit changed the decoded belief
at all: at the strength we use it does, in \result{patchFlipAttn} of cases. The
norm-matched control applies a random direction of identical magnitude at the
same position, which separates belief from mere perturbation sensitivity.

Edit strength has to be calibrated rather than chosen, because a large enough
push degrades the forward pass and will move behaviour for reasons having nothing
to do with state. Table~\ref{tab:patchcal} reports the sweep. Below a quarter of
the residual-stream magnitude the edit does not reliably flip the belief; above
roughly one, the random control also starts destroying the output distribution,
so any effect there is damage rather than belief. We report the intervention at
\result{patchAlphaAttn}, inside the window where the manipulation succeeds and
the control does not.

At that strength, editing the state direction removes \result{patchCutAttn} of
the probability mass on moves from the affected square, decreasing it in
\result{patchDecAttn} of \result{patchNAttn} cases. The norm-matched random edit
removes \result{patchRandCutAttn} and decreases the mass in
\result{patchRandDecAttn} of cases, which is indistinguishable from a coin flip.
The state representation is not a decorative correlate of the model's
computation. It is read.

\subsection{A second domain that did not work, and why}
\label{sec:synthfail}

We attempted a second oracle-bearing domain, a program updating integer variables
with interleaved queries, to test whether these findings are properties of
long-horizon state tracking or of chess. Four designs failed to leave chance, and
we report the sequence because the diagnosis is more useful than the intention.

The first placed a single query at the end of a sequence of random updates.
Training loss settled at the token-type entropy of the generator, which is the
value a model achieves by learning the format and nothing else, and the reason is
arithmetic: the only learnable signal was one answer token in roughly one
hundred and thirty, so state tracking carried under one percent of the gradient.
Interleaving queries raised that share and did not move accuracy. Reducing the
modulus and the number of variables did not move it either. A final version
separated a pure read-out query from a computed one, so that state tracking could
succeed without the arithmetic succeeding, and read-out accuracy still sat at
chance.

We draw two conclusions rather than one. The narrow conclusion is that this task
family asks a small model to learn modular arithmetic before any state tracking
becomes observable, and that our compute did not reach it. The broader one is a
warning for anyone building synthetic state-tracking benchmarks: a task can be
perfectly well specified, carry an exact oracle, and still measure nothing,
because the quantity of interest contributes almost none of the training signal.
The loss curve diagnoses this cheaply. A model that has learned the format and no
content sits at the generator's token-type entropy, which is computable in
advance and worth computing before trusting such a benchmark. We report this in
place of a cross-domain result, which we do not have.

"""

anchor = r"""\subsection{A degenerate metric, declared}"""
assert anchor in s
s = s.replace(anchor, CAUSAL + anchor, 1)

# a calibration table, generated content lives in tables.tex but this one is fixed text
CAL = r"""
\begin{table}[t]
\centering\small
\caption{Calibrating the edit strength for the causal intervention. Strength is
in units of the residual stream's own magnitude. The usable window is where the
manipulation check succeeds while the norm-matched random control still behaves
like a coin flip. Values from \texttt{results/attn\_8L256\_s0\_patch\_sweep.json}.}
\label{tab:patchcal}
\begin{tabular}{cccccc}
\toprule
strength & belief flipped & state edit: mass after & state decreased
& random: mass after & random decreased \\
\midrule
0.10 & 0.652 & 0.046 & 0.980 & 0.076 & 0.540 \\
0.25 & 0.996 & 0.020 & 0.988 & 0.078 & 0.452 \\
0.50 & 1.000 & 0.005 & 0.996 & 0.065 & 0.484 \\
1.00 & 1.000 & 0.000 & 0.996 & 0.054 & 0.556 \\
2.00 & 1.000 & 0.000 & 0.996 & 0.025 & 0.712 \\
6.00 & 1.000 & 0.000 & 1.000 & 0.013 & 0.784 \\
\bottomrule
\end{tabular}
\end{table}

"""
s = s.replace(r"""\subsection{Inducing a false belief changes the action}""",
              CAL + r"""\subsection{Inducing a false belief changes the action}""", 1)

# contributions: promote the causal result
s = s.replace(
    r"""\item The belief-consistency result: long-horizon errors are coherent actions on
a decayed state, established against mismatched-belief and random-move controls.""",
    r"""\item The belief-consistency result: long-horizon errors are coherent actions on
a decayed state, established against mismatched-belief and random-move controls,
and supported causally by an intervention that induces a false belief and changes
the action accordingly, with a calibrated edit strength and a norm-matched
control.""")

# limitations: L5 is now stronger, and the domain-2 failure is a stated limitation
s = s.replace(
    r"""\item[L5] \textbf{Probes show availability, not use.} Linear decodability shows
information is present and linearly available. The belief-consistency result is
much stronger evidence of use than correlation would be, but it remains
observational rather than interventional.""",
    r"""\item[L5] \textbf{Scope of the causal claim.} The intervention establishes that
the decoded state direction is read by the policy at the layer tested. It does
not establish which downstream computation consumes it, and a single-direction
edit is a coarse instrument.""")

s = s.replace(
    r"""\item[L2] \textbf{One domain family.} Chess carries an exact oracle by
construction, which is what makes it measurable and also what makes it
unrepresentative of open-ended reasoning.""",
    r"""\item[L2] \textbf{One domain family.} Chess carries an exact oracle by
construction, which is what makes it measurable and also what makes it
unrepresentative of open-ended reasoning. Our attempt at a second domain did not
reach a measurable regime (Section~\ref{sec:synthfail}), so the findings rest on
one instrument and should be read that way.""")

open(p, "w", encoding="utf-8").write(s)
print("patched: causal section, calibration table, domain-2 negative result")
