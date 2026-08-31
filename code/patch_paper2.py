"""One-off: add the second domain, correct the abstract/contributions, and make
the limitations section reflect what was actually run."""
import os

p = os.path.join(os.path.dirname(__file__), "..", "paper", "main.tex")
s = open(p, encoding="utf-8").read()

# ---------------------------------------------------------------- abstract
old_ab = r"""the branching tree, giving $O(1)$ per-token state memory. We pre-register the full
evaluation, including transfer to long-horizon mathematical reasoning (GSM8K,
MATH, competition-style multi-step problems), under a protocol in which the
diagnostic and the architecture are evaluated on disjoint splits."""
new_ab = r"""the branching tree, giving $O(1)$ per-token state memory. We evaluate on two
domains with exact state oracles --- chess from notation, and a synthetic
long-horizon variable-tracking task whose answers require computing over the
tracked state --- under a protocol in which probes and evaluation use disjoint
splits. The full transfer test to natural-language mathematical reasoning
specified in our protocol was not run; we state precisely what that leaves open."""
assert old_ab in s
s = s.replace(old_ab, new_ab)

# ------------------------------------------------------------- contributions
old_c = r"""\item \textbf{Cross-domain transfer protocol.} A pre-registered test of whether an
architecture selected on chess-derived diagnostics improves long-horizon
mathematical reasoning (\S\ref{sec:transfer})."""
new_c = r"""\item \textbf{A second domain with an exact oracle.} A synthetic long-horizon
variable-tracking task, arithmetic rather than spatial, on which the same
decomposition is applied --- testing whether findings obtained with the chess
instrument are properties of long-horizon state tracking or of chess
(\S\ref{sec:synth})."""
assert old_c in s
s = s.replace(old_c, new_c)

# ------------------------------------------------------------------ domain 2
anchor = r"""\subsection{Results}"""
synth_sec = r"""\subsection{Domain 2: long-horizon variable-state tracking}
\label{sec:synth}

Chess is one domain, and a reviewer is entitled to ask whether anything measured
with it is a fact about long-horizon state tracking or a fact about chess. We
therefore repeat the decomposition on a second task built to share only the
property we rely on --- an exact per-token state oracle --- while differing in
almost everything else.

A program updates $8$ integer variables modulo $100$ over $5$ to $40$ steps, each
step of the form $v_i \leftarrow v_j \oplus c$, and then queries
$(v_a + v_b) \bmod 100$. The state is arithmetic rather than spatial; updates are
sparse (one variable per step) rather than structured by movement rules; and the
answer requires a computation \emph{on top of} the retrieved state, so the
conditioning that defines $E_{\mathrm{plan}}$ is exact rather than
probe-estimated: we know precisely which two variables the answer depends on, and
can ask whether both are decodable at the query token. Depth is the number of
steps, a clean horizon axis.

This is deliberately \emph{not} the natural-language mathematics transfer test of
\S\ref{sec:transfer}, and we do not present it as one. It tests the weaker
question --- does the architectural intervention generalise beyond the instrument
it was designed against? --- which is answerable with the compute available.

\subsection{Results}"""
assert anchor in s
s = s.replace(anchor, synth_sec, 1)

# ----------------------------------------------------------------- limitations
old_lim_start = s.index(r"\section{Limitations}")
old_lim_end = s.index(r"\section{Conclusion}")
new_lim = r"""\section{Limitations}

\paragraph{Scale.} Our models are $7.3$M parameters trained on roughly $9$M tokens
of chess notation. This is orders of magnitude below both the chess-specific
literature and any frontier reasoning system. Small undertrained models are
\emph{easier} to study, because they fail within a measurable range --- that is
why the instrument works at all --- but it means our central curves describe
models that a reviewer may reasonably regard as unrepresentative. The depth and
width axes address this only weakly, with two additional points. We claim the
decomposition and the measurement methodology generalise; we do not claim the
particular horizon values do.

\paragraph{The transfer experiment was not run.} \S\ref{sec:transfer} specifies a
test on GSM8K, MATH and competition-style problems, requiring two matched
general-corpus pre-trainings. That is beyond the single consumer GPU available
for this work, and the experiment is absent rather than approximated. The
synthetic domain of \S\ref{sec:synth} tests a weaker claim --- generalisation to
another long-horizon state-tracking task --- and should not be read as evidence
about natural-language mathematics. Whether an architectural fix selected on
oracle-bearing domains helps where no oracle exists remains open, and it is the
question that most determines whether this line of work matters.

\paragraph{Oracle dependence.} Auxiliary state supervision needs an oracle during
training. It exists for chess and for synthetic programs; it does not exist for
general corpora. The $\lambda{=}0$ condition measures how much of the effect
survives without it, and that condition is the one that would have to carry any
practical application.

\paragraph{Fixed-width state.} A constant $d_s$ is a hard capacity ceiling. Tasks
whose state grows without bound --- open-ended fact accumulation, dialogue
history --- should not benefit, and we predict that explicitly rather than
leaving it as an untested boundary.

\paragraph{Probes show presence, not use.} Linear decodability establishes that
information is present and linearly available; it does not establish that the
model reads it. The patching experiment addresses this directly but bounds the
concern rather than eliminating it: a null patching result would be evidence
against use, while a positive one shows the direction is causally live at the
layer tested.

\paragraph{A revised metric.} The horizon $H_\varepsilon$ was pre-registered on
exact-position fidelity. On these models that quantity reaches zero by roughly
ply $20$ in \emph{every} condition and therefore cannot discriminate between
them. We report it, and additionally report horizons defined on occupied-square
accuracy and on the illegal-move rate. The revision was made after inspecting
baseline data and before evaluating any ALSB condition, and all definitions are
applied unchanged across conditions; we flag it here rather than presenting the
revised metric as the original plan.

\paragraph{Implementation.} The ALSB gate depends on $z_{t-1}$, so the recurrence
is not an associative scan and our implementation is a sequential loop, roughly
$5\times$ the cost of an attention block at $T{=}160$. Factorising
$W_g[h_t; z_{t-1}] = W_{gh} h_t + W_{gz} z_{t-1}$ permits precomputing the
$h$-term across all positions and would roughly halve this; we did not apply it
mid-experiment because it perturbs initialisation and would have made seeds
non-comparable.

"""
s = s[:old_lim_start] + new_lim + s[old_lim_end:]

open(p, "w", encoding="utf-8").write(s)
print("patched: domain 2 added, abstract/contributions corrected, limitations rewritten")
