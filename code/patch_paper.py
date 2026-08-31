"""One-off: update the manuscript to match measured reality."""
import os

p = os.path.join(os.path.dirname(__file__), "..", "paper", "main.tex")
lines = open(p, encoding="utf-8").read().split("\n")

# --- remove the two placeholder tables, replace with generated ones
i0 = next(i for i, l in enumerate(lines) if l.startswith(r"\begin{table}")
          and "Planning horizon" in lines[i + 2])
i1 = i0
depth = 0
for i in range(i0, len(lines)):
    if lines[i].startswith(r"\begin{table}"):
        depth += 1
    if lines[i] == r"\end{table}":
        depth -= 1
        if depth == 0 and "Transfer to mathematical" in "\n".join(lines[i0:i]):
            i1 = i
            break
lines = lines[:i0] + [r"\input{tables}", ""] + lines[i1 + 1:]
s = "\n".join(lines)

old = r"""Two consequences are testable. First, fidelity should degrade with $t$ even when
$t$ is far inside the context window, because the re-derivation circuit must
compose more update steps within a fixed layer budget --- \emph{depth of
composition}, not distance of retrieval, is the binding constraint. Second,
fidelity should peak at intermediate layers and \emph{fall} toward the output
layers as representations specialise for next-token prediction; the state is a
means, not the target."""

new = r"""Two consequences are testable. First, fidelity should degrade with $t$ even when
$t$ is far inside the context window, because the re-derivation circuit must
compose more update steps within a fixed layer budget --- \emph{depth of
composition}, not distance of retrieval, is the binding constraint. This is borne
out (\S\ref{sec:results}).

Second, we predicted that fidelity would peak at intermediate layers and
\emph{fall} toward the output as representations specialise for next-token
prediction, the state being a means rather than the target. \textbf{This
prediction is false.} State decodability rises monotonically through the layer
stack and is maximal at the final block (\S\ref{sec:results}). The natural
reading is that for this task the state \emph{is} close to the target: the next
legal move is a direct function of the position, so a representation that makes
the position linearly available is also a good representation for the
language-modelling head. We record the failed prediction rather than deleting
it, since it bears on where a state carrier should be inserted --- our $k$-spaced
insertion was chosen before this was known."""

assert old in s, "prediction paragraph not found"
s = s.replace(old, new)
open(p, "w", encoding="utf-8").write(s)
print("patched: placeholders removed, falsified prediction recorded")
