"""Insert figures into the manuscript at the points where they carry the argument."""
import os

p = os.path.join(os.path.dirname(__file__), "..", "paper", "main.tex")
s = open(p, encoding="utf-8").read()


def once(old, new):
    global s
    assert old in s, old[:60]
    s = s.replace(old, new, 1)


# decay figure, in the decay subsection
once(r"""Occupied-square decodability falls from \result{occFirst} in the opening to""",
     r"""\begin{figure}[t]
\centering
\includegraphics[width=\linewidth]{decay.png}
\caption{Internal state and external behaviour decay together. (a) Occupied-square
state accuracy at the most decodable layer, against a majority predictor, a
label-permutation control, and a randomised-weight model of identical
architecture. (b) The illegal top-1 move rate and the probability mass the model
places on legal continuations.}
\label{fig:decay}
\end{figure}

Occupied-square decodability falls from \result{occFirst} in the opening to""")

# layer figure, in the bottleneck section
once(r"""our spacing was chosen before we knew.""",
     r"""our spacing was chosen before we knew.

\begin{figure}[t]
\centering
\includegraphics[width=0.62\linewidth]{layers.png}
\caption{State decodability by layer and depth. Contrary to our prediction, the
position is most linearly available at the final block rather than in the middle
of the stack.}
\label{fig:layers}
\end{figure}""")

# coupling figure
once(r"""We think this deserves emphasis because it is a negative result about a standard
practice.""",
     r"""\begin{figure}[t]
\centering
\includegraphics[width=\linewidth]{coupling.png}
\caption{(a) Within a fixed depth, the correlation between the number of
misremembered squares and playing an illegal move is close to zero. (b) The same
representation, scored only on the squares the move actually uses, separates
illegal from legal moves sharply.}
\label{fig:coupling}
\end{figure}

We think this deserves emphasis because it is a negative result about a standard
practice.""")

# belief figure
once(r"""Of the illegal moves the model plays, \result{belIllAttn} are legal in its own""",
     r"""\begin{figure}[t]
\centering
\includegraphics[width=\linewidth]{belief.png}
\caption{The errors are coherent. (a) Against depth, restricted to buckets with
at least thirty illegal cases. (b) Pooled: an illegal move is far more likely to
be legal in the model's own believed board than under another position's belief
at equal depth, or than an arbitrary illegal move is under the model's own
belief. The dashed line is the ceiling this test can detect, set by how often
genuinely legal moves are recovered as legal.}
\label{fig:belief}
\end{figure}

Of the illegal moves the model plays, \result{belIllAttn} are legal in its own""")

# patch calibration figure
once(r"""At that strength, editing the state direction removes \result{patchCutAttn} of""",
     r"""\begin{figure}[t]
\centering
\includegraphics[width=0.66\linewidth]{patch_calibration.png}
\caption{Choosing the edit strength. Too small and the belief does not flip; too
large and a norm-matched random direction also destroys the output distribution,
so any effect there is damage rather than belief. We report inside the shaded
window.}
\label{fig:patchcal}
\end{figure}

At that strength, editing the state direction removes \result{patchCutAttn} of""")

# scale figure placeholder, inserted into the scale section
once(r"""\subsection{Scale}
\label{sec:scale}""",
     r"""\subsection{Scale}
\label{sec:scale}

\begin{figure}[t]
\centering
\includegraphics[width=\linewidth]{scale.png}
\caption{(a) The horizon against model size. (b) Belief consistency and its
matched control against model size. The question is whether the mechanism is a
property of small models or survives scale.}
\label{fig:scale}
\end{figure}""")

open(p, "w", encoding="utf-8").write(s)
print("patched: figures placed")
