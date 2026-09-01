"""Repair the second-domain section, which reported a task it never described."""
import os

p = os.path.join(os.path.dirname(__file__), "..", "paper", "main.tex")
s = open(p, encoding="utf-8").read()


def once(old, new):
    global s
    assert old in s, old[:70]
    s = s.replace(old, new, 1)


BRIDGE = r"""place of an easy cross-domain result.

\paragraph{A domain that does work.}
Abandoning arithmetic entirely, we built the second domain around an assignment
instead. Four objects sit in four boxes. The sequence states the initial
assignment, then applies operations that permute it, and after every operation
asks where one object now is. A \textsc{swap} names two boxes and moves their
contents without naming the objects, so the affected object's location cannot be
retrieved by attending to its last mention and has to be maintained. A
\textsc{move} names the object, which gives an anchor and keeps the task
learnable. The state is the assignment of objects to boxes, exact at every
token, and the answer is a single symbol, so belief consistency here is exact
rather than estimated: a wrong answer either is or is not the box the model's own
probe says holds that object.

Two further design choices were forced on us, and both are worth recording. Our
first attempt used five objects. That makes the state an element of $S_5$, the
smallest non-solvable symmetric group, which is precisely the case bounded-depth
sequence models are known to be unable to track \citep{merrill2024illusion}; we
had chosen the hardest available instance by accident. Four objects give $S_4$,
which is solvable. Our second attempt queried only occasionally, which left the
task all-or-nothing: with uniformly random updates and sparse queries a model
either tracks the permutation exactly or sits at chance, a structure that is hard
to learn from endpoint supervision. Querying after every operation supplies
partial credit, since a query one step after an operation is answerable directly,
and that is the foothold from which the model extends. Chess has this property
for free, because a plausible move needs only partial state.

With those two changes the task became measurable, and the outcome follows."""

once(r"""place of an easy cross-domain result.""", BRIDGE)

# the section now states four lessons, so the earlier count is wrong
once(r"""We draw two conclusions rather than one. The narrow conclusion""",
     r"""We draw two conclusions from the arithmetic attempts. The narrow one""")

# the introduction promised three results and the paper now carries more
once(r"""what the model actually does. Three results structure the paper.""",
     r"""what the model actually does. Three results carry the argument, and the rest of
the paper tests how far they reach.""")

open(p, "w", encoding="utf-8").write(s)
print("patched: second domain described before its results are reported")
