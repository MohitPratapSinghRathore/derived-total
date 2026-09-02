# FR2: program execution traces as the third domain

The question this answers is the one the chess arm cannot: **is this a fact about
state tracking, or a fact about chess?** It is chosen over a natural-language
arm deliberately. Matched general-corpus pretraining costs thousands of dollars
and still lands at a scale nobody finds convincing, whereas execution traces need
no pretraining at all, carry an exact oracle for free, and reach 100M parameters
on a single rented A100 for a few hundred dollars.

## Why this domain and not another

Three properties of chess made it work as an instrument, and this domain has all
three, which none of our seven synthetic attempts did.

1. **An exact oracle at every token.** An interpreter knows the value of every
   variable after every statement. No labelling, no approximation.
2. **Eager state maintenance is forced.** Predicting the next statement's effect
   requires the current environment, not just the last mention of one variable.
   This is the property the boxes task lacked, and why its attention-only model
   solved the task without a decodable state.
3. **A graded signal.** Predicting plausible next tokens rewards partial state,
   so there is a gradient of partial competence to climb. This is what the four
   arithmetic designs lacked.

It also has a property chess does not: the state is heterogeneous (integers,
lists, strings, references), so "action-relevant state" becomes richer than two
squares, and the measurement generalises past a spatial board.

## Task

Straight-line and lightly branching Python restricted to a small grammar:
assignment, arithmetic on integers, list append and index, conditionals on
comparisons, and bounded loops. Programs of 20 to 200 statements. The model is
trained on the token sequence of the program **only**, never on the values.

    x0 = 7
    x3 = x0 + 2
    xs = [x3, x0]
    xs[1] = x3 * 2
    if x3 > 8: x0 = x0 - 1
    ...

## The oracle and the three measurements

Run each program under `sys.settrace`, recording the full environment after every
statement. That gives `s_t` for every token boundary, exactly as the chess rules
engine did.

**E_state.** Linear probes on the residual stream predicting each live
variable's value, fitted on a split disjoint from evaluation, scored against a
per-variable majority baseline and a randomised-weight control.

**Behavioural observable.** The analogue of an illegal move is a **type or
reference error**: the model proposes a statement that is invalid in the true
environment (indexing past the end of a list, using an undefined name, arithmetic
on a string). This is decidable by the interpreter, which is what makes it the
right analogue.

**Belief consistency, the central test.** Reconstruct the environment the probe
says the model believes, and ask whether the invalid statement is **valid in that
believed environment**. Controls carry over unchanged: the same statement scored
against another position's believed environment at equal depth, an arbitrary
invalid statement scored against the model's own believed environment, and the
decoding ceiling from statements that were genuinely valid.

**Causal test.** Edit the probe direction encoding one variable's value toward a
counterfactual, with the same calibration sweep, and measure whether the emitted
statement follows the induced belief. The manipulation check is direct here: read
the variable back off the probe after the edit.

## What would confirm, and what would refute

Confirmation is not "the numbers are similar to chess". It is:

- belief consistency well above both controls, with the enrichment surviving
  normalisation by the decoding ceiling;
- the action-relevant measure beating the aggregate by a margin that grows with
  depth, as in chess;
- the causal edit moving behaviour while a norm-matched random edit does not.

Refutation is equally well defined, and worth stating because it is a live
possibility. If invalid statements are **not** valid under the decoded
environment, then coherent error is a property of chess, where the action space
is small and highly structured, and not of state tracking in general. That result
would materially weaken the paper's framing and we would report it.

## Scale and budget

Train 30M, 100M and 300M parameter decoders on the same corpus, three seeds at
the two smaller sizes. On one A100 this is a few days of wall time and a few
hundred dollars. The point of the ladder is not the absolute horizon but whether
belief consistency, normalised by the ceiling, keeps rising as it does in chess
from 3.4M to 39.9M.

## Order of work

1. Generator and `sys.settrace` oracle, with the same validation the chess
   pipeline had: verify the recorded environment reproduces the interpreter's on
   a held-out sample before training anything.
2. Compute the generator's token-type entropy in advance. A model sitting at that
   value has learned the format and no content, which is the cheap diagnostic
   that would have saved four of our seven failed designs.
3. Baseline at 30M. Confirm the behavioural observable degrades with depth at
   all. If it does not, the programs are too easy and the grammar widens before
   anything else proceeds.
4. Probes, controls, belief consistency, causal edit.
5. Scale ladder.
