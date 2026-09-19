# Preregistration: Pythia partition test

**Written 2026-09-19, before any model was loaded or any loss computed.**

This file fixes the decisions that could otherwise be adjusted after seeing results.
It is committed before the first run and is not to be edited afterwards; if
something here turns out to be wrong, the correction goes in a new dated section at
the bottom marked as an amendment, with the reason, and the original text stays.

Note on naming: `PREREGISTRATION.md` in this repository is the protocol for the
chess experiments and is unrelated. This file covers only the Pythia partition test
in `code/pythia_partition.py`.

## What is being tested

Whether a per-category scaling decomposition and a separately fitted aggregate are
mutually inconsistent on a public model ladder in a domain we did not construct,
with categories we did not define.

Categories are the named sources of the Pile. The quantity per category is mean
token-level cross entropy. The ladder is EleutherAI's Pythia deduplicated suite.

## The identity under test

Loss is not an error rate, so the parts do not sum to the whole. The aggregate is
the **token-weighted mean** of the per-source losses,

    L(N) = sum_s w_s L_s(N),    w_s = tokens_s / sum_t tokens_t

This is a stated choice, not a property of the data: a different evaluation mixture
gives different weights and a different aggregate. The weights are fixed by the
evaluation corpus and the token budget below, not by any model, and are reported
alongside every result.

**Weights are held constant across the ladder.** All Pythia models share one
tokenizer, so the token count per source is expected to be identical at every rung.
This is verified rather than assumed (check 1b). If counts differ across rungs, the
run does not silently switch to per-model weights: it reports the discrepancy and
the analysis is reported as invalid until the cause is found.

## Check 1: in-range reproduction, and its tolerance

The aggregate is computed two independent ways at every rung:

  (a) **pooled**: total negative log likelihood over all documents of all sources,
      divided by total tokens;
  (b) **weighted**: sum_s w_s L_s, from the per-source means and token counts.

These are algebraically identical, so any disagreement is an implementation fault,
not a finding. The tolerance is therefore tight:

> **Check 1 passes at a rung if |a - b| / a <= 1e-6.**

> **Check 1b passes if every source's token count is identical at every rung**,
> exactly, with no tolerance, since the tokenizer is shared.

If check 1 fails at any rung, that rung's numbers are discarded and the failure is
reported. Nothing downstream is computed from a rung that fails check 1. No result
from this test enters any manuscript unless check 1 and 1b pass at every rung in the
ladder.

## Token budget and data

- Corpus: `monology/pile-uncopyrighted`, validation split, **streamed**, never
  downloaded in full.
- Budget: **200,000 tokens per source per model**. Reading of a source stops once
  the budget is reached. The same documents, in the same order, are used at every
  rung.
- Truncation: documents are truncated at 1024 tokens; the count charged against the
  budget is the number of predicted tokens, which is one fewer than the input length.
- The mirror omits some sources present in the original Pile. Whatever is absent is
  recorded and carried into the manuscripts' limitations. The absence is a property
  of the corpus and is not a reason to substitute a different dataset.

## Ladder

`EleutherAI/pythia-{70m,160m,410m,1b,1.4b,2.8b,6.9b,12b}-deduped`, ascending.
70m through 410m on CPU locally; 1b and above on free Kaggle T4s. Precision and
device for every rung are recorded per rung in the output, because they are not
uniform and a reader needs to know which rungs were quantised.

## Analysis, fixed in advance

1. Fit each source's loss as a power law in parameter count, log-log, ordinary least
   squares on the rung means.
2. Fit the aggregate independently the same way.
3. Divergence rate is `max_s a_s - b`, with `a_s` the weighted per-source exponents
   and `b` the aggregate's.
4. Breakdown scale is the smallest N **above the top of the observed range** at
   which the weighted parts exceed the fitted whole by more than **10 percent**,
   the same tolerance used for the chess ladder.
5. Intervals come from resampling the **size rungs** with replacement, 4000 draws,
   reported as 2.5th and 97.5th percentiles. The rung is the unit of variation, as
   elsewhere in this project.
6. Aggregate curvature: fit `log L = c + b log N + q (log N)^2`, centred, and
   bootstrap `q` over size rungs the same way. Curvature is called detected only if
   the bootstrap interval excludes zero.

## What counts as each outcome

Committed in advance so that neither result can be presented as the interesting one
after the fact.

- **Positive.** The divergence rate is positive and the breakdown scale falls at or
  below 1e12 parameters, which is within the range people extrapolate to. Reported
  as the decomposition and aggregate being inconsistent at scales of practical
  interest.
- **Negative.** The divergence rate is negative, or is positive with a breakdown
  scale far beyond any scale anyone extrapolates to. **This is reported plainly, in
  the abstract, as evidence that the conflict this project describes can be
  arithmetically real and practically negligible on a real ladder.** The chess
  result is not thereby withdrawn, but its generality is reported as bounded by this.
- **Void.** Check 1 or 1b fails anywhere, or the ladder is incomplete. Nothing is
  reported except the failure.

Nothing is tuned to produce a breakdown. The tolerance in step 4, the budget, the
weights, and the bootstrap are all fixed by this document before the first model
was loaded.

## What would make this test uninformative

Recorded now so it cannot be discovered conveniently later. The per-source losses
may be nearly parallel in log-log, in which case all `a_s` are close together, the
divergence rate is near zero, and the test has little power to detect anything. That
would be a real finding about the Pile decomposition, not a failure of the method,
and would be reported as such.
