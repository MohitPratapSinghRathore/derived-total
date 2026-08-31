# Experimental protocol

Everything below must be executed before any number enters the paper.

## 1. Data

- Source: public PGN archive (e.g. Lichess monthly dumps), filtered to games with
  both players above a fixed rating floor, deduplicated by move-sequence hash.
- Splits by *game*, never by position, to prevent prefix leakage between the
  probe-fitting split and the evaluation split.
  - `train-lm` — language-model pre-training
  - `train-probe` — probe fitting only
  - `eval` — all reported numbers; disjoint from both above
- Tokenisation: per-move vocabulary (UCI), no board rendering in the input.
- Tactical puzzles: separate set, stratified by required combination length
  (2/4/6/8/10 ply), used only for the lookahead axis.

## 2. Oracle labels

For each game in `train-probe` and `eval`, replay with a rules engine and emit,
per ply, the exact state: 64-square occupancy, side to move, castling rights,
en-passant target. This gives per-token multiclass targets for the probes and for
`L_state`.

## 3. Model grid

| Axis | Values |
|---|---|
| depth | 8, 12, 16, 24 |
| width | 256, 512, 768 |
| variant | attention-only; ALSB λ=0; ALSB λ>0 |

Matched token budget across all cells. Plus one parameter-matched attention-only
control against ALSB λ>0 (widen the FFN to absorb the ALSB parameter count), so a
gain cannot be attributed to capacity.

ALSB hyperparameters to sweep: `d_s ∈ {32, 64, 128}`, insertion period
`k ∈ {2, 4, 8}`, `λ ∈ {0.1, 0.5, 1.0}`.

## 4. Measurements

**State fidelity `F^(l)(t)`** — per-square linear heads on frozen activations at
every layer, evaluated per ply bucket. Report the full curve and the per-layer
heatmap, not just `H_0.05`. Include Hewitt–Liang control tasks (shuffled labels)
and a randomised-weight model baseline, or the probe result means nothing.

**E_state** — illegal-move rate per ply bucket, plus probe/oracle disagreement
rate. Report both; they should agree in trend.

**E_plan** — engine centipawn loss at fixed depth, conditioned on the emitted move
being legal *and* the probe agreeing with the oracle. Blunder threshold fixed in
advance (e.g. ≥100 cp) and not tuned.

**Puzzles** — accuracy by required combination length.

**Causal patching** — take probe-identified state directions for square *sq*,
patch them toward a counterfactual occupancy, and measure whether the emitted move
distribution shifts consistently with the counterfactual position. Report effect
size and the fraction of squares where patching succeeds.

## 5. Transfer

Two decoder LMs, identical corpus/budget/schedule, differing only in ALSB. For the
ALSB model, `L_state` is supervised on a synthetic arithmetic/state subcorpus where
an oracle exists (intermediate variable values), not on general text.

Evaluate on GSM8K, MATH, competition-style multi-step problems — **stratified by
ground-truth solution length**. H5 is an interaction prediction; a main-effect-only
report does not test it.

Report the arithmetic analogue of `E_state`: correctness of intermediate values
extracted from the generated chain, conditioned on the chain reaching that step.

## 6. Statistics

- ≥3 seeds per cell; report mean ± std, not best-of.
- Bootstrap CIs over games (the resampling unit is the game, not the position).
- Pre-register H1–H5 before running §5; record any deviation in the paper.
- If H4 fails — i.e. ALSB improves `E_plan` as much as `E_state` — report it. That
  outcome falsifies the stated mechanism even if end-task accuracy improves.
