# Pre-registration: decomposing long-horizon failure into state-tracking and policy components

Status: **S1, S2, S3 frozen. S5 arms frozen, thresholds inherit from S2.**
Frozen on 2026-09-05, before any repair edit has been applied.

This document is hashed and the hash cited in the manuscript. Any change after
freezing is recorded as a dated amendment below rather than by editing the text
above it, so the diff is legible.

---

## 0. Disclosure of prior observation

Honest pre-registration requires stating what was already seen, because it shaped
the design. The following were measured before freezing, on `attn_8L256_s0`,
70,078 positions from 1,200 held-out games, using probe and oracle only, with no
edits and no training:

| quantity | value |
|---|---|
| illegal top-1 rate, ply >= 20 | 0.1950 |
| belief consistency (illegal move legal on decoded board) | 0.3429 [0.3342, 0.3526] |
| policy share (belief correct at both move squares, among illegal) | 0.2047 [0.1971, 0.2131] |
| divergent squares, median, legal / illegal top-1 | 14.0 / 14.0 |
| belief correct at move squares, legal / illegal | 0.5091 / 0.2047 |

Two consequences, both of which changed the plan before it was frozen:

- Belief consistency replicates Paper 1's 0.355 by an independent code path,
  which is why the probe is treated as a validated instrument below.
- Natural divergence is **diffuse**, not sparse. The original S3 hypothesis
  (natural errors are sparse, induced errors diffuse) is false as written and is
  replaced in section 3 by a local-against-global coupling hypothesis.

No repair edit has been applied at time of freezing. S1's effect sizes are
unobserved.

---

## 1. S1 — repair validation

### 1.1 Primary observable

For each position entering the analysis set, let

- `m_ill` = the model's original top-1 move (illegal on the true board),
- `m*` = the highest-probability move that **is** legal on the true board.

The primary observable is the change under repair in the **forced-choice
preference**

```
r = P(m*) / (P(m*) + P(m_ill))
```

reported as `Δr` = r_post − r_pre.

Rationale, and a deliberate departure from "probability mass moved onto the
implied legal move". Repair is a perturbation and can flatten the distribution.
Raw `ΔP(m*)` rises under uniform flattening whenever P(m*) < P(m_ill), which is
the common case, so it can fake success exactly as illegal-move rate did in the
Paper 2 ablation. `Δr` is invariant to any uniform rescaling of the distribution
and therefore cannot be produced by hedging alone.

Secondary observables, all reported for every condition: `ΔP(m*)`, `ΔP(m_ill)`,
`Δ` entropy of the move distribution, `Δ` top-1 logit margin, and change in
illegal top-1 rate. The last is reported for comparability with prior work and is
**not** a success criterion.

### 1.2 Analysis set

Positions with ply >= 20 at which the model's top-1 move is illegal on the true
board **and** the probe-decoded belief disagrees with the oracle at at least one
of `m_ill.from_square`, `m_ill.to_square`. Positions failing the second condition
are the policy bucket, measured at 0.2047, and are excluded from S1 while being
reported as the ceiling in 1.4.

### 1.3 Conditions

All edits install probe-decoded occupancy directions at the residual stream of
the probe's best layer, using the constructive edit validated in Paper 1.

1. **correct repair** — install true occupancy at divergent action-relevant squares
2. **null repair** — same edit applied at action-relevant squares already correct
3. **norm-matched random** — random direction, matched norm, same squares
4. **wrong-target** — install true occupancy at the wrong square
5. **irrelevant-square** — install true occupancy at correct but action-irrelevant squares
6. **magnitude sweep** — correct repair at edit strengths {0.25, 0.5, 1.0, 2.0, 4.0}x calibrated

Condition 4 is the primary control. The claim requires correct repair to exceed
**wrong-target**, not merely to exceed no-op.

### 1.4 Entropy matching

Pre-edit top-1 entropy is binned into deciles computed on the analysis set. `Δr`
is computed within each decile and aggregated as the sample-weighted mean across
deciles. Both the stratified and unstratified estimates are reported. Confidence
intervals are bootstrap, 500 resamples, **clustered on games**.

### 1.5 Ceiling

The model's illegal top-1 rate on positions where belief is already correct at
both action-relevant squares. Repair cannot beat this number and the headline
effect is reported against it.

---

## 2. S2 — the decomposition

Four-way classification of each illegal top-1 move. Thresholds frozen here.

| bucket | rule |
|---|---|
| **memory** | repair (condition 1, calibrated strength) makes top-1 legal on the true board, and Stockfish 17 centipawn loss of the repaired move <= 100 relative to its best move |
| **mixed** | repair makes top-1 legal, centipawn loss > 100 |
| **policy** | not repaired to legal, belief already correct at both action-relevant squares |
| **unresolved** | not repaired to legal, belief wrong at action-relevant squares |

Stockfish 17, fixed depth 12, single thread, 100 ms per position, evaluated from
the true board with the side to move taken from the oracle.

Reported for all six ladder rungs and three seeds, with CIs clustered on games.

**Prior.** Belief consistency 0.343 and policy share 0.205 imply memory + mixed
should land near 0.34 and policy near 0.20. A split far outside that envelope is
treated as evidence of a threshold or implementation error before it is treated
as a finding.

---

## 3. S3 — the Balogh reconciliation (revised)

**Superseded hypothesis.** Natural divergence is sparse, induced divergence is
diffuse. Falsified pre-freeze: median 14 of 64 divergent squares for natural
errors, only 2% at or under 5 squares.

**Replacement hypothesis.** Coupling between probe-decoded belief and generation
is **local, not global**. Whole-board fidelity is weakly coupled to behaviour,
which is Balogh's finding, while belief at action-relevant squares is strongly
coupled, which is Paper 1's.

Pre-freeze support, stated as suggestive only: belief is correct at both move
squares in 0.509 of legal moves against 0.205 of illegal, a separation of +0.304,
where whole-board divergence differs by 0.523 squares of 64.

That comparison is **partly structural** — an illegal move is frequently illegal
precisely because its from-square is empty, so disagreement there is close to
definitional for part of the population. It therefore cannot establish the
hypothesis.

**Confirmatory test, which is not structural.** Regress `Δr` from S1 on
action-relevant divergence and on whole-board divergence entered jointly. The
hypothesis predicts a coefficient on action-relevant divergence with a CI
excluding zero, and a coefficient on whole-board divergence whose CI includes
zero. Because the divergent squares are here identified by the edit rather than
by what made the move illegal, the circularity above does not apply.

S3 now depends on S1 and is scheduled immediately after it.

---

## 4. S4 — external validity

Full pipeline on Karvonen's Chess-GPT checkpoints and Li's OthelloGPT. No
training. Pre-registered as a replication: the S2 split is expected to differ in
level across models but the ordering memory > policy is expected to hold.

---

## 5. S5 — interventions

Arms: baseline; uniform auxiliary board loss; action-weighted state loss;
transition-consistency loss; plus an inference-time verifier baseline priced in
compute per corrected move.

Three sizes x three seeds x four training arms = 36 runs. Seeds 0, 1, 2.

Every arm reported as its effect on the memory fraction, the policy fraction and
the unresolved remainder, entropy-matched per section 1.4. An arm that reduces
total error while leaving the memory fraction flat is reported as having improved
the policy, not the state tracking.

Explicit carried state is out of scope and named as the obvious architectural
test in the discussion.

**Stopping rule.** All 36 runs complete at a fixed step budget fixed in advance.
No interim analysis and no seed added after seeing results. If a run diverges it
is rerun with the same seed and the event is reported.

---

## 6. Kill criteria

Decided at **week 3**, not later.

1. If `Δr` for correct repair does not exceed **wrong-target** with a
   game-clustered CI excluding zero, the instrument is not valid and S5 has no
   metric.
2. If the effect vanishes under entropy matching (section 1.4), same conclusion.

On either trigger the programme becomes the reconciliation paper alone: probes
are weakly coupled to generation even for natural errors, which agrees with
Balogh and is publishable on its own. S5 is not attempted.

---

## 7. Amendments

None.
