# Content Rot

Manuscript: **State-Space Bottlenecks in Autoregressive Transformers — Quantifying
Planning Horizons and Latent Hallucination in Search-Free Reasoning.**

Target venues: *Neural Networks*, *IEEE T-NNLS*, *Machine Learning*.

## Status

The paper is written through Limitations/Conclusion. **No experiments have been
run.** The two result tables in `paper/main.tex` §7.3 contain placeholder cells
marked `[Pending run]` — they are not results and must not be presented as such.
Everything in §5–§7 is a pre-registered protocol.

## Layout

- `paper/main.tex` — manuscript
- `paper/refs.bib` — bibliography (real works only)
- `protocol/experiments.md` — what has to be run to fill the tables

## Build

```
cd paper && pdflatex main && bibtex main && pdflatex main && pdflatex main
```

## What still has to happen before submission

1. Run the protocol in `protocol/experiments.md` and fill both tables.
2. Add the $F(t)$ fidelity-vs-ply curves and the per-layer fidelity heatmap —
   these are the paper's central figures and currently do not exist.
3. Expand Related Work with per-claim citations; the current section argues
   positions but under-cites.
4. Reconcile with any concurrent hybrid-attention/SSM long-horizon results.
