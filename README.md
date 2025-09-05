# FASE

FASE (Finite-throughput Analysis with Σ-aware enhancement) is a research
prototype for analyzing information-limited physical systems.  The code in this
repository implements the **ICU** capacity-limited law in contrast to
unbounded power-law scaling (FCT).  It provides k-fold cross validation,
Σ-weighted residualization, and a symbolic grammar search (OG‑SET) for building
interpretable models.

## Theory

The ICU model assumes that signals increase linearly with a drive variable until
a context-agnostic throughput ceiling is reached; additional drive compresses
response.  Power-law models lack such a ceiling.  Model comparison using AICc,
Vuong tests, and information criteria in bits helps determine which description
fits a dataset better.  Hierarchical fits can share the global capacity while
allowing dataset-specific transduction parameters.

## Implementation

`FASE_v21.py` provides:

* Σ-aware ridge regression and residualization with cached whitening matrices.
* A k‑fold driver with out‑of‑fold metrics and stability selection.
* OG‑SET feature generation and selection using an MDL criterion measured in
  bits.
* Optional comparison to the PySR symbolic-regression baseline.

Run the demo on bundled PMLB datasets:

```bash
python FASE_v21.py
```

Results include per-fold scores, operator stability frequencies, and evidence in
bits for accepted blocks.

## References

* A finite-throughput view of nanoscale heat transfer and high-energy data
  streams: empirical tests of a capacity-limited law.
