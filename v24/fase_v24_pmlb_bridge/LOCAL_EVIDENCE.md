# Local evidence included with the bridge probe

These runs validate the mechanics of the artifact; they do not replace the frozen 20-fold external run.

## Two-fold smoke run

- Horizon: `SUPPORTED_EXTERNAL_INDICATION`
- Ridge mean OOF R²: **0.7436**
- Safe v23.1 recommendation mean OOF R²: **0.7434**
- Raw v23 minimum OOF R²: **-368.8746**
- Raw refined v23.1 minimum OOF R²: **-422.9211**
- Stable modes: **1**
- Pair-shuffle false promotions: **0**
- Target-shuffle false promotions: **0**

The smoke run intentionally uses very small folds. It exposed severe numerical instability in the raw symbolic branches on one fold. The non-degrading recommendation remained stable by retaining the raw-feature ridge anchor. This is reported as a capability boundary, not erased.

## Preliminary quick trajectory

- Horizon: `SUPPORTED_EXTERNAL_INDICATION`
- Completed folds: **3**
- Ridge mean OOF R²: **0.7915**
- v23 mean OOF R²: **0.7784**
- v23.1 recommendation mean OOF R²: **0.7814**
- Real promotions: **15 / 24**
- Stable modes: **6**
- Pair-shuffle false promotions: **0 / 24**
- Target-shuffle false promotions: **0 / 24**

The quick evidence uses a partial local matrix without PySR and is included only to show the likely conversion shape. The full 20-fold runner determines the final horizon classification.
