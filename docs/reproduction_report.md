# Reproduction report

Phase 1 status: repository, datasets, preprocessing, architecture, attention, and training flow are documented. The isolated environment was created with Python 3.11.14 and the pinned packages recorded in `environment.md`.

`scripts/smoke_test.py` passed on KIBA: it loaded two samples, produced drug `(2,100)`, protein `(2,1200)`, and prediction `(2,1)` tensors, computed MSE, ran backward, and completed an AdamW step.

Before the portability fixes, the unmodified `AttentionDTA_main.py` correctly loaded/shuffled Davis but failed before training because this host has no CUDA build/device. The current source selects an available CUDA, MPS, or CPU device.

For this Apple M4 host, `scripts/run_original_apple_silicon.py` is implemented and tested in adapter-check mode. It detects MPS and routes the training process to MPS if available, otherwise CPU. A one-epoch Davis drug-disjoint run also completed on MPS with test MSE `0.75205547`, MAE `0.55114901`, and R² `0.07625249` on 14 unseen drugs; this is a partial cold-start result, not a paper reproduction.
