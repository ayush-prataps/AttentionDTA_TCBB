# Reproduce AttentionDTA Phase 1

This repository contains the official source plus portable device handling and a verified CPU/MPS-compatible smoke path. It does not claim a completed paper reproduction on the current macOS host.

## Local inspected environment

```bash
python3.11 -m venv .venv
.venv/bin/pip install 'torch==2.7.1' 'numpy==1.23.5' 'scikit-learn==1.3.2' 'tqdm==4.67.1' 'tensorboardX==2.6.4' 'prefetch-generator==1.0.3'
.venv/bin/python scripts/smoke_test.py
```

The smoke command is an explicitly documented compatibility check, not a complete original run.

## Apple Silicon launcher

On an Apple Silicon Mac, first verify device selection:

```bash
.venv/bin/python scripts/run_original_apple_silicon.py
```

To start the full Davis experiment (five folds and up to 500 epochs each):

```bash
.venv/bin/python scripts/run_original_apple_silicon.py --run
```

The launcher selects MPS when supported and otherwise CPU. On this macOS 26.3 M4 host PyTorch reports MPS available, so the launcher selects MPS unless `--cpu` is supplied. The main source files also select CUDA, MPS, or CPU directly.

## Original run requirement

On a CUDA-enabled host, use the checked-in entry point from the repository root:

```bash
python AttentionDTA_main.py
```

It defaults to Davis. To select KIBA or Metz, change the active `DATASET` assignment in `AttentionDTA_main.py`.
