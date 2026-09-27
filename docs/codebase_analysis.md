# Codebase analysis

Inspected checkout: official remote `https://github.com/zhaoqichang/AttentionDTA_TCBB`, commit `803e2799a070b8dcebe32aef910e64ca6b32c299`.

| Path | Purpose |
|---|---|
| `AttentionDTA_main.py` | Actual five-fold training, validation, checkpointing, and evaluation entry point. Defaults to Davis. |
| `dataset.py` | Character vocabularies, fixed-length encoding, `Dataset`, and batch collation. |
| `model.py` | `AttentionDTA` and its custom `mutil_head_attention` (author spelling). |
| `Hyperparameter_research.py` | Sequential KIBA searches over attention-head count, batch size, and dropout. |
| `Learning_rate_select.py` | KIBA learning-rate range run (10 epochs, factor-10 StepLR). |
| `datasets/{Davis,Metz,KIBA}.txt` | The repository-provided, whitespace-delimited five-field data. |
| `README.md` | Minimal dependency and run information. |

There is no requirements file, `setup.py`, `pyproject.toml`, test suite, or `utils.py`. The commented `utils` import is not needed by the checked-in scripts.

## README discrepancy

The README's training command is `python AttentionDTA_main.py`, which is the source's actual training entry point.
