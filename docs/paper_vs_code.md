# Paper versus code

The cited journal article is Zhao et al., *IEEE/ACM TCBB* 20(2), 852--863 (2023), DOI `10.1109/TCBB.2022.3170365`. The accessible bibliographic abstract confirms the stated two 1D-CNN branches, two-sided multi-head attention, and Davis/Metz/KIBA evaluation, but its full methods/tables were not included in this checkout and could not be verified from the abstract alone. Values are intentionally not invented.

| Paper specification | Code implementation | File/function | Match? | Notes |
|---|---|---|---|---|
| Datasets Davis, Metz, KIBA | All three local files, selectable manually | main lines 95--101 | Partial | Main defaults only to Davis. |
| Separate 1D CNNs for SMILES/protein | Three valid Conv1d layers per branch | `model.AttentionDTA` | Match at high level | Exact paper layer table needs full text verification. |
| Two-sided multi-head attention | Project to 8×96, scaled/tanh interaction map, head mean, sidewise summed/tanh gates | `mutil_head_attention.forward` | Match at high level | It is custom gating, not Transformer attention. |
| Embedding dimension | 128 | `AttentionDTA.__init__` | Pending | Need paper methods table. |
| CNN kernels / filters | Drug 4/6/8; protein 4/8/12; 32/64/96 | `AttentionDTA.__init__` | Pending | Code fact established. |
| Pooling | Global max over 85/1179 positions | `AttentionDTA.forward` | Pending | Code fact established. |
| Fully connected stack | 192→1024→1024→512→1 | `AttentionDTA.__init__` | Pending | Code fact established. |
| Dropout | 0.1 after first two FC layers | `AttentionDTA.forward` | Pending | Declared `self.dropout` is unused. |
| Optimizer / LR / loss | AdamW, cyclic 5e-5→5e-4, MSE | main lines 150--155 | Pending | Full paper verification needed. |
| Batch / epochs | 128 / max 500, patience 50 | main lines 108--112 | Pending | Full paper verification needed. |
| Split / metrics | shuffled 5-fold; 64/16/20 train/valid/test per fold; MSE/MAE/R² | main | Partial | Need paper protocol confirmation. |

Status: PARTIAL. The checked-in implementation is fully traced; numerical and exact paper-method comparison must await the complete paper and a completed CUDA run.
