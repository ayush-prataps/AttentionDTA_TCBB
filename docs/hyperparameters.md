# Hyperparameter scripts

`Learning_rate_select.py` is a KIBA-only 80/20 random split experiment. It starts at LR `1e-10`, uses batch 128, AdamW with decay `1e-4`, 10 epochs, and `StepLR(step_size=1,gamma=10)`, printing a geometric LR range. It does not calculate validation metrics or select/report a final LR beyond that trace.

`Hyperparameter_research.py` is also KIBA-only and expensive: each candidate uses a freshly random 60/20/20 split, up to 500 epochs with patience 50, and a 10x CyclicLR. It searches sequentially: heads `[12,10,8,6,4,2]`, batch sizes `[512,256,128,64,32,16]`, then dropout `[0.1,...,0.9]`. Defaults feed later stages (`best_head_num=4`, `best_batch_size=128`, `best_dropout_rate=0.1`). Its code has a methodological defect: it reassigns `dataset = CustomDataSet(dataset)` inside the candidate loops, so the second iteration attempts to wrap a Dataset rather than the original line list; it is not needed for main reproduction and was not run.

The main script hard-codes the final-looking architecture defaults from `model.py`: 8 heads, batch 128, dropout 0.1. This does not match the hyperparameter script's initial head default of 4, and the search script does not persist selected values into main.
