# Training pipeline

`AttentionDTA_main.py` is the training entry point. It defaults to `DATASET="Davis"` (Metz and KIBA are commented alternatives), seed 4321, five folds, 500 epochs, patience 50, batch 128, AdamW base learning rate `5e-5`, and weight decay `1e-4` on non-bias parameters only. It initializes matrices with Xavier uniform.

For each fold, a CyclicLR goes from `5e-5` to `5e-4` with `step_size_up=train_size//128` and `cycle_momentum=False`; it steps after every optimization step. MSE is the loss. Each epoch evaluates validation MSE/MAE/R². The lowest validation MSE checkpoint is saved as `valid_best_checkpoint.pth`; a final `stable_checkpoint.pth` is also saved. Then that validation-best checkpoint is evaluated on train, validation, and held-out fold test sets, saved to `results/<dataset>/<fold>_Fold/`, and fold test MSE/MAE/R² are meaned with population standard deviation.

The source selects CUDA when available, otherwise MPS when available, otherwise CPU. It sets `CUDA_VISIBLE_DEVICES="0"` and calls `torch.cuda.manual_seed_all`, but does not set CuDNN deterministic mode and does not supply a `Generator` to the main script's `random_split`, so exact repeatability is not guaranteed across releases/devices.
