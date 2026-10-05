"""Run a short AttentionDTA experiment on unseen Davis drugs."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch.utils.data import DataLoader, random_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dataset import CustomDataSet, collate_fn
from model import AttentionDTA
from drug_disjoint_davis import DATASET_PATH, load_rows, make_split


device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))


def evaluate(model: nn.Module, loader: DataLoader) -> tuple[float, float, float]:
    model.eval()
    predictions = []
    labels = []
    with torch.no_grad():
        for compounds, proteins, batch_labels in loader:
            outputs = model(compounds.to(device), proteins.to(device)).cpu().flatten()
            predictions.append(outputs)
            labels.append(batch_labels)
    predicted = torch.cat(predictions).numpy()
    observed = torch.cat(labels).numpy()
    return (
        mean_squared_error(observed, predicted),
        mean_absolute_error(observed, predicted),
        r2_score(observed, predicted),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=4321)
    parser.add_argument(
        "--output-dir",
        "--output",
        dest="output_dir",
        type=Path,
        default=ROOT / "results" / "Davis" / "drug_disjoint_partial",
        help="Directory for results and resumable checkpoints (for example, a Google Drive path).",
    )
    args = parser.parse_args()

    if args.epochs < 1:
        raise ValueError("--epochs must be at least 1")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    rows = load_rows(DATASET_PATH)
    train_rows, test_rows, train_drugs, test_drugs = make_split(rows, test_fraction=0.2, seed=args.seed)
    train_dataset, valid_dataset = random_split(
        CustomDataSet(train_rows),
        [int(len(train_rows) * 0.8), len(train_rows) - int(len(train_rows) * 0.8)],
        generator=torch.Generator().manual_seed(args.seed),
    )
    test_dataset = CustomDataSet(test_rows)
    loader_kwargs = {"batch_size": 128, "num_workers": 0, "collate_fn": collate_fn}
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    valid_loader = DataLoader(valid_dataset, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_kwargs)

    model = AttentionDTA().to(device)
    for parameter in model.parameters():
        if parameter.dim() > 1:
            nn.init.xavier_uniform_(parameter)
    weights = [parameter for name, parameter in model.named_parameters() if "bias" not in name]
    biases = [parameter for name, parameter in model.named_parameters() if "bias" in name]
    optimizer = optim.AdamW(
        [{"params": weights, "weight_decay": 1e-4}, {"params": biases, "weight_decay": 0}],
        lr=5e-5,
    )
    scheduler = optim.lr_scheduler.CyclicLR(
        optimizer,
        base_lr=5e-5,
        max_lr=5e-4,
        cycle_momentum=False,
        step_size_up=max(1, len(train_loader)),
    )
    loss_function = nn.MSELoss()
    best_validation_mse = float("inf")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.output_dir / "valid_best_checkpoint.pth"
    resume_checkpoint_path = args.output_dir / "resume_checkpoint.pth"
    start_epoch = 1
    if resume_checkpoint_path.exists():
        resume_checkpoint = torch.load(resume_checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(resume_checkpoint["model_state_dict"])
        optimizer.load_state_dict(resume_checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(resume_checkpoint["scheduler_state_dict"])
        best_validation_mse = resume_checkpoint["best_validation_mse"]
        start_epoch = resume_checkpoint["epoch"] + 1
        np.random.set_state(resume_checkpoint["numpy_rng_state"])
        torch.set_rng_state(resume_checkpoint["torch_rng_state"])
        if torch.cuda.is_available() and resume_checkpoint["cuda_rng_state_all"] is not None:
            torch.cuda.set_rng_state_all(resume_checkpoint["cuda_rng_state_all"])
        print(f"Resuming from epoch {start_epoch}.")

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        train_losses = []
        for compounds, proteins, labels in train_loader:
            compounds = compounds.to(device)
            proteins = proteins.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            loss = loss_function(model(compounds, proteins), labels.view(-1, 1))
            loss.backward()
            optimizer.step()
            scheduler.step()
            train_losses.append(loss.item())
        validation_mse, validation_mae, validation_r2 = evaluate(model, valid_loader)
        print(
            f"epoch={epoch} train_loss={np.mean(train_losses):.8f} "
            f"valid_mse={validation_mse:.8f} valid_mae={validation_mae:.8f} "
            f"valid_r2={validation_r2:.8f}"
        )
        if validation_mse < best_validation_mse:
            best_validation_mse = validation_mse
            torch.save(model.state_dict(), checkpoint_path)
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_validation_mse": best_validation_mse,
                "numpy_rng_state": np.random.get_state(),
                "torch_rng_state": torch.get_rng_state(),
                "cuda_rng_state_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            },
            resume_checkpoint_path,
        )

    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    test_mse, test_mae, test_r2 = evaluate(model, test_loader)
    if train_drugs & test_drugs:
        raise AssertionError("Drug overlap detected after training")
    print(f"device={device}")
    print(f"train_drugs={len(train_drugs)} test_drugs={len(test_drugs)}")
    print(f"test_mse={test_mse:.8f} test_mae={test_mae:.8f} test_r2={test_r2:.8f}")
    print(f"checkpoint={checkpoint_path}")
    print(f"resume_checkpoint={resume_checkpoint_path}")


if __name__ == "__main__":
    main()
