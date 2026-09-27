"""CPU-only smoke test for the AttentionDTA source."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from torch.nn import MSELoss
from torch.optim import AdamW
from torch.utils.data import DataLoader

from dataset import CustomDataSet, collate_fn
from model import AttentionDTA


def main():
    dataset_name = "KIBA"
    lines = (ROOT / "datasets" / f"{dataset_name}.txt").read_text().splitlines()
    loader = DataLoader(CustomDataSet(lines[:2]), batch_size=2, shuffle=False, collate_fn=collate_fn)
    drugs, proteins, labels = next(iter(loader))
    model = AttentionDTA().to(torch.device("cpu"))
    predictions = model(drugs, proteins)
    loss = MSELoss()(predictions, labels.view(-1, 1))
    optimizer = AdamW(model.parameters(), lr=5e-5, weight_decay=1e-4)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    assert drugs.shape == (2, 100)
    assert proteins.shape == (2, 1200)
    assert labels.shape == (2,)
    assert predictions.shape == (2, 1)
    print(f"Dataset: {dataset_name}")
    print("Batch size: 2")
    print(f"Drug tensor shape: {tuple(drugs.shape)}")
    print(f"Protein tensor shape: {tuple(proteins.shape)}")
    print(f"Prediction shape: {tuple(predictions.shape)}")
    print(f"Loss: {loss.item():.6f}")
    print("Forward pass: PASS")
    print("Backward pass: PASS")
    print("Optimizer step: PASS")


if __name__ == "__main__":
    main()
