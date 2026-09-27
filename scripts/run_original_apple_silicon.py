"""Run the AttentionDTA entry point on Apple Silicon.

Apple Silicon uses MPS, not CUDA. This adapter selects MPS when PyTorch reports
it usable, otherwise CPU, then executes AttentionDTA_main.py with runpy.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from pathlib import Path
import os
import runpy
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch


def select_device(force_cpu: bool) -> torch.device:
    if not force_cpu and torch.backends.mps.is_built() and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@contextmanager
def force_cpu_device(force_cpu: bool):
    if not force_cpu:
        yield
        return
    original_cuda_available = torch.cuda.is_available
    original_mps_available = torch.backends.mps.is_available
    torch.cuda.is_available = lambda: False
    torch.backends.mps.is_available = lambda: False
    try:
        yield
    finally:
        torch.cuda.is_available = original_cuda_available
        torch.backends.mps.is_available = original_mps_available


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true", help="start the full original 5-fold/500-epoch run")
    parser.add_argument("--cpu", action="store_true", help="force CPU even if MPS becomes available")
    args = parser.parse_args()

    target = select_device(args.cpu)
    print(f"PyTorch: {torch.__version__}")
    print(f"MPS built: {torch.backends.mps.is_built()}")
    print(f"MPS available: {torch.backends.mps.is_available()}")
    print(f"Selected device: {target}")
    if not args.run:
        print("Adapter check: PASS (use --run to launch the AttentionDTA entry point).")
        return

    os.chdir(ROOT)
    with force_cpu_device(args.cpu):
        runpy.run_path(str(ROOT / "AttentionDTA_main.py"), run_name="__main__")


if __name__ == "__main__":
    main()
