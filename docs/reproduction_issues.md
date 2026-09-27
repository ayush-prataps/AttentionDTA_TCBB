# Reproduction issues

## Device availability on this host

Issue: CUDA is unavailable on this host; the installed PyTorch build reports MPS as available.

Cause: this macOS ARM host reports `torch.cuda.is_available() == False`; its Apple-Silicon PyTorch runtime provides MPS instead.

Evidence: the pre-fix source failed at model construction with `AssertionError: Torch not compiled with CUDA enabled`.

Current behavior: source modules select CUDA, MPS, or CPU and move models and tensors with `.to(device)`. The smoke test runs the forward, backward, and optimizer-step path on CPU.

Effect: full training is possible on CPU, although substantially slower than CUDA or MPS.

Effect on reproduction: a CPU/MPS-compatible execution does not establish a CUDA-equivalent paper result.

## Apple M4 MPS backend

Issue: this Apple M4's Metal-capable GPU is selected through MPS rather than CUDA.

Cause: CUDA is unavailable on Apple Silicon; the portable device check selects MPS when `torch.backends.mps.is_available()` is true.

Evidence: local hardware inspection reports Apple M4, 8 GPU cores, and Metal support. The current PyTorch 2.8.0 runtime reports `MPS built: True`, `MPS available: True`, and selects `mps`.

The launcher invokes `AttentionDTA_main.py` via `runpy` after selecting the available device. The source itself now performs portable device selection.

Why safe: it changes device placement only; it leaves data, tensors' shapes, architecture, initialisation, splitting, optimizer, scheduler, and hyperparameters unchanged.

Effect on reproduction: MPS enables Apple-Silicon-compatible execution, but it does not establish a CUDA-equivalent paper result.

## Legacy NumPy alias (resolved)

The old source used a removed NumPy scalar alias, which raises `AttributeError` with NumPy >=1.24.

Cause: `np.float` was removed from NumPy.

The current `dataset.py` uses built-in `float(label)` instead.

The compatibility fix is included in the current source.

Why safe: it preserves the original code path and float conversion exactly.

Effect on reproduction: label conversion works with supported modern NumPy versions.
