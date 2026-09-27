# Environment

Repository README requests Python 3.6, PyTorch >=1.2, NumPy, sklearn, tqdm, tensorboardX, and prefetch_generator; no lockfile or precise version constraints are supplied.

This host is macOS 26.3 ARM64. System `python3` is 3.14.2 with no project packages; the only other discovered interpreter is Python 3.11.14. No conda executable was found. CUDA is unavailable (no GPU), while the installed PyTorch runtime reports MPS available.

Created only a repository-local `.venv` using Python 3.11.14. Installed pinned compatibility set: torch 2.8.0, numpy 1.23.5, scikit-learn 1.3.2, tqdm 4.67.1, tensorboardX 2.6.4, prefetch-generator 1.0.3. NumPy 1.23.5 is retained for environment reproducibility; `dataset.py` now uses built-in `float` for label conversion. PyTorch 1.2 cannot be used on this Python/ARM platform; 2.8.0 is a compatible Apple-Silicon wheel, not an architectural change.

Hardware inspection identifies Apple M4 with an 8-core Metal-capable integrated GPU. The installed PyTorch 2.8.0 wheel reports MPS built and available, and the compatibility launcher selects MPS on this host when not forced to CPU.
