# Preprocessing

`dataset.py` defines fixed dictionaries rather than learning a vocabulary: SMILES tokens map to 1--64 (`0` is padding; embedding has 65 rows); protein tokens map to 1--25 (`0` padding; embedding has 26 rows). It iterates over individual characters, not multi-character SMILES atoms.

`label_smiles(line, ..., MAX_SMI_LEN=100)` returns an `int64` zero vector of shape `(100,)`, writing at most its first 100 characters. `label_sequence(..., MAX_SEQ_LEN=1200)` analogously returns `(1200,)`. Consequently long strings are truncated and shorter strings are right-padded with zero.

`collate_fn` splits each raw line with `strip().split()`, takes `pair[-3:]` (SMILES, protein, affinity), and constructs a batch: compound `LongTensor (B,100)`, protein `LongTensor (B,1200)`, label `FloatTensor (B,)`. Labels use built-in `float` conversion, which avoids the removed NumPy scalar alias.

The main script reads all lines, calls `np.random.seed(4321); np.random.shuffle`, then makes contiguous five-fold test partitions. For each fold it wraps the remaining 80% in `CustomDataSet` and applies `torch.utils.data.random_split` into 80% train / 20% validation of that remainder: 64%/16%/20% of all records, subject to integer rounding. DataLoaders have batch size 128, `num_workers=2`; train shuffles, validation and test do not. No affinity scaling or transformation occurs.
