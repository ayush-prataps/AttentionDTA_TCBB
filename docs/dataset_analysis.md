# Dataset analysis

Each line has exactly five whitespace-delimited fields, with no header:
`Drug_ID Protein_ID Drug_SMILES Amino_acid_sequence affinity`. All field counts are five; affinity is numeric and has no NaN/Inf. SMILES and protein sequences contain no characters outside `dataset.py`'s corresponding vocabularies.

| Dataset | Records | Unique drug IDs | Unique protein IDs | Unique SMILES | Unique sequences | Affinity min--max | SMILES >100 | Proteins >1200 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Davis | 24,956 | 68 | 367 | 68 | 365 | 5.0--10.6197887583 | 0 | 3,128 |
| Metz | 35,259 | 1,423 | 170 | 1,423 | 170 | 4.0--11.1 | 115 | 5,046 |
| KIBA | 118,254 | 2,111 | 229 | 2,068 | 229 | 0.0--17.200179498 | 982 | 15,247 |

The original loader does not use IDs at all; it reads the final three fields. Each dataset has a SMILES string and an uppercase amino-acid sequence per record. The loader has no missing-value handling or affinity normalization; its only implicit transformation is float32 casting in `collate_fn`.

File SHA-256: Davis `728311b0b8a796bb1881646209f175bcdeed481d5f4fbb8c26727b362829579d`; Metz `0a3633e7e5582e710105724f61f4a991d4e872b3fa6e2ef315b01110afb85c86`; KIBA `3f1bc13b46e0ee238b0ef6de319a3310dedc026e67fb3b673bf93f33998297cb`.
