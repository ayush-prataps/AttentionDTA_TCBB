# Model architecture

Default batch size is denoted `B`. Padding is embedded as zero, but convolutions and attention are not masked, so padded positions remain in the computation.

| Stage | Drug shape / parameters | Protein shape / parameters |
|---|---|---|
| Input encoding | `(B,100)` integer IDs | `(B,1200)` integer IDs |
| Embedding | `(B,100,128)`, `Embedding(65,128,padding_idx=0)` | `(B,1200,128)`, `Embedding(26,128,padding_idx=0)` |
| Permute | `(B,128,100)` | `(B,128,1200)` |
| Conv1 + ReLU | `(B,32,97)`, kernel 4 | `(B,32,1197)`, kernel 4 |
| Conv2 + ReLU | `(B,64,92)`, kernel 6 | `(B,64,1190)`, kernel 8 |
| Conv3 + ReLU | `(B,96,85)`, kernel 8 | `(B,96,1179)`, kernel 12 |
| Two-sided attention | `(B,96,85)` | `(B,96,1179)` |
| Global max pool | `MaxPool1d(85)` -> `(B,96)` | `MaxPool1d(1179)` -> `(B,96)` |

The concatenated pair is `(B,192)`. It passes through `Linear(192,1024)`, LeakyReLU, Dropout(0.1); `Linear(1024,1024)`, LeakyReLU, Dropout(0.1); `Linear(1024,512)`, LeakyReLU; and `Linear(512,1)`. The output bias is initially set to 5, then the main script Xavier-initializes all tensors with dimension >1 (thus retaining that bias). `self.dropout` is declared but never used.
