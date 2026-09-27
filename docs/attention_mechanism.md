# Attention mechanism

This is not PyTorch `MultiheadAttention` and has no softmax, query/key/value projection, residual path, normalization, or masking. Let `C=96=3*conv`, `Ld=85`, `Lp=1179`, and default heads `H=8`.

1. CNN outputs arrive as drug `D: (B,C,Ld)` and protein `P: (B,C,Lp)`.
2. `D.permute(0,2,1)` is `(B,Ld,C)`. `d_a: Linear(C,C*H)` followed by ReLU produces `(B,Ld,768)`, then `.view(B,H,Ld,C)` produces `D_h` `(B,8,85,96)`. The analogous `p_a` produces `P_h` `(B,8,1179,96)`. The reshape partitions the final projection in contiguous head blocks; it does not transpose an explicit head dimension.
3. `matmul(D_h, P_h.permute(0,1,3,2))` is the per-head drug--protein interaction matrix `(B,8,85,1179)`. It divides by `sqrt(C)=sqrt(96)`, applies `tanh`, then averages heads (`mean(...,1)`) to form `I: (B,85,1179)`.
4. Drug position scores are `tanh(sum(I, dim=2)).unsqueeze(1)`: `(B,1,85)`. Protein scores are `tanh(sum(I, dim=1)).unsqueeze(1)`: `(B,1,1179)`. Thus two-sided attention means each side receives the summed interaction evidence over the opposite sequence, not two cross-attention output mixtures.
5. Broadcasting multiplies the original CNN features: `D * drug_scores` -> `(B,96,85)` and `P * protein_scores` -> `(B,96,1179)`. Max pooling over every remaining position yields one 96-vector from each branch.

The code defines the attention scale on the selected CUDA, MPS, or CPU device and aligns it with the input device during the forward pass. This is a device-placement scalar, not a learned parameter or buffer.
