# BSD 3-Clause attribution:
# Adapted from castacks/tartanvo Network/PWC/correlation.py.
# Original used CuPy CUDA kernels; this port keeps the 9x9 displacement
# correlation semantics in pure PyTorch for torch 2.8 / CUDA 12.x.

import torch
import torch.nn.functional as F


def FunctionCorrelation(tenFirst: torch.Tensor, tenSecond: torch.Tensor) -> torch.Tensor:
    if tenFirst.shape != tenSecond.shape:
        raise ValueError(f"correlation input shape mismatch: {tenFirst.shape} vs {tenSecond.shape}")
    _, channels, height, width = tenFirst.shape
    padded = F.pad(tenSecond, (4, 4, 4, 4))
    corr = []
    for dy in range(-4, 5):
        for dx in range(-4, 5):
            shifted = padded[:, :, 4 + dy : 4 + dy + height, 4 + dx : 4 + dx + width]
            corr.append((tenFirst * shifted).sum(dim=1, keepdim=True) / channels)
    return torch.cat(corr, dim=1)