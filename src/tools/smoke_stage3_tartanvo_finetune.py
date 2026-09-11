from __future__ import annotations

import torch
from torch import nn

from src.config import DEVICE
from src.models import Stage3TartanVOGRU


def main() -> None:
    model = Stage3TartanVOGRU(load_pretrained=False).to(DEVICE)
    model.tartanvo_mode = "finetune"
    model.apply_tartanvo_unfreeze("last_block")
    model.train()
    video = torch.rand(1, 3, 2, 224, 224, device=DEVICE)
    target_accel = torch.tensor([0], device=DEVICE)
    target_steer = torch.tensor([1], device=DEVICE)
    accel, steer = model(video)
    loss = nn.functional.cross_entropy(accel, target_accel) + nn.functional.cross_entropy(steer, target_steer)
    loss.backward()
    unfrozen = [(n, p) for n, p in model.named_parameters() if n.startswith("tartanvo.") and p.requires_grad]
    frozen = [(n, p) for n, p in model.named_parameters() if n.startswith("tartanvo.") and not p.requires_grad]
    assert unfrozen, "no unfreezed TartanVO parameters"
    assert any(p.grad is not None for _, p in unfrozen), "unfreezed TartanVO parameters have no gradients"
    assert all(p.grad is None for _, p in frozen), "frozen TartanVO parameter received gradient"
    print("forward pass: ok")
    print("loss:", float(loss.detach().cpu()))
    print("unfreezed grad example:", next(n for n, p in unfrozen if p.grad is not None))
    print("frozen params checked:", len(frozen))
    print("TartanVO last block path: tartanvo.vonet.flowPoseNet.layer5")
    print("finetune smoke passed")


if __name__ == "__main__":
    main()