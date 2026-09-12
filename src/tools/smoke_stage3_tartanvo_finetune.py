from __future__ import annotations

import torch
from torch import nn

from src.config import DEVICE
from src.models import Stage3TartanVOGRU


def main() -> None:
    model = Stage3TartanVOGRU(load_pretrained=False).to(DEVICE)
    model.tartanvo_mode = "finetune"
    model.apply_tartanvo_unfreeze("full")
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
    assert not frozen, "full mode left frozen TartanVO parameters"
    checks = {
        "early": "tartanvo.vonet.flowNet.conv1a.0.weight",
        "middle": "tartanvo.vonet.flowPoseNet.layer3.0.conv1.0.weight",
        "layer5": "tartanvo.vonet.flowPoseNet.layer5.0.conv1.0.weight",
        "gru": "gru.weight_ih_l0",
        "head": "accel.weight",
    }
    params = dict(model.named_parameters())
    for label, name in checks.items():
        assert params[name].grad is not None, f"{label} parameter has no gradient: {name}"
    print("forward pass: ok")
    print("loss:", float(loss.detach().cpu()))
    print("TartanVO mode: finetune")
    print("TartanVO unfreeze: full")
    print("Trainable TartanVO parameters:", sum(p.numel() for _, p in unfrozen))
    print("Frozen TartanVO parameters:", sum(p.numel() for _, p in frozen))
    for label, name in checks.items():
        print(f"{label} grad: {name}")
    print("full finetune smoke passed")


if __name__ == "__main__":
    main()