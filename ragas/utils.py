from __future__ import annotations

import typing as t
from warnings import warn

import torch
from torch import device as Device

DEVICES = ["cpu", "cuda"]


def device_check(device: t.Literal["cpu", "cuda"] | Device) -> torch.device:
    if isinstance(device, Device):
        return device
    if device not in DEVICES:
        raise ValueError(f"Invalid device {device}")
    if device == "cuda" and not torch.cuda.is_available():
        warn("cuda not available, using cpu")
        device = "cpu"

    return torch.device(device)
