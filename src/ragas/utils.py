from __future__ import annotations

import logging
import os
import typing as t
from functools import lru_cache
from warnings import warn

import torch
from torch import device as Device

DEVICES = ["cpu", "cuda"]
DEBUG_ENV_VAR = "RAGAS_DEBUG"


def device_check(device: t.Literal["cpu", "cuda"] | Device) -> torch.device:
    if isinstance(device, Device):
        return device
    if device not in DEVICES:
        raise ValueError(f"Invalid device {device}")
    if device == "cuda" and not torch.cuda.is_available():
        warn("cuda not available, using cpu")
        device = "cpu"

    return torch.device(device)


@lru_cache(maxsize=1)
def get_debug_mode() -> bool:
    if os.environ.get(DEBUG_ENV_VAR, str(False)).lower() == "true":
        logging.basicConfig(level=logging.DEBUG)
        return True
    else:
        return False
