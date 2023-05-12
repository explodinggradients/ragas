import torch
import typing as t
from warnings import warn

DEVICES = ["cpu", "cuda"]


def device_check(device: t.Literal[DEVICES]):
    if device == "cuda":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            warn("cuda not available, using cpu")
    elif device == "cpu":
        device = torch.device("cpu")
    else:
        raise ValueError(f"Invalid device {device}")

    return device
