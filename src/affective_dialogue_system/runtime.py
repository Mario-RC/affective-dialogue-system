"""Runtime helpers that keep optional ML dependencies lazy."""

from __future__ import annotations


def resolve_device(device: str | None = None) -> str:
    """Return a torch device string without importing torch at package import."""

    if device:
        return device

    import torch

    return "cuda:0" if torch.cuda.is_available() else "cpu"


def resolve_torch_dtype(device: str, torch_dtype: str | None = None):
    """Return a torch dtype object for a device string."""

    import torch

    if torch_dtype:
        return getattr(torch, torch_dtype) if isinstance(torch_dtype, str) else torch_dtype
    return torch.float16 if device.startswith("cuda") else torch.float32


def pipeline_device_index(device: str) -> int:
    """Return the device value expected by transformers pipelines."""

    if device.startswith("cuda"):
        if ":" in device:
            return int(device.split(":", 1)[1])
        return 0
    return -1

