#!/usr/bin/env python3
"""Run EfficientVit train_seg.py with a PyTorch AMP compatibility shim."""

from __future__ import annotations

import os
import runpy
import types
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EFFICIENTVIT_ROOT = ROOT / "external/EfficientVit"


def efficientvit_train_seg() -> Path:
    root = Path(os.environ.get("EFFICIENTVIT_ROOT", str(DEFAULT_EFFICIENTVIT_ROOT))).expanduser().resolve()
    train_seg = root / "scripts/train_seg.py"
    if not train_seg.exists():
        raise FileNotFoundError(f"Missing EfficientVit train script: {train_seg}")
    return train_seg


def install_grad_scaler_compat() -> None:
    if not hasattr(torch, "amp"):
        torch.amp = types.SimpleNamespace()

    native_grad_scaler = getattr(torch.amp, "GradScaler", None)
    cuda_grad_scaler = torch.cuda.amp.GradScaler

    def grad_scaler_compat(*args, **kwargs):
        if native_grad_scaler is not None:
            try:
                return native_grad_scaler(*args, **kwargs)
            except TypeError:
                pass

        cleaned_args = args
        cleaned_kwargs = dict(kwargs)
        if cleaned_args and isinstance(cleaned_args[0], str):
            cleaned_args = cleaned_args[1:]
        cleaned_kwargs.pop("device_type", None)
        return cuda_grad_scaler(*cleaned_args, **cleaned_kwargs)

    torch.amp.GradScaler = grad_scaler_compat


def main() -> None:
    install_grad_scaler_compat()
    runpy.run_path(str(efficientvit_train_seg()), run_name="__main__")


if __name__ == "__main__":
    main()
