"""Lazy compilation and loading of the UE5M3 CUDA extension."""

from __future__ import annotations

import hashlib
import os
import sys
from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=1)
def load_ue5m3_extension():
    """Build the UE5M3 extension on first use and return its pybind module."""
    try:
        import torch
        from torch.utils.cpp_extension import CUDA_HOME, load
    except ImportError as exc:
        raise RuntimeError(
            "xkernels requires a PyTorch installation with CUDA support."
        ) from exc

    if not torch.cuda.is_available():
        raise RuntimeError("xkernels requires an available CUDA device.")
    if CUDA_HOME is None:
        raise RuntimeError("xkernels requires NVCC; CUDA_HOME is not set.")
    if torch.cuda.get_device_capability() < (8, 0):
        raise RuntimeError("xkernels requires an SM80 or newer CUDA device.")

    source = Path(__file__).with_name("csrc") / "ue5m3" / "ue5m3.cu"
    venv_bin = Path(sys.executable).parent
    os.environ["PATH"] = f"{venv_bin}{os.pathsep}{os.environ.get('PATH', '')}"
    digest = hashlib.sha256(
        source.read_bytes() + torch.__version__.encode() + str(torch.version.cuda).encode()
    ).hexdigest()[:12]
    return load(
        name=f"xkernels_marlin_ue5m3_{digest}",
        sources=[str(source)],
        extra_cuda_cflags=["-O3", "-lineinfo"],
        extra_cflags=["-O3"],
        with_cuda=True,
        verbose=os.environ.get("XKERNELS_VERBOSE_BUILD") == "1",
    )
