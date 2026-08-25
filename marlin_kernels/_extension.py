"""Lazy compilation and loading of the CUDA extension."""

from __future__ import annotations

import hashlib
import os
import sys
from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=1)
def load_extension():
    """Build the extension on first use and return its pybind module."""
    try:
        import torch
        from torch.utils.cpp_extension import CUDA_HOME, load
    except ImportError as exc:
        raise RuntimeError(
            "marlin-kernels requires a PyTorch installation with CUDA support."
        ) from exc

    if not torch.cuda.is_available():
        raise RuntimeError("marlin-kernels requires an available CUDA device.")
    if CUDA_HOME is None:
        raise RuntimeError("marlin-kernels requires NVCC; CUDA_HOME is not set.")
    if torch.cuda.get_device_capability() < (8, 0):
        raise RuntimeError("marlin-kernels requires an SM80 or newer CUDA device.")

    source = Path(__file__).with_name("csrc") / "mxfp4_bf16.cu"
    # ``cpp_extension`` invokes ``ninja`` from PATH. A caller using the venv's
    # interpreter directly (rather than activating it) does not get its bin
    # directory on PATH, despite ninja being installed in that environment.
    venv_bin = Path(sys.executable).parent
    os.environ["PATH"] = f"{venv_bin}{os.pathsep}{os.environ.get('PATH', '')}"
    digest = hashlib.sha256(
        (source.read_bytes() + torch.__version__.encode() + str(torch.version.cuda).encode())
    ).hexdigest()[:12]
    name = f"marlin_mxfp4_{digest}"
    extra_cuda_cflags = ["-O3", "--use_fast_math", "-lineinfo"]
    arch_list = os.environ.get("TORCH_CUDA_ARCH_LIST")
    if arch_list:
        # cpp_extension consumes TORCH_CUDA_ARCH_LIST itself.
        os.environ["TORCH_CUDA_ARCH_LIST"] = arch_list
    return load(
        name=name,
        sources=[str(source)],
        extra_cuda_cflags=extra_cuda_cflags,
        extra_cflags=["-O3"],
        with_cuda=True,
        verbose=os.environ.get("MARLIN_KERNELS_VERBOSE_BUILD") == "1",
    )
