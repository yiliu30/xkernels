"""Lazy JIT loader for the standalone vLLM Marlin source port."""

from __future__ import annotations

import hashlib
import os
import sys
from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=1)
def load_vllm_marlin_extension():
    """Build the vendored Tensor-Core Marlin extension on first use."""
    try:
        import torch
        from torch.utils.cpp_extension import CUDA_HOME, load
    except ImportError as exc:
        raise RuntimeError("xkernels requires CUDA-enabled PyTorch.") from exc
    if not torch.cuda.is_available() or CUDA_HOME is None:
        raise RuntimeError("the Marlin port requires CUDA and NVCC")
    if torch.cuda.get_device_capability() < (8, 0):
        raise RuntimeError("the current Marlin port requires SM80 or newer")

    source_dir = Path(__file__).with_name("csrc") / "vllm_marlin"
    source = source_dir / "bindings.cu"
    if not source.is_file():
        raise RuntimeError(
            "vendored Marlin CUDA sources are missing from this installation"
        )
    venv_bin = Path(sys.executable).parent
    os.environ["PATH"] = f"{venv_bin}{os.pathsep}{os.environ.get('PATH', '')}"
    digest = hashlib.sha256(
        b"".join(path.read_bytes() for path in sorted(source_dir.rglob("*"))
                 if path.is_file())
        + torch.__version__.encode()
        + str(torch.version.cuda).encode()
    ).hexdigest()[:12]
    return load(
        name=f"xkernels_marlin_vllm_{digest}",
        sources=[str(source)],
        extra_include_paths=[str(source_dir)],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3", "-lineinfo"],
        with_cuda=True,
        verbose=os.environ.get("XKERNELS_VERBOSE_BUILD") == "1",
    )
