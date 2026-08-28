"""Raw UE5M3 scale-format conversion APIs."""

from __future__ import annotations

from ._ue5m3_extension import load_ue5m3_extension
from .api import _torch, _validate_cuda_tensor


def fp32_to_ue5m3(input):
    """Encode CUDA float32 values as raw uint8 UE5M3 bytes without checks.

    UE5M3 is the unsigned `EEEEE MMM` PTX scale format, not a PyTorch FP8
    dtype. This fast path is intended for known-valid non-negative scale
    tensors and is CUDA Graph capture-safe. Positive overflow and infinity
    saturate to `0xfe`; NaN encodes as `0xff`.
    """
    torch = _torch()
    _validate_cuda_tensor(input, "input", torch.float32)
    return load_ue5m3_extension().fp32_to_ue5m3(input)


def fp32_to_ue5m3_checked(input):
    """Encode a validated non-negative CUDA float32 UE5M3 scale tensor.

    Negative values and NaNs are rejected because they are not valid scale
    inputs. This debug/reference API synchronizes with the device to report
    validation failures; use :func:`fp32_to_ue5m3` in production.
    """
    torch = _torch()
    _validate_cuda_tensor(input, "input", torch.float32)
    if torch.any(input < 0).item():
        raise ValueError("input must not contain negative UE5M3 scales")
    if torch.any(torch.isnan(input)).item():
        raise ValueError("input must not contain NaN UE5M3 scales")
    return fp32_to_ue5m3(input)


def ue5m3_to_fp32(input):
    """Decode raw CUDA uint8 UE5M3 bytes to float32 values.

    The reserved byte `0xff` decodes to NaN.
    """
    torch = _torch()
    _validate_cuda_tensor(input, "input", torch.uint8)
    return load_ue5m3_extension().ue5m3_to_fp32(input)
