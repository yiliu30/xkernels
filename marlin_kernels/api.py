"""Public MXFP4 W4A16 API."""

from __future__ import annotations

from dataclasses import dataclass

from ._extension import load_extension


@dataclass(frozen=True)
class MarlinMXFP4Weight:
    """Prepared MXFP4 weight data bound to a CUDA device.

    Attributes:
        packed: Raw row-major packed E2M1 values with shape ``[N, K // 2]``.
        scales: Raw E8M0 bit patterns with shape ``[N, K // 32]``.
        n: Logical output dimension.
        k: Logical reduction dimension.
    """

    packed: object
    scales: object
    n: int
    k: int


@dataclass(frozen=True)
class MarlinMXFP4UE5M3Weight:
    """Prepared MXFP4 E2M1 weight with raw UE5M3 block scales."""

    packed: object
    scales: object
    n: int
    k: int
    block_size: int


def _torch():
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("marlin-kernels requires PyTorch.") from exc
    return torch


def _validate_cuda_tensor(tensor, name: str, dtype) -> None:
    torch = _torch()
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if not tensor.is_cuda:
        raise ValueError(f"{name} must be a CUDA tensor")
    if tensor.dtype != dtype:
        raise TypeError(f"{name} must have dtype {dtype}, got {tensor.dtype}")
    if not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous")


def prepare_mxfp4_weight(weight, scales) -> MarlinMXFP4Weight:
    """Validate and prepare raw row-major MXFP4 E2M1/E8M0 weights.

    Args:
        weight: CUDA uint8 tensor of shape ``[N, K // 2]``. Low then high
            nibbles contain consecutive E2M1 values.
        scales: CUDA uint8 tensor of shape ``[N, K // 32]`` containing raw
            unsigned E8M0 scale bit patterns.

    Returns:
        A device-specific prepared weight object reusable across GEMM calls.
    """
    torch = _torch()
    _validate_cuda_tensor(weight, "weight", torch.uint8)
    _validate_cuda_tensor(scales, "scales", torch.uint8)
    if weight.ndim != 2 or scales.ndim != 2:
        raise ValueError("weight and scales must both be rank-2 tensors")
    n, packed_k = weight.shape
    k = packed_k * 2
    if k == 0 or n == 0 or k % 32:
        raise ValueError("weight dimensions must be non-zero and K divisible by 32")
    if scales.shape != (n, k // 32):
        raise ValueError("scales must have shape [N, K // 32]")
    if weight.device != scales.device:
        raise ValueError("weight and scales must be on the same CUDA device")
    return MarlinMXFP4Weight(weight, scales, n, k)


def mxfp4_bf16_gemm(activations, prepared: MarlinMXFP4Weight, bias=None):
    """Compute BF16 activations times raw MXFP4 weights.

    The CUDA kernel dequantizes E2M1 values and their E8M0 scales in registers;
    it never materializes a BF16 weight matrix.
    """
    torch = _torch()
    _validate_cuda_tensor(activations, "activations", torch.bfloat16)
    if activations.ndim < 2:
        raise ValueError("activations must have shape [..., K]")
    if activations.shape[-1] != prepared.k:
        raise ValueError("activations last dimension must equal prepared.k")
    if activations.device != prepared.packed.device:
        raise ValueError("activations and prepared weights must share a device")
    if bias is not None:
        _validate_cuda_tensor(bias, "bias", torch.bfloat16)
        if bias.shape != (prepared.n,) or bias.device != activations.device:
            raise ValueError("bias must have shape [N] on the activation device")
    flat = activations.reshape(-1, prepared.k)
    output = load_extension().mxfp4_bf16_gemm(
        flat, prepared.packed, prepared.scales, bias
    )
    return output.reshape(*activations.shape[:-1], prepared.n)


def prepare_mxfp4_ue5m3_weight(
    weight, scales, block_size: int = 32
) -> MarlinMXFP4UE5M3Weight:
    """Prepare MXFP4-packed E2M1 weights with UE5M3 scales.

    Args:
        weight: CUDA uint8 tensor shaped ``[N, K // 2]``. Its byte packing is
            the same as the existing MXFP4 E8M0 API: low then high E2M1 nibble.
        scales: CUDA uint8 UE5M3 bit patterns shaped ``[N, K // block_size]``.
        block_size: Number of K values sharing each UE5M3 scale: 16 or 32.

    Returns:
        A device-specific prepared weight object for UE5M3 W4A16 GEMM.
    """
    torch = _torch()
    if block_size not in (16, 32):
        raise ValueError("block_size must be 16 or 32")
    _validate_cuda_tensor(weight, "weight", torch.uint8)
    _validate_cuda_tensor(scales, "scales", torch.uint8)
    if weight.ndim != 2 or scales.ndim != 2:
        raise ValueError("weight and scales must both be rank-2 tensors")
    n, packed_k = weight.shape
    k = packed_k * 2
    if k == 0 or n == 0 or k % block_size:
        raise ValueError("weight dimensions must be non-zero and K block-aligned")
    if scales.shape != (n, k // block_size):
        raise ValueError("scales must have shape [N, K // block_size]")
    if weight.device != scales.device:
        raise ValueError("weight and scales must be on the same CUDA device")
    return MarlinMXFP4UE5M3Weight(weight, scales, n, k, block_size)


def mxfp4_ue5m3_bf16_gemm(
    activations, prepared: MarlinMXFP4UE5M3Weight, bias=None
):
    """Compute BF16 activations × MXFP4 E2M1 weights × UE5M3 scales.

    UE5M3 byte ``0xff`` is decoded as NaN and therefore propagates to outputs
    whose reduction spans that scale block.
    """
    torch = _torch()
    _validate_cuda_tensor(activations, "activations", torch.bfloat16)
    if activations.ndim < 2:
        raise ValueError("activations must have shape [..., K]")
    if activations.shape[-1] != prepared.k:
        raise ValueError("activations last dimension must equal prepared.k")
    if activations.device != prepared.packed.device:
        raise ValueError("activations and prepared weights must share a device")
    if bias is not None:
        _validate_cuda_tensor(bias, "bias", torch.bfloat16)
        if bias.shape != (prepared.n,) or bias.device != activations.device:
            raise ValueError("bias must have shape [N] on the activation device")
    flat = activations.reshape(-1, prepared.k)
    output = load_extension().mxfp4_ue5m3_bf16_gemm(
        flat, prepared.packed, prepared.scales, prepared.block_size, bias
    )
    return output.reshape(*activations.shape[:-1], prepared.n)
