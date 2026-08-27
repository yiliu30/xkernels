"""SM80+ Marlin Tensor-Core execution for raw MXFP4 UE5M3 weights."""

from __future__ import annotations

import math
from dataclasses import dataclass

from ._vllm_marlin_extension import load_vllm_marlin_extension
from .api import _torch, _validate_cuda_tensor


@dataclass(frozen=True)
class MarlinMXFP4UE5M3TensorCoreWeight:
    """MXFP4 UE5M3 data prepared for the Marlin BF16 Tensor-Core path."""

    packed: object
    scales: object
    marlin_weight: object
    marlin_scales: object
    workspace: object
    n: int
    k: int
    padded_n: int
    padded_k: int
    block_size: int


def _round_up(value: int, multiple: int) -> int:
    return (value + multiple - 1) // multiple * multiple


def _padded_nk(n: int, k: int, group_size: int) -> tuple[int, int]:
    candidates = (
        (_round_up(n, 64), _round_up(k, math.lcm(128, group_size))),
        (_round_up(n, 128), _round_up(k, math.lcm(64, group_size))),
    )
    return min(candidates, key=lambda nk: (nk[0] * nk[1], nk[0] + nk[1]))


def _decode_ue5m3(scales):
    torch = _torch()
    exponent = scales >> 3
    mantissa = scales.bitwise_and(0x7)
    value = torch.where(
        exponent.eq(0),
        torch.ldexp(mantissa.float(), torch.full_like(exponent, -17, dtype=torch.int32)),
        torch.ldexp(1.0 + mantissa.float() * 0.125, exponent.to(torch.int32) - 15),
    )
    return torch.where(scales.eq(0xFF), torch.full_like(value, torch.nan), value)


def _permute_scales(scales, group_size: int):
    torch = _torch()
    perm = [i + 8 * j for i in range(8) for j in range(8)]
    if group_size < 0:
        perm = [2 * i + j for i in range(4) for j in (0, 1, 8, 9, 16, 17, 24, 25)]
    return scales.reshape(-1, len(perm))[:, perm].reshape(-1, scales.size(1)).contiguous()


def _permute_bias(bias):
    perm = [2 * i + j for i in range(4) for j in (0, 1, 8, 9, 16, 17, 24, 25)]
    return bias.reshape(-1, 32)[:, perm].reshape_as(bias).contiguous()


def prepare_mxfp4_ue5m3_marlin_weight(weight, scales, block_size: int = 32):
    """Repack raw MXFP4 weights and decode UE5M3 scales once for Marlin."""
    torch = _torch()
    if block_size not in (16, 32):
        raise ValueError("block_size must be 16 or 32")
    _validate_cuda_tensor(weight, "weight", torch.uint8)
    _validate_cuda_tensor(scales, "scales", torch.uint8)
    if torch.cuda.get_device_capability(weight.device) < (8, 0):
        raise RuntimeError("MXFP4 UE5M3 Marlin GEMM requires SM80 or newer")
    if weight.ndim != 2 or scales.ndim != 2:
        raise ValueError("weight and scales must both be rank-2 tensors")
    n, packed_k = weight.shape
    k = packed_k * 2
    if not n or not k or k % block_size or scales.shape != (n, k // block_size):
        raise ValueError("invalid MXFP4 UE5M3 weight or scale shape")
    if weight.device != scales.device:
        raise ValueError("weight and scales must be on the same CUDA device")

    padded_n, padded_k = _padded_nk(n, k, block_size)
    qweight = weight.view(torch.int32).transpose(0, 1).contiguous()
    qweight = torch.nn.functional.pad(qweight, (0, padded_n - n, 0, (padded_k - k) // 8))
    marlin_weight = load_vllm_marlin_extension().gptq_marlin_repack(
        qweight, torch.empty(0, dtype=torch.int32, device=weight.device),
        padded_k, padded_n, 4, False,
    )
    decoded = _decode_ue5m3(scales).transpose(0, 1).to(torch.bfloat16)
    decoded = torch.nn.functional.pad(decoded, (0, padded_n - n, 0, (padded_k - k) // block_size))
    marlin_scales = _permute_scales(decoded, block_size)
    workspace = torch.zeros(
        torch.cuda.get_device_properties(weight.device).multi_processor_count,
        dtype=torch.int32, device=weight.device,
    )
    return MarlinMXFP4UE5M3TensorCoreWeight(
        weight, scales, marlin_weight, marlin_scales, workspace, n, k,
        padded_n, padded_k, block_size,
    )


def mxfp4_ue5m3_marlin_bf16_gemm(activations, prepared, bias=None):
    """Run prepared UE5M3 MXFP4 weights through Marlin BF16 Tensor Cores."""
    torch = _torch()
    _validate_cuda_tensor(activations, "activations", torch.bfloat16)
    if activations.ndim < 2 or activations.shape[-1] != prepared.k:
        raise ValueError("activations must have shape [..., K] matching prepared weights")
    if activations.device != prepared.packed.device:
        raise ValueError("activations and prepared weights must share a device")
    if bias is not None:
        _validate_cuda_tensor(bias, "bias", torch.bfloat16)
        if bias.shape != (prepared.n,) or bias.device != activations.device:
            raise ValueError("bias must have shape [N] on the activation device")
        bias = _permute_bias(torch.nn.functional.pad(bias, (0, prepared.padded_n - prepared.n)))
    flat = activations.reshape(-1, prepared.k)
    padded_a = torch.nn.functional.pad(flat, (0, prepared.padded_k - prepared.k)).contiguous()
    output = load_vllm_marlin_extension().mxfp4_bf16_marlin_gemm(
        padded_a, prepared.marlin_weight, prepared.marlin_scales,
        prepared.workspace.zero_(), bias, padded_a.size(0), prepared.padded_n,
        prepared.padded_k, False, True,
    )
    return output[:, :prepared.n].reshape(*activations.shape[:-1], prepared.n)
