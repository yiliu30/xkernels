"""MXFP4 UE5M3 preparation and Marlin Tensor-Core parity tests."""

import pytest

torch = pytest.importorskip("torch")

from marlin_kernels import (
    fp32_to_ue5m3,
    mxfp4_ue5m3_bf16_gemm,
    prepare_mxfp4_ue5m3_weight,
    ue5m3_to_fp32,
)


def _decode_e2m1(weight):
    nibbles = torch.stack((weight.bitwise_and(0xF), weight >> 4), -1).reshape(
        weight.size(0), -1
    )
    sign, exponent, mantissa = nibbles >> 3, (nibbles >> 1) & 3, nibbles & 1
    value = torch.where(
        exponent.eq(0), mantissa.float() * 0.5,
        torch.ldexp(1.0 + mantissa.float() * 0.5, exponent.int() - 1),
    )
    return torch.where(sign.bool(), -value, value)


def _decode_ue5m3(scales):
    exponent, mantissa = scales >> 3, scales & 7
    value = torch.where(
        exponent.eq(0), torch.ldexp(mantissa.float(), torch.full_like(exponent, -17, dtype=torch.int32)),
        torch.ldexp(1.0 + mantissa.float() * 0.125, exponent.int() - 15),
    )
    return torch.where(scales.eq(0xFF), torch.full_like(value, torch.nan), value)


def _quantize_e2m1(weight, block_size):
    """Quantize an [N, K] BF16 weight matrix into raw MXFP4 tensors."""
    n, k = weight.shape
    blocks = weight.float().reshape(n, k // block_size, block_size)
    scales = fp32_to_ue5m3(blocks.abs().amax(dim=-1) / 6)
    decoded_scales = ue5m3_to_fp32(scales).reshape(n, k // block_size, 1)
    normalized = blocks / decoded_scales
    magnitudes = normalized.abs().unsqueeze(-1)
    e2m1_values = torch.tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], device=weight.device
    )
    codes = (magnitudes - e2m1_values).abs().argmin(dim=-1).to(torch.uint8)
    codes |= normalized.signbit().to(torch.uint8) << 3
    codes = codes.reshape(n, k)
    packed = (codes[:, 0::2] | (codes[:, 1::2] << 4)).contiguous()
    return packed, scales


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("block_size", [16, 32])
def test_ue5m3_marlin_matches_reference(block_size):
    torch.manual_seed(101 + block_size)
    m, n, k = 8, 64, 128
    weight = torch.randint(0, 256, (n, k // 2), dtype=torch.uint8, device="cuda")
    scales = torch.randint(0, 0xFF, (n, k // block_size), dtype=torch.uint8, device="cuda")
    activations = torch.randn(m, k, dtype=torch.bfloat16, device="cuda")
    bias = torch.randn(n, dtype=torch.bfloat16, device="cuda")

    prepared = prepare_mxfp4_ue5m3_weight(weight, scales, block_size)
    actual = mxfp4_ue5m3_bf16_gemm(activations, prepared, bias)
    decoded = _decode_e2m1(weight) * _decode_ue5m3(scales).repeat_interleave(block_size, 1)
    expected = (activations.float() @ decoded.float().T + bias.float()).bfloat16()
    torch.testing.assert_close(actual, expected, rtol=3e-2, atol=4.0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_ue5m3_marlin_nan_scale_propagates():
    weight = torch.full((64, 64), 0x11, dtype=torch.uint8, device="cuda")
    scales = torch.full((64, 4), 0x78, dtype=torch.uint8, device="cuda")
    scales[:, 0] = 0xFF
    prepared = prepare_mxfp4_ue5m3_weight(weight, scales, 32)
    output = mxfp4_ue5m3_bf16_gemm(
        torch.ones((8, 128), dtype=torch.bfloat16, device="cuda"), prepared
    )
    assert torch.isnan(output).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("block_size", [16, 32])
@pytest.mark.parametrize(
    "m,n,k",
    [
        (1, 64, 128),
        (13, 96, 160),
        (64, 128, 256),
        (129, 192, 320),
    ],
)
def test_bf16_linear_quantized_to_ue5m3_mxfp4_runs_marlin(
    block_size, m, n, k
):
    """Run a BF16 Linear's MXFP4/UE5M3-packed weights through Marlin."""
    torch.manual_seed(211 + block_size + m + n + k)
    reference = torch.nn.Linear(
        k, n, bias=True, device="cuda", dtype=torch.bfloat16
    )
    activations = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)

    packed, scales = _quantize_e2m1(reference.weight.detach(), block_size)
    decoded_weight = _decode_e2m1(packed) * _decode_ue5m3(scales).repeat_interleave(
        block_size, dim=1
    )
    reference.weight = torch.nn.Parameter(decoded_weight.bfloat16())

    prepared = prepare_mxfp4_ue5m3_weight(packed, scales, block_size)
    actual = mxfp4_ue5m3_bf16_gemm(activations, prepared, reference.bias)
    expected = reference(activations)
    torch.testing.assert_close(actual, expected, rtol=3e-2, atol=4.0)
