"""MXFP4 UE5M3 preparation and Marlin Tensor-Core parity tests."""

import pytest

torch = pytest.importorskip("torch")

from marlin_kernels import (
    mxfp4_ue5m3_bf16_gemm,
    prepare_mxfp4_ue5m3_weight,
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
