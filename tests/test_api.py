import pytest

pytest.importorskip("torch")

import torch

from xkernels import (
    mxfp4_bf16_gemm,
    mxfp4_ue5m3_bf16_gemm,
    prepare_mxfp4_ue5m3_weight,
    prepare_mxfp4_weight,
)


def test_prepare_rejects_cpu_tensors():
    with pytest.raises(ValueError, match="CUDA"):
        prepare_mxfp4_weight(torch.zeros((16, 16), dtype=torch.uint8),
                             torch.zeros((16, 1), dtype=torch.uint8))


def test_prepare_rejects_invalid_scale_shape(monkeypatch):
    class FakeTensor:
        pass

    # Validation deliberately checks tensor type before accessing CUDA fields.
    with pytest.raises(TypeError, match="torch.Tensor"):
        prepare_mxfp4_weight(FakeTensor(), FakeTensor())


def test_prepare_ue5m3_rejects_invalid_block_size():
    with pytest.raises(ValueError, match="16 or 32"):
        prepare_mxfp4_ue5m3_weight(None, None, block_size=8)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_mxfp4_gemm_zero_weight():
    weight = torch.zeros((16, 16), dtype=torch.uint8, device="cuda")
    scales = torch.full((16, 1), 127, dtype=torch.uint8, device="cuda")
    prepared = prepare_mxfp4_weight(weight, scales)
    x = torch.randn((2, 32), dtype=torch.bfloat16, device="cuda")
    expected = torch.zeros((2, 16), dtype=torch.bfloat16, device="cuda")
    assert torch.equal(mxfp4_bf16_gemm(x, prepared), expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_mxfp4_gemm_matches_reference_for_subnormal_value():
    # Low nibble 0b0001 is the E2M1 subnormal +0.5; E8M0 127 means ×1.
    weight = torch.full((16, 16), 0x11, dtype=torch.uint8, device="cuda")
    scales = torch.full((16, 1), 127, dtype=torch.uint8, device="cuda")
    prepared = prepare_mxfp4_weight(weight, scales)
    x = torch.ones((1, 32), dtype=torch.bfloat16, device="cuda")
    expected = torch.full((1, 16), 16.0, dtype=torch.bfloat16, device="cuda")
    assert torch.equal(mxfp4_bf16_gemm(x, prepared), expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("m", [1, 17, 129])
def test_mxfp4_gemm_matches_vllm_marlin(m):
    """Compare raw MXFP4 input directly with vLLM's Marlin W4A16 path."""
    from vllm import _custom_ops as ops
    from vllm.model_executor.layers.quantization.utils.marlin_utils import (
        marlin_make_workspace_new,
        marlin_permute_scales,
    )
    from vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 import (
        apply_fp4_marlin_linear,
        mxfp4_marlin_process_scales,
    )

    torch.manual_seed(17 + m)
    n, k = 128, 256
    raw_weight = torch.randint(0, 256, (n, k // 2), dtype=torch.uint8, device="cuda")
    raw_scales = torch.randint(116, 130, (n, k // 32), dtype=torch.uint8, device="cuda")
    activations = torch.randn((m, k), dtype=torch.bfloat16, device="cuda")
    bias = torch.randn((n,), dtype=torch.bfloat16, device="cuda")

    ours = mxfp4_bf16_gemm(
        activations, prepare_mxfp4_weight(raw_weight, raw_scales), bias
    )

    # This is vLLM's `prepare_fp4_layer_for_marlin` data path expressed from
    # the public helpers, avoiding a model/layer fixture.
    qweight = raw_weight.view(torch.int32).transpose(0, 1).contiguous()
    marlin_weight = ops.gptq_marlin_repack(
        b_q_weight=qweight,
        perm=torch.empty(0, dtype=torch.int, device="cuda"),
        size_k=k,
        size_n=n,
        num_bits=4,
        is_a_8bit=False,
    )
    vllm_scales = raw_scales.transpose(0, 1).contiguous().view(torch.float8_e8m0fnu)
    vllm_scales = marlin_permute_scales(vllm_scales.to(torch.bfloat16), k, n, 32)
    vllm_scales = mxfp4_marlin_process_scales(vllm_scales).contiguous()
    reference = apply_fp4_marlin_linear(
        input=activations,
        weight=marlin_weight,
        weight_scale=vllm_scales,
        weight_global_scale=None,
        workspace=marlin_make_workspace_new(torch.device("cuda")),
        size_n=n,
        size_k=k,
        bias=bias,
    )

    # Both paths accumulate FP32 products but convert the final result to BF16.
    # This keeps the comparison strict enough to catch decoding/layout errors
    # while allowing the different reduction order of Marlin's Tensor Cores.
    torch.testing.assert_close(ours, reference, rtol=3e-2, atol=4.0)


def _decode_e2m1_reference(weight: torch.Tensor) -> torch.Tensor:
    n, packed_k = weight.shape
    nibbles = torch.stack((weight.bitwise_and(0xF), weight >> 4), dim=-1).reshape(n, -1)
    sign = nibbles >> 3
    exponent = (nibbles >> 1).bitwise_and(0x3)
    mantissa = nibbles.bitwise_and(1)
    value = torch.where(
        exponent.eq(0), mantissa.float() * 0.5,
        torch.ldexp(1.0 + mantissa.float() * 0.5, exponent.to(torch.int32) - 1),
    )
    return torch.where(sign.bool(), -value, value)


def _decode_ue5m3_reference(scales: torch.Tensor) -> torch.Tensor:
    exponent = scales >> 3
    mantissa = scales.bitwise_and(0x7)
    value = torch.where(
        exponent.eq(0),
        torch.ldexp(
            mantissa.float(), torch.full(exponent.shape, -17, dtype=torch.int32, device=scales.device)
        ),
        torch.ldexp(1.0 + mantissa.float() * 0.125, exponent.to(torch.int32) - 15),
    )
    return torch.where(scales.eq(0xFF), torch.full_like(value, torch.nan), value)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("block_size", [16, 32])
@pytest.mark.parametrize("m", [1, 17])
def test_mxfp4_ue5m3_gemm_matches_reference(block_size, m):
    torch.manual_seed(81 + block_size + m)
    n, k = 32, 64
    weight = torch.randint(0, 256, (n, k // 2), dtype=torch.uint8, device="cuda")
    scales = torch.randint(0, 0xFF, (n, k // block_size), dtype=torch.uint8, device="cuda")
    activations = torch.randn((m, k), dtype=torch.bfloat16, device="cuda")
    bias = torch.randn((n,), dtype=torch.bfloat16, device="cuda")

    actual = mxfp4_ue5m3_bf16_gemm(
        activations, prepare_mxfp4_ue5m3_weight(weight, scales, block_size), bias
    )
    decoded = _decode_e2m1_reference(weight)
    decoded = decoded * _decode_ue5m3_reference(scales).repeat_interleave(block_size, 1)
    expected = (activations.float() @ decoded.float().transpose(0, 1) + bias.float()).to(
        torch.bfloat16
    )
    finite = ~torch.isnan(expected)
    assert torch.equal(torch.isnan(actual), torch.isnan(expected))
    torch.testing.assert_close(actual[finite], expected[finite], rtol=2e-2, atol=2e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("block_size", [16, 32])
def test_mxfp4_ue5m3_nan_scale_propagates(block_size):
    n, k = 16, 32
    weight = torch.full((n, k // 2), 0x11, dtype=torch.uint8, device="cuda")
    scales = torch.full((n, k // block_size), 0x78, dtype=torch.uint8, device="cuda")
    scales[:, 0] = 0xFF
    x = torch.ones((1, k), dtype=torch.bfloat16, device="cuda")
    output = mxfp4_ue5m3_bf16_gemm(
        x, prepare_mxfp4_ue5m3_weight(weight, scales, block_size)
    )
    assert torch.isnan(output).all()
