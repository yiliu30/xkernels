"""Parity checks for the directly ported vLLM Marlin source."""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("vllm")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires CUDA"
)

from marlin_kernels.vllm_marlin import gptq_marlin_repack, marlin_gemm


def test_gptq_repack_matches_vllm():
    from vllm import _custom_ops as vllm_ops

    qweight = torch.randint(
        0, 2**31 - 1, (16, 64), dtype=torch.int32, device="cuda"
    )
    perm = torch.empty(0, dtype=torch.int32, device="cuda")
    actual = gptq_marlin_repack(qweight, perm, 128, 64, 4, False)
    expected = vllm_ops.gptq_marlin_repack(qweight, perm, 128, 64, 4, False)
    torch.testing.assert_close(actual, expected)


def test_gptq_bf16_gemm_matches_vllm():
    from vllm import _custom_ops as vllm_ops
    from vllm.model_executor.layers.quantization.utils.marlin_utils import (
        marlin_make_workspace_new,
    )
    from vllm.model_executor.layers.quantization.utils.marlin_utils_test import (
        marlin_quantize,
    )
    from vllm.scalar_type import scalar_types

    m, n, k = 8, 64, 128
    activations = torch.randn(m, k, dtype=torch.bfloat16, device="cuda")
    weight = torch.randn(k, n, dtype=torch.bfloat16, device="cuda")
    _, packed, scales, g_idx, perm, _ = marlin_quantize(
        weight, scalar_types.uint4b8, -1, False, input_dtype=torch.bfloat16
    )
    workspace = marlin_make_workspace_new(weight.device)
    args = (
        activations, None, packed, None, scales, None, None, None, g_idx,
        perm, workspace, scalar_types.uint4b8, m, n, k, False, False,
        True, False,
    )
    expected = vllm_ops.marlin_gemm(*args)
    actual = marlin_gemm(*args[:11], scalar_types.uint4b8.id, *args[12:])
    torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)
