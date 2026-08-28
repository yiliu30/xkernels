"""Marlin JIT-compiled CUDA kernels for MXFP4 GEMM and UE5M3 conversion."""

from .api import (
    MarlinMXFP4UE5M3Weight,
    MarlinMXFP4Weight,
    mxfp4_bf16_gemm,
    mxfp4_ue5m3_bf16_gemm,
    prepare_mxfp4_ue5m3_weight,
    prepare_mxfp4_weight,
)
from .ue5m3 import fp32_to_ue5m3, fp32_to_ue5m3_checked, ue5m3_to_fp32
from .mxfp4_ue5m3_marlin import (
    MarlinMXFP4UE5M3TensorCoreWeight,
    mxfp4_ue5m3_marlin_bf16_gemm,
    prepare_mxfp4_ue5m3_marlin_weight,
)

from .vllm_marlin import (
    awq_marlin_repack,
    gptq_marlin_repack,
    marlin_gemm,
    marlin_int4_fp8_preprocess,
)

__all__ = [
    "MarlinMXFP4Weight",
    "MarlinMXFP4UE5M3Weight",
    "MarlinMXFP4UE5M3TensorCoreWeight",
    "fp32_to_ue5m3",
    "fp32_to_ue5m3_checked",
    "mxfp4_bf16_gemm",
    "mxfp4_ue5m3_bf16_gemm",
    "mxfp4_ue5m3_marlin_bf16_gemm",
    "prepare_mxfp4_ue5m3_marlin_weight",
    "prepare_mxfp4_ue5m3_weight",
    "prepare_mxfp4_weight",
    "ue5m3_to_fp32",
    "awq_marlin_repack",
    "gptq_marlin_repack",
    "marlin_gemm",
    "marlin_int4_fp8_preprocess",
]
