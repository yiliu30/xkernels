"""Thin, vLLM-compatible API for the directly ported Marlin source."""

from ._vllm_marlin_extension import load_vllm_marlin_extension


def marlin_gemm(*args, **kwargs):
    """Run the directly ported vLLM Marlin CUDA kernel."""
    return load_vllm_marlin_extension().marlin_gemm(*args, **kwargs)


def gptq_marlin_repack(*args, **kwargs):
    """Repack GPTQ weights into the vLLM Marlin layout."""
    return load_vllm_marlin_extension().gptq_marlin_repack(*args, **kwargs)


def awq_marlin_repack(*args, **kwargs):
    """Repack AWQ weights into the vLLM Marlin layout."""
    return load_vllm_marlin_extension().awq_marlin_repack(*args, **kwargs)


def marlin_int4_fp8_preprocess(*args, **kwargs):
    """Apply the vLLM Marlin INT4 FP8 preprocessing kernel."""
    return load_vllm_marlin_extension().marlin_int4_fp8_preprocess(*args, **kwargs)
