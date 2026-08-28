"""Public umbrella namespace coverage."""

import xkernels


def test_marlin_api_is_available_from_both_public_namespaces():
    assert xkernels.marlin_gemm is xkernels.marlin.marlin_gemm
    assert xkernels.gptq_marlin_repack is xkernels.marlin.gptq_marlin_repack
    assert xkernels.mxfp4_bf16_gemm is xkernels.marlin.mxfp4_bf16_gemm
    assert (
        xkernels.prepare_mxfp4_ue5m3_weight
        is xkernels.marlin.prepare_mxfp4_ue5m3_weight
    )
    assert xkernels.ue5m3_to_fp32 is xkernels.marlin.ue5m3_to_fp32
