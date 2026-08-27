// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// Standalone pybind entrypoint for the vendored dense Marlin implementation.

#include <torch/extension.h>

#include "marlin.cu"
#include "gptq_marlin_repack.cu"
#include "awq_marlin_repack.cu"
#include "marlin_int4_fp8_preprocess.cu"

// PyTorch's JIT extension loader does not perform CUDA device linking. Keep
// the generated upstream instantiations in this translation unit so the
// dispatcher's function pointers resolve without changing the kernel code.
#include "sm80_kernel_float16_u4b8_float16.cu"
#include "sm80_kernel_bfloat16_u4b8_bfloat16.cu"
#include "sm80_kernel_bfloat16_fe2m1f_bfloat16.cu"

torch::Tensor mxfp4_bf16_marlin_gemm(
    torch::Tensor& a, torch::Tensor& b_q_weight, torch::Tensor& b_scales,
    torch::Tensor& workspace, std::optional<torch::Tensor> const& bias,
    int64_t size_m, int64_t size_n, int64_t size_k, bool use_atomic_add,
    bool use_fp32_reduce) {
  return marlin_gemm(a, std::nullopt, b_q_weight, bias, b_scales,
                     std::nullopt, std::nullopt, std::nullopt, std::nullopt,
                     std::nullopt, workspace, vllm::kFE2M1f.id(), size_m,
                     size_n, size_k, true, use_atomic_add, use_fp32_reduce,
                     false);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("marlin_gemm", &marlin_gemm, "vLLM-compatible Marlin GEMM");
  m.def("gptq_marlin_repack", &gptq_marlin_repack,
        "vLLM-compatible GPTQ Marlin repack");
  m.def("awq_marlin_repack", &awq_marlin_repack,
        "vLLM-compatible AWQ Marlin repack");
  m.def("marlin_int4_fp8_preprocess", &marlin_int4_fp8_preprocess,
        "vLLM-compatible INT4 FP8 preprocess",
        pybind11::arg("qweight"), pybind11::arg("qzeros_or_none") = std::nullopt,
        pybind11::arg("inplace") = false);
  m.def("mxfp4_bf16_marlin_gemm", &mxfp4_bf16_marlin_gemm,
        "MXFP4 E2M1 with predecoded BF16 scales Marlin GEMM");
}
