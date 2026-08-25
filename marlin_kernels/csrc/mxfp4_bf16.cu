// SPDX-License-Identifier: Apache-2.0
// Initial MXFP4 W4A16 implementation. The dequantization sequence is adapted
// from Marlin's E2M1/E8M0 register conversion path.

#include <cuda_bf16.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <torch/extension.h>

namespace {

__device__ __forceinline__ float decode_e2m1(unsigned int bits) {
  const int sign = bits >> 3;
  const int exponent = (bits >> 1) & 0x3;
  const int mantissa = bits & 1;
  if (exponent == 0) {
    const float value = 0.5f * mantissa;
    return sign ? -value : value;
  }
  const float value = ldexpf(1.0f + 0.5f * mantissa, exponent - 1);
  return sign ? -value : value;
}

__device__ __forceinline__ float decode_e8m0(unsigned int bits) {
  // E8M0 is an unsigned power of two, encoded as exponent + 127.
  return ldexpf(1.0f, static_cast<int>(bits) - 127);
}

__device__ __forceinline__ float decode_ue5m3(unsigned int bits) {
  if (bits == 0xff) return nanf("");
  const int exponent = bits >> 3;
  const int mantissa = bits & 0x7;
  if (exponent == 0) return ldexpf(static_cast<float>(mantissa), -17);
  return ldexpf(1.0f + 0.125f * mantissa, exponent - 15);
}

__global__ void mxfp4_bf16_kernel(
    const __nv_bfloat16* __restrict__ a, const unsigned char* __restrict__ w,
    const unsigned char* __restrict__ scales, const __nv_bfloat16* __restrict__ bias,
    __nv_bfloat16* __restrict__ out, int m, int n, int k, bool has_bias) {
  const int row = blockIdx.y;
  const int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= m || col >= n) return;

  float acc = has_bias ? __bfloat162float(bias[col]) : 0.0f;
  const __nv_bfloat16* a_row = a + row * k;
  const unsigned char* w_row = w + col * (k / 2);
  const unsigned char* s_row = scales + col * (k / 32);
  for (int kk = 0; kk < k; ++kk) {
    const unsigned char packed = w_row[kk >> 1];
    const unsigned int e2m1 = (kk & 1) ? packed >> 4 : packed & 0xF;
    const float weight = decode_e2m1(e2m1) * decode_e8m0(s_row[kk >> 5]);
    acc += __bfloat162float(a_row[kk]) * weight;
  }
  out[row * n + col] = __float2bfloat16_rn(acc);
}

__global__ void mxfp4_ue5m3_bf16_kernel(
    const __nv_bfloat16* __restrict__ a, const unsigned char* __restrict__ w,
    const unsigned char* __restrict__ scales, const __nv_bfloat16* __restrict__ bias,
    __nv_bfloat16* __restrict__ out, int m, int n, int k, int block_size,
    bool has_bias) {
  const int row = blockIdx.y;
  const int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= m || col >= n) return;

  float acc = has_bias ? __bfloat162float(bias[col]) : 0.0f;
  const __nv_bfloat16* a_row = a + row * k;
  const unsigned char* w_row = w + col * (k / 2);
  const unsigned char* s_row = scales + col * (k / block_size);
  bool has_nan_scale = false;
  for (int kk = 0; kk < k; ++kk) {
    const unsigned char scale_bits = s_row[kk / block_size];
    if (scale_bits == 0xff) {
      has_nan_scale = true;
      break;
    }
    const unsigned char packed = w_row[kk >> 1];
    const unsigned int e2m1 = (kk & 1) ? packed >> 4 : packed & 0xF;
    const float weight = decode_e2m1(e2m1) * decode_ue5m3(scale_bits);
    acc += __bfloat162float(a_row[kk]) * weight;
  }
  if (has_nan_scale) {
    // Construct BF16 quiet NaN bits explicitly rather than relying on a
    // floating-point conversion that could canonicalize under compiler flags.
    reinterpret_cast<unsigned short*>(out)[row * n + col] = 0x7fc0;
  } else {
    out[row * n + col] = __float2bfloat16_rn(acc);
  }
}

torch::Tensor mxfp4_bf16_gemm(torch::Tensor a, torch::Tensor w,
                               torch::Tensor scales,
                               c10::optional<torch::Tensor> bias) {
  TORCH_CHECK(a.is_cuda() && w.is_cuda() && scales.is_cuda(), "inputs must be CUDA");
  TORCH_CHECK(a.scalar_type() == torch::kBFloat16, "a must be bfloat16");
  TORCH_CHECK(w.scalar_type() == torch::kUInt8 && scales.scalar_type() == torch::kUInt8,
              "weight and scales must be uint8");
  TORCH_CHECK(a.is_contiguous() && w.is_contiguous() && scales.is_contiguous(),
              "inputs must be contiguous");
  TORCH_CHECK(w.dim() == 2 && scales.dim() == 2, "weight and scales must be rank 2");
  const int m = a.size(0), k = a.size(1), n = w.size(0);
  TORCH_CHECK(k % 32 == 0 && w.size(1) == k / 2,
              "weight must have shape [N, K / 2] with K divisible by 32");
  TORCH_CHECK(scales.size(0) == n && scales.size(1) == k / 32,
              "scales must have shape [N, K / 32]");
  if (bias.has_value()) {
    TORCH_CHECK(bias->is_cuda() && bias->is_contiguous() &&
                    bias->scalar_type() == torch::kBFloat16 &&
                    bias->dim() == 1 && bias->size(0) == n,
                "bias must be a contiguous CUDA bfloat16 tensor with shape [N]");
  }
  auto out = torch::empty({m, n}, a.options());
  constexpr int threads = 128;
  const dim3 blocks((n + threads - 1) / threads, m);
  const bool has_bias = bias.has_value();
  const auto* bias_ptr =
      has_bias ? reinterpret_cast<const __nv_bfloat16*>(bias->data_ptr()) : nullptr;
  mxfp4_bf16_kernel<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const __nv_bfloat16*>(a.data_ptr()), w.data_ptr<unsigned char>(),
      scales.data_ptr<unsigned char>(), bias_ptr,
      reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), m, n, k, has_bias);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

torch::Tensor mxfp4_ue5m3_bf16_gemm(
    torch::Tensor a, torch::Tensor w, torch::Tensor scales, int64_t block_size,
    c10::optional<torch::Tensor> bias) {
  TORCH_CHECK(block_size == 16 || block_size == 32,
              "block_size must be 16 or 32");
  TORCH_CHECK(a.is_cuda() && w.is_cuda() && scales.is_cuda(), "inputs must be CUDA");
  TORCH_CHECK(a.scalar_type() == torch::kBFloat16, "a must be bfloat16");
  TORCH_CHECK(w.scalar_type() == torch::kUInt8 && scales.scalar_type() == torch::kUInt8,
              "weight and scales must be uint8");
  TORCH_CHECK(a.is_contiguous() && w.is_contiguous() && scales.is_contiguous(),
              "inputs must be contiguous");
  TORCH_CHECK(w.dim() == 2 && scales.dim() == 2, "weight and scales must be rank 2");
  const int m = a.size(0), k = a.size(1), n = w.size(0);
  TORCH_CHECK(k % block_size == 0 && w.size(1) == k / 2,
              "weight must have shape [N, K / 2] with K block-aligned");
  TORCH_CHECK(scales.size(0) == n && scales.size(1) == k / block_size,
              "scales must have shape [N, K / block_size]");
  if (bias.has_value()) {
    TORCH_CHECK(bias->is_cuda() && bias->is_contiguous() &&
                    bias->scalar_type() == torch::kBFloat16 &&
                    bias->dim() == 1 && bias->size(0) == n,
                "bias must be a contiguous CUDA bfloat16 tensor with shape [N]");
  }
  auto out = torch::empty({m, n}, a.options());
  constexpr int threads = 128;
  const dim3 blocks((n + threads - 1) / threads, m);
  const bool has_bias = bias.has_value();
  const auto* bias_ptr = has_bias ? reinterpret_cast<const __nv_bfloat16*>(bias->data_ptr()) : nullptr;
  mxfp4_ue5m3_bf16_kernel<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const __nv_bfloat16*>(a.data_ptr()), w.data_ptr<unsigned char>(),
      scales.data_ptr<unsigned char>(), bias_ptr,
      reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), m, n, k, block_size, has_bias);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("mxfp4_bf16_gemm", &mxfp4_bf16_gemm, "MXFP4 W4A16 BF16 GEMM");
  m.def("mxfp4_ue5m3_bf16_gemm", &mxfp4_ue5m3_bf16_gemm,
        "MXFP4 UE5M3 W4A16 BF16 GEMM");
}
