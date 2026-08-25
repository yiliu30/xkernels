// SPDX-License-Identifier: Apache-2.0
// Software UE5M3 scale conversion for targets without native PTX support.

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <torch/extension.h>

#include <cmath>
#include <cstdint>

namespace {

constexpr float kUE5M3MinNormal = 0x1.0p-14f;
constexpr float kUE5M3MaxFinite = 114688.0f;

__device__ __forceinline__ uint8_t fp32_to_ue5m3_rn_satfinite(float value) {
  if (isnan(value)) {
    return 0xff;
  }
  if (value <= 0.0f) {
    return 0;
  }
  if (value == 0.0f) {
    return 0;
  }
  if (isinf(value) || value >= kUE5M3MaxFinite) {
    return 0xfe;
  }

  if (value < kUE5M3MinNormal) {
    const int mantissa = __float2int_rn(ldexpf(value, 17));
    return static_cast<uint8_t>(mantissa);
  }

  int exponent;
  const float significand = frexpf(value, &exponent);
  int mantissa = __float2int_rn((significand - 0.5f) * 16.0f);
  int encoded_exponent = exponent + 14;
  if (mantissa == 8) {
    mantissa = 0;
    ++encoded_exponent;
  }
  if (encoded_exponent > 31 ||
      (encoded_exponent == 31 && mantissa == 7)) {
    return 0xfe;
  }
  return static_cast<uint8_t>((encoded_exponent << 3) | mantissa);
}

__device__ __forceinline__ float ue5m3_to_fp32_value(uint8_t bits) {
  const int exponent = bits >> 3;
  const int mantissa = bits & 0x7;
  if (bits == 0xff) {
    return nanf("");
  }
  if (exponent == 0) {
    return ldexpf(static_cast<float>(mantissa), -17);
  }
  return ldexpf(1.0f + static_cast<float>(mantissa) * 0.125f, exponent - 15);
}

__global__ void fp32_to_ue5m3_kernel(
    const float* input,
    uint8_t* output,
    int64_t numel) {
  for (int64_t index = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x;
       index < numel;
       index += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    output[index] = fp32_to_ue5m3_rn_satfinite(input[index]);
  }
}

__global__ void ue5m3_to_fp32_kernel(
    const uint8_t* input,
    float* output,
    int64_t numel) {
  for (int64_t index = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x;
       index < numel;
       index += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    output[index] = ue5m3_to_fp32_value(input[index]);
  }
}

void validate_input(const torch::Tensor& input, c10::ScalarType dtype) {
  TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
  TORCH_CHECK(input.scalar_type() == dtype, "input has an unexpected dtype");
  TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
}

torch::Tensor fp32_to_ue5m3(torch::Tensor input) {
  validate_input(input, torch::kFloat);
  auto output = torch::empty(input.sizes(), input.options().dtype(torch::kUInt8));
  const int64_t numel = input.numel();
  if (numel == 0) {
    return output;
  }
  constexpr int threads = 256;
  const int blocks = static_cast<int>((numel + threads - 1) / threads);
  fp32_to_ue5m3_kernel<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
      input.data_ptr<float>(), output.data_ptr<uint8_t>(), numel);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

torch::Tensor ue5m3_to_fp32(torch::Tensor input) {
  validate_input(input, torch::kUInt8);
  auto output = torch::empty(input.sizes(), input.options().dtype(torch::kFloat));
  const int64_t numel = input.numel();
  if (numel == 0) {
    return output;
  }
  constexpr int threads = 256;
  const int blocks = static_cast<int>((numel + threads - 1) / threads);
  ue5m3_to_fp32_kernel<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
      input.data_ptr<uint8_t>(), output.data_ptr<float>(), numel);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

} // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("fp32_to_ue5m3", &fp32_to_ue5m3, "Encode FP32 scales as UE5M3 bytes");
  m.def("ue5m3_to_fp32", &ue5m3_to_fp32, "Decode UE5M3 bytes to FP32");
}
