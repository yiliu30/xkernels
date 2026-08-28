// Standalone replacement for the small vLLM CUDA utility surface used here.
#pragma once

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

inline torch::Tensor marlin_empty(c10::IntArrayRef sizes,
                                  torch::ScalarType dtype,
                                  c10::Device device) {
  return torch::empty(sizes, torch::TensorOptions().dtype(dtype).device(device));
}
