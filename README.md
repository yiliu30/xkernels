# marlin-kernels

`marlin-kernels` provides a CUDA JIT extension for dense MXFP4 W4A16 GEMM:
BF16 activations multiplied by packed E2M1 weights with E8M0 scales per
32-value block. The extension builds the first time GEMM is invoked and is
cached by PyTorch's extension cache. This first functional version preserves
the Marlin-style fused in-register dequantization contract; optimized Marlin
tile repacking and Tensor Core scheduling are deliberately a later step.

It also provides standalone CUDA conversion for raw UE5M3 scale bytes. UE5M3
is the unsigned PTX `EEEEE MMM` scale format; PyTorch represents its payload
as `uint8` bit-pattern tensors, not as a built-in FP8 dtype.

## Requirements

- CUDA-enabled PyTorch and NVCC
- NVIDIA SM80 or newer GPU
- Python 3.10+

## Usage

```python
from marlin_kernels import (
    fp32_to_ue5m3,
    mxfp4_bf16_gemm,
    prepare_mxfp4_weight,
    ue5m3_to_fp32,
)

prepared = prepare_mxfp4_weight(raw_e2m1_uint8, raw_e8m0_uint8)
out = mxfp4_bf16_gemm(activations_bf16, prepared, bias=None)

ue5m3 = fp32_to_ue5m3(scales_fp32)
scales_fp32 = ue5m3_to_fp32(ue5m3)
```

`raw_e2m1_uint8` has shape `[N, K / 2]`; the low then high nibble represent
consecutive E2M1 values. `raw_e8m0_uint8` has shape `[N, K / 32]` and contains
the E8M0 bit pattern for each 32-weight block.

`fp32_to_ue5m3` accepts contiguous, non-negative CUDA FP32 scale tensors and
returns raw CUDA `uint8` bytes. It uses round-to-nearest, ties-to-even;
positive overflow and infinity saturate to `0xfe`. Negative and NaN input are
rejected because they are not valid scale inputs. `ue5m3_to_fp32` decodes all
bytes, including the reserved `0xff` NaN encoding.

Run tests with:

```bash
/home/yi4l/workspace/vllm/.venv/bin/python -m pytest
```

## UE5M3 conversion performance

The software UE5M3 path was benchmarked on an NVIDIA A100-SXM4-80GB (SM80)
with PyTorch 2.13.0+cu130. Timings use 16,777,216 elements, 25 warm-up
iterations, and 200 CUDA-event-timed iterations; bandwidth counts input and
output bytes.

| Conversion | Median | Effective bandwidth |
|---|---:|---:|
| UE5M3 encode, raw extension | 0.163 ms | 515 GB/s |
| UE5M3 encode, public API | 0.370 ms | 227 GB/s |
| E4M3FN encode, `tensor.to(torch.float8_e4m3fn)` | 0.067 ms | 1,260 GB/s |
| UE5M3 decode | 0.138 ms | 607 GB/s |
| E4M3FN decode, `tensor.to(torch.float32)` | 0.109 ms | 773 GB/s |

The raw UE5M3 encoder is roughly 2.4x slower than the native E4M3FN cast on
this GPU, and UE5M3 decode is roughly 1.3x slower. The public encoder is
slower because validation performs whole-tensor negative and NaN checks before
launching the conversion kernel. These are software-fallback measurements;
UE5M3 native PTX conversion requires an `sm_107f`-family target and is not
included here.

## License and attribution

The E2M1/E8M0 dequantization design follows the Apache-2.0 Marlin code in
vLLM, originally adapted from IST-DASLab Marlin and modified by Neural Magic.
