# MXFP4 E2M1 + UE5M3 W4A16

## Format contract

This fallback uses the same MXFP4 payload packing as the E8M0 API. `weight`
is a CUDA `uint8` tensor shaped `[N, K / 2]`; each byte stores two E2M1
values, with the first K value in its low nibble and the second in its high
nibble.

Only the block-scale format changes. `scales` is a CUDA `uint8` tensor of raw
UE5M3 `EEEEE MMM` bytes with one scale per output channel and K block:

| Block size | Scale shape |
| --- | --- |
| 16 | `[N, K / 16]` |
| 32 | `[N, K / 32]` |

The decoded weight is `decode_e2m1(payload) * decode_ue5m3(scale)`. UE5M3
`0xff` is NaN and is intentionally propagated by GEMM; callers that need to
reject NaN scales should validate them before preparing weights.

## API

```python
prepared = prepare_mxfp4_ue5m3_weight(weight, scales, block_size=32)
output = mxfp4_ue5m3_bf16_gemm(activations, prepared, bias=None)
```

Activations and optional bias are CUDA BF16. The GEMM supports SM80+ as a
portable software fallback: E2M1 and UE5M3 are decoded in the CUDA kernel,
then accumulated in FP32 and converted to BF16 output.

This is not native UE5M3 `tcgen05.mma.kind::mxf4nvf4` execution. Native W4A4
UE5M3 Tensor Core support requires the SM107f family and CUDA 13.4 preview
toolchain support; it is outside this W4A16 fallback.
