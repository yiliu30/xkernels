# UE5M3 Scale Kernel Developer Spec
## PTX ISA 9.4 / FP4 Block-Scaled Tensor Core Development

**Status:** Kernel-development reference  
**Primary source:** NVIDIA PTX ISA 9.4, CUDA 13.4 Developer Preview  
**Scope:** UE5M3 scale format, conversion, E2M1 FP4 block-scaled MMA, quantization flow, architecture support, validation, and compatibility fallbacks.

> **Important:** NVIDIA spells the format **UE5M3**: unsigned, 5 exponent bits, 3 mantissa bits.
> Do not confuse it with signed FP8 E5M2.

> **Preview warning:** PTX ISA 9.4 is part of CUDA 13.4 Developer Preview. Re-check the final GA documentation before production deployment.

---

# 1. Executive summary

UE5M3 is a new 8-bit **unsigned scale format** introduced in PTX ISA 9.4.

Its primary FP4 use is:

```text
E2M1 FP4 values
    +
UE5M3 block scales
    ↓
tcgen05.mma(.sp)
.kind::mxf4nvf4
    ↓
block-scaled Tensor Core GEMM
```

Key differences from UE4M3:

| Property | UE4M3 | UE5M3 |
|---|---:|---:|
| Sign | none | none |
| Exponent bits | 4 | 5 |
| Mantissa bits | 3 | 3 |
| Storage | 1 byte, MSB padded zero | 1 byte |
| Min positive non-zero | `2^-9` | `2^-17` |
| Max finite | `448` | `114688` |
| Infinity | No | No |
| NaN encoding | `0x7f` | `0xff` |
| Native B200 `sm_100` scale | Yes | **No** |
| Native B300 `sm_103` scale | Yes | **No** |
| Native `sm_107f` scale | Yes | **Yes** |

The extra exponent bit expands the scale range by 256x in both directions while keeping the same 3-bit mantissa precision.

A UE5M3 FP4 recipe can avoid the separate global scale used by standard NVFP4-style hierarchical scaling, but this is a **quantization policy**, not a PTX requirement.

---

# 2. Normative PTX format

PTX ISA 9.4 defines `ue5m3` as:

- 8-bit unsigned floating-point format.
- 5 exponent bits.
- 3 mantissa bits.
- No sign bit.
- No infinity encoding.
- NaN limited to `0xff`.
- Scalar storage in a `.b8` register.
- Alternate floating-point format, not a fundamental PTX register type.

Conceptual encoding:

```text
bit:    7 6 5 4 3 | 2 1 0
       +-----------+-------+
       | exponent  | mant. |
       |  5 bits   | 3 bits|
       +-----------+-------+

       EEEEE MMM
```

Contrast:

```text
UE4M3: 0 EEEE MMM
UE5M3:   EEEEE MMM
```

---

# 3. Numeric landmarks

Useful values for validation:

```text
0x00 = 0

0x01 = 2^-17
       minimum positive non-zero UE5M3 value

0xFE = 114688
       maximum finite UE5M3 value

0xFF = NaN
```

Compared with UE4M3:

```text
UE4M3 min positive non-zero = 2^-9
UE5M3 min positive non-zero = 2^-17

UE4M3 max finite = 448
UE5M3 max finite = 114688
```

UE5M3 therefore extends scale range by 256x while retaining the same three mantissa bits.

---

# 4. Register and packing ABI

A scalar UE5M3 value is stored as bits:

```ptx
.reg .b8 scale;
```

UE5M3 is not a fundamental register type, so do not expect:

```ptx
.reg .ue5m3 scale;    // not the PTX register model
```

Two UE5M3 values can be packed as `.ue5m3x2` in `.b16`:

```ptx
.reg .b16 scales;
```

For two FP32 inputs:

```text
destination .b16

bits 15:8 = UE5M3(a)
bits  7:0 = UE5M3(b)
```

---

# 5. PTX 9.4 conversion support

PTX 9.4 adds `.ue5m3x2` to `cvt`.

Representative forms:

```ptx
cvt.rn.satfinite.ue5m3x2.f32 d, a, b;
cvt.rz.satfinite.ue5m3x2.f32 d, a, b;
cvt.rp.satfinite.ue5m3x2.f32 d, a, b;
```

PTX also defines a scaled conversion form:

```ptx
cvt.rn.satfinite.scaled::n1::ue8m0.ue5m3x2.f32
    d, a, b, scale;
```

Up-conversion includes forms such as:

```ptx
cvt.rn.f16x2.ue5m3x2 d, a;
cvt.rn.bf16x2.ue5m3x2 d, a;
```

### Target requirement

`cvt` with `.ue5m3x2` requires:

```text
sm_107f or higher in the same family
```

---

# 6. Rounding

PTX floating-point rounding modes include:

| Modifier | Meaning |
|---|---|
| `.rn` | nearest, ties to even |
| `.rz` | toward zero |
| `.rm` | toward negative infinity |
| `.rp` | toward positive infinity |
| `.rs` | stochastic-style PTX rounding |

For block-scale generation, the software recipe must define exactly how the ideal scale is mapped to UE5M3. Do not assume the ISA itself defines your block-scale selection policy.

---

# 7. Native FP4 Tensor Core use

PTX ISA 9.4 adds UE5M3 as a scale type for:

```text
tcgen05.mma
tcgen05.mma.sp
```

with:

```text
.kind::mxf4nvf4
element type = E2M1
scale type   = UE5M3
```

Conceptually:

```text
D = C + (A × scale_A) × (B × scale_B)
```

Elementwise over K:

```text
D[m,n] =
    C[m,n]
    +
    sum_k(
        A_fp4[m,k] * SFA[m, block(k)]
        *
        B_fp4[k,n] * SFB[block(k), n]
    )
```

Scale matrices are provided through Tensor Memory according to the `tcgen05` scale-factor layout.

---

# 8. Valid block sizes

For `.kind::mxf4nvf4` + E2M1 + UE5M3:

| MMA kind | Element | Scale | Valid mode |
|---|---|---|---|
| `mxf4nvf4` | E2M1 | UE5M3 | `scale_vec::2X / block32` |
| `mxf4nvf4` | E2M1 | UE5M3 | `scale_vec::4X / block16` |

For an NVFP4-like comparison, use:

```text
E2M1 + UE5M3 + block16
```

to preserve the usual 16-element scale granularity.

---

# 9. Scale dtype is part of the MMA descriptor

The scale byte is not self-describing.

For `.kind::mxf4nvf4`, PTX ISA 9.4 defines descriptor bits `23:24`:

```text
0 = UE4M3
1 = UE8M0
2 = UE5M3
```

Therefore:

```text
UE5M3 bytes
+
descriptor configured as UE4M3
```

is incorrect.

The kernel, descriptor, checkpoint metadata, and scale memory layout must agree.

---

# 10. Architecture support

PTX ISA 9.4 states:

```text
Support for scale type UE5M3 requires
sm_107f or higher in the same family.
Otherwise behavior is undefined.
```

Practical table:

| GPU / target | Native UE5M3 block-scaled FP4 MMA |
|---|---:|
| B200 / GB200 — `sm_100` | **No** |
| B300 / GB300 — `sm_103` | **No** |
| `sm_107f` family | **Yes** |
| `sm_107a` | **Yes** |
| `sm_110` family | **No; not inherited from sm_107f** |
| `sm_120` family | **No; not inherited from sm_107f** |

Important:

```text
sm_120 > sm_107
```

numerically does not imply feature inheritance. Family-specific `f` features are only guaranteed within the compatible family.

---

# 11. B200/B300 relationship

B200 and B300 have native FP4 block-scaled Tensor Core paths, but not UE5M3 scale decoding.

Typical B200/B300 native paths:

```text
NVFP4-like:
E2M1 + UE4M3 + block16
→ tcgen05.mma.kind::mxf4nvf4

MXFP4:
E2M1 + UE8M0 + block32
→ tcgen05.mma
```

Unsupported natively on B200/B300:

```text
E2M1 + UE5M3
```

Do not reinterpret UE5M3 scale bytes as UE4M3.

---

# 12. Recommended UE5M3 FP4 quantization recipe

This section is algorithm guidance, not a PTX requirement.

For E2M1:

```text
FP4_MAX = 6.0
```

For each block:

```text
amax_b = max(abs(x_b))

ideal_scale = amax_b / 6

s_b = Q_UE5M3(ideal_scale)

q_i = Q_E2M1(x_i / decode(s_b))

x_hat_i = q_i * decode(s_b)
```

Pseudo-code:

```python
FP4_MAX = 6.0

for block in blocks(x, BLOCK_SIZE):
    amax = max(abs(block))

    if amax == 0:
        scale = ue5m3_zero()
        q = zero_fp4_block()
        continue

    ideal_scale = amax / FP4_MAX
    scale = quantize_scale_to_ue5m3(ideal_scale)

    inv_scale = 1.0 / decode_ue5m3(scale)
    q = quantize_to_e2m1(block * inv_scale)
```

The exact scale-rounding policy should be specified in the software format contract.

---

# 13. Global-scale policy

### PTX requirement

There is **no PTX requirement** for a separate global FP32 scale.

`tcgen05.mma` consumes the block scale matrices supplied to the MMA.

### Quantization recipe

Standard NVFP4-style hierarchical scaling is commonly represented as:

```text
x_hat =
    E2M1
    × UE4M3 block scale
    × FP32 global/row scale
```

A UE5M3 recipe can instead use:

```text
x_hat =
    E2M1
    × UE5M3 block scale
```

with no global scale.

The motivation is UE5M3's wider exponent range, especially its ability to represent small block scales without zero-rounding.

The paper *Is Finer Better? The Limits of Microscaling Formats in Large Language Models* reports that UE5M3 block scales without per-tensor scaling can achieve accuracy comparable to UE4M3 with hierarchical scaling on the evaluated workloads.

This is a model/recipe result, not an ISA guarantee.

---

# 14. Suggested checkpoint/kernel data contract

Store enough metadata to fully define the format:

```yaml
element_dtype: e2m1
scale_dtype: ue5m3
block_size: 16
scale_axis: k
global_scale: none
scale_layout: tcgen05-compatible
value_rounding: rn
scale_rounding: explicit_recipe
```

Do not serialize only:

```yaml
dtype: fp4
```

The consumer needs at least:

```text
FP4 encoding
scale encoding
block size
scale axis
scale layout/swizzle
global/row-scale policy
rounding policy
MMA kind
```

---

# 15. Tensor Memory scale layout

`tcgen05` consumes scale matrices from Tensor Memory.

Examples:

```text
block32 / scale_vec::2X:
    K=64  → 2 scales per A row
    K=128 → 4 scales per A row

block16 / scale_vec::4X:
    K=64  → 4 scales per A row
    K=128 → 8 scales per A row
```

Do not assume a linear `[M, K/block]` array can be passed directly.

The scale bytes must be arranged in the exact TMEM scale-factor layout required by `tcgen05`.

Performance-oriented flow:

```text
GMEM packed scales
    ↓
TMA/load/layout transform
    ↓
TMEM scale matrix
    ↓
tcgen05.mma
```

Keep layout conversion away from the steady-state MMA critical path where possible.

---

# 16. Suggested native kernel flow

## Static weights

```text
BF16/FP32 weight
    ↓
per-block amax
    ↓
UE5M3 scale generation
    ↓
divide by decoded scale
    ↓
E2M1 quantization
    ↓
pack 2 FP4 values / byte
    ↓
store FP4 payload + UE5M3 scale bytes
```

Prefer offline preprocessing for weights.

## Dynamic activations

```text
BF16/FP16 activation tile
    ↓
block amax reduction
    ↓
FP32 → UE5M3 scale conversion
    ↓
E2M1 quantization
    ↓
pack FP4
    ↓
prepare scale layout
    ↓
tcgen05 MMA pipeline
```

Performance goals:

- Fuse amax + scale generation + E2M1 conversion where possible.
- Vectorize UE5M3 conversion.
- Avoid scalar conversion chains.
- Pipeline quantization with data movement and MMA.
- Avoid global-memory spill/reload of intermediate scales.
- Generate the final scale layout directly when practical.

---

# 17. Software oracle

Implement a bit-exact scalar oracle:

```cpp
uint8_t fp32_to_ue5m3_rn(float x);
uint8_t fp32_to_ue5m3_rz(float x);
uint8_t fp32_to_ue5m3_rp(float x);

float ue5m3_to_fp32(uint8_t x);

uint16_t fp32x2_to_ue5m3x2_rn(float a, float b);
```

Validation target on native hardware:

```text
software oracle bits
==
PTX cvt.ue5m3x2 bits
```

for all important boundaries.

---

# 18. Edge cases

## Zero block

```text
amax = 0
```

Define the zero-scale and zero-payload behavior explicitly.

## Tiny scales

Test around:

```text
2^-17
```

including just-below, exact, and just-above.

## Maximum scale

Test around:

```text
114688
```

and verify saturation behavior.

## NaN

```text
0xFF = NaN
```

Ensure saturation cannot accidentally produce the reserved NaN encoding.

## Negative scale

UE5M3 is unsigned. A quantizer should never intentionally generate a negative scale.

## Infinity

UE5M3 has no infinity. Explicitly define overflow behavior and `.satfinite` use.

---

# 19. B200/B300 fallback strategies

If a checkpoint is E2M1 + UE5M3 but must run on B200/B300:

## A. Requantize to native NVFP4

```text
UE5M3 FP4
    ↓
UE4M3 + global/row scale
    ↓
native B200/B300 FP4 MMA
```

Best throughput, but not bitwise-equivalent to native UE5M3 MMA.

## B. Decode to BF16

```text
E2M1 × UE5M3
    ↓
BF16 tile
    ↓
BF16 Tensor Core GEMM
```

Higher-fidelity fallback, lower throughput.

Still not guaranteed bitwise-identical to native UE5M3 MMA.

## C. Requantize to FP8

```text
E2M1 × UE5M3
    ↓
E4M3/E5M2
    ↓
FP8 Tensor Core GEMM
```

Potentially faster than BF16, but adds another quantization.

## D. Factor UE5M3 scale

For some tensors:

```text
UE5M3 scale
=
UE4M3 scale × 2^coarse_exponent
```

This can preserve represented values under constrained exponent ranges and keep native B200 FP4 MMA.

It still does not provide a general bitwise guarantee versus native UE5M3 Tensor Core arithmetic.

---

# 20. Bitwise-identical reference requirement

If the requirement is:

```text
B200 output bits
==
native sm_107 UE5M3 Tensor Core output bits
for every legal input
```

there is no supported high-speed B200 path that guarantees it.

Reason:

```text
Native UE5M3:
decode UE5M3
→ scale inside the native block-scaled MMA path
→ product
→ native internal accumulation

B200 fallback:
different scale representation or explicit decode
→ different MMA datatype/path
→ potentially different rounding points
```

For a bitwise oracle, use native `sm_107`-family UE5M3 `tcgen05.mma` with fixed:

```text
instruction shape
K tiling
accumulator type
compiler/toolchain
kernel schedule
```

---

# 21. Validation plan

## Level 1 — Format

Exhaustively test all 256 UE5M3 byte encodings.

Pay special attention to:

```text
0x00
0x01
subnormal/normal boundary
exponent boundaries
0xFE
0xFF
```

## Level 2 — Conversion

Generate FP32 values around every UE5M3 midpoint and representable boundary.

Compare software against PTX for relevant modes:

```text
rn
rz
rp
satfinite
```

## Level 3 — Quantizer

Fingerprint each block:

```text
input amax
ideal scale
scale byte
decoded scale
E2M1 payload
reconstructed block
```

## Level 4 — MMA input fingerprint

Before measuring accuracy or performance, verify:

```text
packed E2M1 A
packed E2M1 B
UE5M3 SFA bytes
UE5M3 SFB bytes
TMEM coordinates
MMA descriptor
block size
```

## Level 5 — MMA correctness

Compare native UE5M3 MMA against a high-precision reference.

Track:

```text
max_abs_error
max_rel_error
ULP/error distribution
NaN/Inf count
exact-bit-match rate
```

---

# 22. Performance checklist

- [ ] Use native `tcgen05.mma.kind::mxf4nvf4`.
- [ ] Choose `block16` or `block32` deliberately.
- [ ] Configure Scale Matrix Type = UE5M3.
- [ ] Produce exact TMEM scale layout.
- [ ] Pack E2M1 efficiently.
- [ ] Fuse activation amax + scale generation + FP4 quantization.
- [ ] Use packed `.ue5m3x2` conversion where beneficial.
- [ ] Pipeline quantization/data movement with MMA.
- [ ] Avoid scale conversion on the MMA dependency chain.
- [ ] Avoid round-trip UE5M3 → wider type → UE5M3 in the mainloop.
- [ ] Track register pressure from scale-generation logic.
- [ ] Measure TMEM/SMEM layout cost separately from MMA throughput.
- [ ] Benchmark block16 and block32 separately.
- [ ] Verify scale generation is not the new bottleneck.

---

# 23. Minimal kernel-oriented pseudo-code

```cpp
// Conceptual only.

for each output_tile {

    for each K_tile {

        // Dynamic A quantization
        for each A block {
            float amax = block_absmax(A_block);

            uint8_t sA =
                fp32_to_ue5m3_scale(amax / 6.0f);

            fp4_block qA =
                quantize_e2m1(
                    A_block / decode_ue5m3(sA)
                );

            write_packed_A(qA);
            write_scale_A_for_tmem(sA);
        }

        // B is typically prequantized.
        load_packed_E2M1_B();
        load_UE5M3_scale_B();

        stage_payloads();
        stage_scales_to_tmem();

        tcgen05_mma_mxf4nvf4_ue5m3(
            A_e2m1,
            B_e2m1,
            SFA_ue5m3,
            SFB_ue5m3,
            accumulator
        );
    }

    store_epilogue(accumulator);
}
```

---

# 24. Developer decision table

| Question | Answer |
|---|---|
| Is UE5M3 signed? | No |
| Storage | 8 bits |
| Exp / mantissa | 5 / 3 |
| Infinity | No |
| NaN | `0xff` |
| Fundamental PTX type | No |
| Scalar register storage | `.b8` |
| Packed conversion | `.ue5m3x2` in `.b16` |
| FP4 payload | E2M1 |
| MMA kind | `tcgen05.mma(.sp).kind::mxf4nvf4` |
| Block16 | Yes on valid target |
| Block32 | Yes on valid target |
| Native minimum target | `sm_107f` family |
| B200 native | No |
| B300 native | No |
| sm120 inherits support | No |
| Global FP32 scale required by PTX | No |
| No-global-scale recipe possible | Yes |
| B200 bitwise emulation guaranteed | No |
| Best bitwise reference | Native sm_107-family UE5M3 MMA |

---

# 25. Sources

## Normative NVIDIA sources

1. NVIDIA PTX ISA 9.4 — CUDA 13.4 Developer Preview  
   https://docs.nvidia.com/cuda/developer-preview/13.4/parallel-thread-execution/index.html

2. PTX ISA 9.4 PDF  
   https://docs.nvidia.com/cuda/developer-preview/13.4/pdf/ptx_isa_9.4.pdf

3. CUDA 13.4 Developer Preview Release Notes  
   https://docs.nvidia.com/cuda/developer-preview/13.4/cuda-toolkit-release-notes/index.html

4. CUTLASS Blackwell SM100 GEMMs  
   https://docs.nvidia.com/cutlass/latest/media/docs/cpp/blackwell_functionality.html

5. CUDA compute capability / family-specific target rules  
   https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/compute-capabilities.html

## Quantization motivation

6. *Is Finer Better? The Limits of Microscaling Formats in Large Language Models*  
   arXiv:2601.19026  
   https://arxiv.org/abs/2601.19026

---

# 26. Bottom line

Think of UE5M3 as:

```text
not a new FP4 value type,
but a wider-range positive 8-bit block-scale format
for E2M1 FP4 Tensor Core computation.
```

The native kernel contract is:

```text
E2M1 payload
+
UE5M3 scale bytes
+
block16/block32 agreement
+
correct TMEM scale layout
+
MMA descriptor Scale Matrix Type = UE5M3
+
sm_107-family target
```

The expected quantization benefit is:

```text
UE4M3 block scale + global scaling
                 ↓
potentially replace with
                 ↓
UE5M3 block scale alone
```

while keeping the same 3-bit scale mantissa and gaining a much wider exponent range.

The key hardware limitation is:

```text
B200 / B300 cannot execute UE5M3 block scaling natively.
Native UE5M3 tcgen05 FP4 support starts with the sm_107f family in PTX ISA 9.4.
```
