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
