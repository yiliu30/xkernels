import math

import pytest

pytest.importorskip("torch")

import torch

from xkernels import fp32_to_ue5m3, fp32_to_ue5m3_checked, ue5m3_to_fp32


def test_encode_rejects_cpu_tensor():
    with pytest.raises(ValueError, match="CUDA"):
        fp32_to_ue5m3_checked(torch.zeros(1, dtype=torch.float32))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_encode_rejects_invalid_scales():
    for value, message in [(-1.0, "negative"), (float("nan"), "NaN")]:
        with pytest.raises(ValueError, match=message):
            fp32_to_ue5m3_checked(torch.tensor([value], device="cuda"))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_ue5m3_landmarks_and_saturation():
    encoded = fp32_to_ue5m3(
        torch.tensor([0.0, 2.0**-17, 2.0**-14, 114688.0, float("inf")], device="cuda")
    )
    assert torch.equal(
        encoded.cpu(), torch.tensor([0x00, 0x01, 0x08, 0xFE, 0xFE], dtype=torch.uint8)
    )
    decoded = ue5m3_to_fp32(torch.tensor([0x00, 0x01, 0x08, 0xFE, 0xFF], dtype=torch.uint8, device="cuda"))
    assert torch.equal(decoded[:4].cpu(), torch.tensor([0.0, 2.0**-17, 2.0**-14, 114688.0]))
    assert math.isnan(decoded[4].item())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_finite_ue5m3_round_trip():
    bits = torch.arange(0xFF, dtype=torch.uint8, device="cuda")
    assert torch.equal(fp32_to_ue5m3(ue5m3_to_fp32(bits)), bits)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_ue5m3_midpoints_round_to_even():
    codes = torch.arange(0xFE, dtype=torch.uint8, device="cuda")
    values = ue5m3_to_fp32(torch.arange(0xFF, dtype=torch.uint8, device="cuda"))
    midpoints = (values[:-1] + values[1:]) * 0.5
    expected_midpoints = torch.where(
        codes.bitwise_and(1).eq(0), codes, codes + 1
    )
    lower = torch.nextafter(midpoints, torch.full_like(midpoints, -torch.inf))
    upper = torch.nextafter(midpoints, torch.full_like(midpoints, torch.inf))
    assert torch.equal(fp32_to_ue5m3(lower), codes)
    assert torch.equal(fp32_to_ue5m3(midpoints), expected_midpoints)
    assert torch.equal(fp32_to_ue5m3(upper), codes + 1)
