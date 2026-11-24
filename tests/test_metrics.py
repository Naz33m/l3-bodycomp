"""Unit tests on synthetic phantoms: known geometry -> known answers."""
import numpy as np
import pytest

from ctbody import metrics as M
from ctbody.evaluation import dice, hd95
from ctbody.pipeline import compute
from ctbody.segment import Masks


def disk(shape, center, radius):
    yy, xx = np.ogrid[: shape[0], : shape[1]]
    return (yy - center[0]) ** 2 + (xx - center[1]) ** 2 <= radius ** 2


def phantom(nz=30, l3=(10, 19)):
    """Body of fat (-100 HU) with a muscle ring (50 HU); L3 block on given slices."""
    shape = (128, 128, nz)
    hu = np.full(shape, -1000.0, dtype=np.float32)       # air
    body = disk(shape[:2], (64, 64), 55)
    ring = disk(shape[:2], (64, 64), 40) & ~disk(shape[:2], (64, 64), 30)
    hu[body] = -100.0                                    # fat
    hu[ring] = 50.0                                      # muscle
    l3m = np.zeros(shape, bool)
    l3m[60:68, 60:68, l3[0]: l3[1] + 1] = True
    muscle = np.repeat(ring[:, :, None], nz, axis=2)
    return hu, l3m, muscle, ring


def test_window_bounds():
    w = M.window(np.array([-1000, -160, 40, 240, 3000]))
    assert w[0] == 0 and w[1] == 0 and w[-1] == 1 and w[-2] == 1
    assert w[2] == pytest.approx(0.5)


def test_l3_slice_is_centre_of_mass():
    _, l3m, _, _ = phantom(l3=(10, 18))
    assert M.l3_slice_index(l3m) == 14


def test_l3_missing_raises():
    with pytest.raises(ValueError, match="L3"):
        M.l3_slice_index(np.zeros((4, 4, 4), bool))


def test_area_uses_spacing():
    m = np.ones((10, 10), bool)
    assert M.area_cm2(m, (1.0, 1.0)) == pytest.approx(1.0)      # 100 mm²
    assert M.area_cm2(m, (0.8, 0.8)) == pytest.approx(0.64)


def test_hu_range_excludes_fat_inside_mask():
    hu, _, _, ring = phantom()
    hu2d = hu[:, :, 0].copy()
    sloppy = disk(hu2d.shape, (64, 64), 45)              # overshoots ring into fat
    s = M.tissue_stats(hu2d, sloppy, (1.0, 1.0), M.MUSCLE_HU)
    assert s.pixels == ring.sum()
    assert s.mean_hu == pytest.approx(50.0)


def test_smi_and_cutoffs():
    assert M.skeletal_muscle_index(160.0, 175.0) == pytest.approx(160 / 1.75 ** 2)
    assert M.sarcopenia(50.0, "male").flag                  # Prado: < 52.4
    assert not M.sarcopenia(40.0, "female").flag            # Prado: < 38.5
    assert M.sarcopenia(50.0, "male", bmi_value=27).flag     # Martin: < 53
    assert not M.sarcopenia(50.0, "male", bmi_value=22).flag # Martin: < 43
    assert M.myosteatosis(35.0, 22).flag and not M.myosteatosis(35.0, 27).flag


def test_compute_end_to_end_on_phantom():
    hu, l3m, muscle, ring = phantom()
    masks = Masks(l3=l3m, muscle=muscle, muscle_source="tissue_types")
    r = compute(hu, (0.8, 0.8, 3.0), masks, height_cm=170, sex="male", weight_kg=70)
    expected_area = ring.sum() * 0.64 / 100
    assert r["muscle"]["area_cm2"] == pytest.approx(expected_area)
    assert r["smi_cm2_m2"] == pytest.approx(expected_area / 1.7 ** 2)
    assert "sarcopenia" in r and "myosteatosis" in r
    assert r["warnings"] == []


def test_partial_muscle_does_not_apply_cutoffs():
    hu, l3m, muscle, _ = phantom()
    masks = Masks(l3=l3m, muscle=muscle, muscle_source="psoas+paraspinal")
    r = compute(hu, (1, 1, 6.0), masks, 170, "female")
    assert "sarcopenia" not in r and "psoas_paraspinal_index_cm2_m2" in r
    assert any("thick slices" in w for w in r["warnings"])


def test_dice_and_hd95():
    a = np.zeros((40, 40, 40), bool)
    a[10:20, 10:20, 10:20] = True
    assert dice(a, a) == 1.0 and hd95(a, a, (1, 1, 1)) == 0.0
    b = np.roll(a, 2, axis=0)                               # shift 2 voxels
    assert dice(a, b) == pytest.approx(0.8)
    assert hd95(a, b, (1, 1, 1)) == pytest.approx(2.0)
    assert hd95(a, b, (0.5, 1, 1)) == pytest.approx(1.0)    # spacing respected
    assert hd95(a, np.zeros_like(a), (1, 1, 1)) == float("inf")
    assert dice(np.zeros_like(a), np.zeros_like(a)) == 1.0
