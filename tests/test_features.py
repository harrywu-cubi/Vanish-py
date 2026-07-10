# tests/test_features.py
import numpy as np
from vanish import features


def _rand(h, w):
    return (np.random.default_rng(0).random((h, w, 3)) * 255).astype(np.uint8)


def test_resize_shrinks_width_to_target():
    out = features.resize(_rand(8, 12), width=9)
    assert out.shape == (8, 9, 3)


def test_resize_shrinks_height_via_transpose():
    out = features.resize(_rand(10, 6), height=7)
    assert out.shape == (7, 6, 3)


def test_resize_width_and_height_together():
    out = features.resize(_rand(10, 12), width=9, height=8)
    assert out.shape == (8, 9, 3)


def test_enlarge_increases_width():
    out = features.enlarge(_rand(8, 10), dwidth=3)
    assert out.shape == (8, 13, 3)


def test_enlarge_increases_height():
    out = features.enlarge(_rand(8, 10), dheight=4)
    assert out.shape == (12, 10, 3)
    assert out.flags["C_CONTIGUOUS"]


def test_resize_enlarges_height_via_transpose():
    out = features.resize(_rand(6, 9), height=9)
    assert out.shape == (9, 9, 3)


def test_enlarge_preserves_uniform_image():
    img = np.full((6, 8, 3), 55, dtype=np.uint8)
    out = features.enlarge(img, dwidth=4)
    assert out.shape == (6, 12, 3)
    assert np.all(out == 55)


def test_enlarge_gradient_stays_monotonic_no_smear():
    # a strict horizontal gradient: column c is brighter than c-1
    w = 12
    img = np.zeros((5, w, 3), dtype=np.uint8)
    img[:, :, :] = (np.arange(w) * 20).astype(np.uint8)[None, :, None]
    out = features.enlarge(img, dwidth=5)
    assert out.shape == (5, w + 5, 3)
    # inserted seams (averaged neighbors) preserve left-to-right ordering;
    # a stacked/smeared insertion would break monotonicity somewhere
    row = out[0, :, 0].astype(int)
    assert np.all(np.diff(row) >= 0)


def test_resize_noop_when_target_equals_current():
    img = _rand(5, 5)
    assert np.array_equal(features.resize(img, width=5), img)


def test_remove_object_erases_masked_blob():
    img = np.full((12, 16, 3), 100, dtype=np.uint8)
    img[:, 7:9, :] = 250
    mask = np.zeros((12, 16), dtype=bool)
    mask[:, 7:9] = True
    out = features.remove_object(img, mask, shrink=True)
    assert out.shape == (12, 14, 3)
    assert out.max() < 200


def test_remove_object_restores_width_when_not_shrinking():
    img = np.full((12, 16, 3), 100, dtype=np.uint8)
    img[:, 7:9, :] = 250
    mask = np.zeros((12, 16), dtype=bool)
    mask[:, 7:9] = True
    out = features.remove_object(img, mask, shrink=False)
    assert out.shape == (12, 16, 3)


def test_remove_object_non_rectangular_mask():
    # an L-shaped mask that spans only some rows/columns
    img = np.full((14, 18, 3), 90, dtype=np.uint8)
    img[2:10, 6:9, :] = 240      # vertical part of the L
    img[8:10, 6:13, :] = 240     # horizontal part of the L
    mask = np.zeros((14, 18), dtype=bool)
    mask[2:10, 6:9] = True
    mask[8:10, 6:13] = True
    out = features.remove_object(img, mask, shrink=True)
    assert out.shape[0] == 14 and out.shape[1] < 18   # narrower, same height
    assert out.max() < 200                            # bright object gone
