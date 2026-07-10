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


def test_resize_enlarges_width():
    out = features.enlarge(_rand(8, 10), dwidth=3)
    assert out.shape == (8, 13, 3)


def test_enlarge_preserves_uniform_image():
    img = np.full((6, 8, 3), 55, dtype=np.uint8)
    out = features.enlarge(img, dwidth=4)
    assert out.shape == (6, 12, 3)
    assert np.all(out == 55)


def test_resize_noop_when_target_equals_current():
    img = _rand(5, 5)
    assert np.array_equal(features.resize(img, width=5), img)
