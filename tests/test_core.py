# tests/test_core.py
import numpy as np
from vanish import core


def test_to_grayscale_shape_and_values():
    img = np.zeros((2, 3, 3), dtype=np.uint8)
    img[..., 0] = 255  # pure red
    gray = core.to_grayscale(img)
    assert gray.shape == (2, 3)
    assert np.allclose(gray, 0.2126 * 255)


def test_energy_uniform_image_is_zero():
    img = np.full((5, 5, 3), 120, dtype=np.uint8)
    e = core.energy_map(img)
    assert e.shape == (5, 5)
    assert np.allclose(e, 0.0)


def test_energy_peaks_at_vertical_edge():
    img = np.zeros((5, 6, 3), dtype=np.uint8)
    img[:, 3:, :] = 255
    e = core.energy_map(img)
    assert e[:, 2:4].mean() > e[:, 0].mean()


def test_cumulative_energy_small_known_table():
    e = np.array([[1.0, 4.0, 3.0],
                  [2.0, 1.0, 5.0],
                  [4.0, 2.0, 1.0]])
    M, back = core.cumulative_energy(e)
    assert np.allclose(M[0], [1, 4, 3])
    assert np.allclose(M[1], [3, 2, 8])
    assert np.allclose(M[2], [6, 4, 3])
    assert back.shape == (3, 3)
