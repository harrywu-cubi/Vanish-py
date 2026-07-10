# tests/test_io.py
import numpy as np
import pytest
from PIL import Image
from vanish import io


def test_load_normalizes_to_rgb(tmp_path):
    p = tmp_path / "gray.png"
    Image.fromarray(np.full((4, 5), 100, dtype=np.uint8), mode="L").save(p)
    img = io.load_image(str(p))
    assert img.shape == (4, 5, 3)
    assert img.dtype == np.uint8


def test_load_drops_alpha(tmp_path):
    p = tmp_path / "rgba.png"
    Image.fromarray(np.zeros((3, 3, 4), dtype=np.uint8), mode="RGBA").save(p)
    assert io.load_image(str(p)).shape == (3, 3, 3)


def test_save_and_roundtrip(tmp_path):
    img = (np.random.default_rng(0).random((6, 7, 3)) * 255).astype(np.uint8)
    p = tmp_path / "out.png"
    io.save_image(str(p), img)
    assert io.load_image(str(p)).shape == (6, 7, 3)


def test_load_mask_boolean_and_shape(tmp_path):
    m = np.zeros((4, 4), dtype=np.uint8)
    m[1:3, 1:3] = 255
    p = tmp_path / "mask.png"
    Image.fromarray(m, mode="L").save(p)
    mask = io.load_mask(str(p), (4, 4))
    assert mask.dtype == bool
    assert mask.sum() == 4


def test_load_mask_shape_mismatch_raises(tmp_path):
    p = tmp_path / "mask.png"
    Image.fromarray(np.zeros((2, 2), dtype=np.uint8), mode="L").save(p)
    with pytest.raises(ValueError):
        io.load_mask(str(p), (4, 4))


def test_load_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        io.load_image("does_not_exist_1234.png")
