# tests/test_api.py
import numpy as np
import vanish


def test_public_api_exposed():
    img = (np.random.default_rng(0).random((8, 10, 3)) * 255).astype(np.uint8)
    assert vanish.resize(img, width=8).shape == (8, 8, 3)
    assert vanish.enlarge(img, dwidth=2).shape == (8, 12, 3)
    assert hasattr(vanish, "remove_object")
    assert hasattr(vanish, "Carver")
