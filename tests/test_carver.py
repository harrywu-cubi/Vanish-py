# tests/test_carver.py
import numpy as np
from vanish.carver import Carver


def test_iter_removals_yields_narrowing_frames():
    img = (np.random.default_rng(0).random((8, 10, 3)) * 255).astype(np.uint8)
    c = Carver(img)
    widths = [frame.shape[1] for frame, seam in c.iter_removals(3)]
    assert widths == [9, 8, 7]


def test_iter_removals_yields_seam_of_full_height():
    img = (np.random.default_rng(0).random((8, 10, 3)) * 255).astype(np.uint8)
    c = Carver(img)
    _, seam = next(c.iter_removals(1))
    assert seam.shape == (8,)


def test_remove_returns_final_image_and_does_not_mutate_input():
    img = (np.random.default_rng(0).random((8, 10, 3)) * 255).astype(np.uint8)
    original = img.copy()
    out = Carver(img).remove(4)
    assert out.shape == (8, 6, 3)
    assert np.array_equal(img, original)
