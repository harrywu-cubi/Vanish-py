# tests/test_naive.py
import numpy as np
from vanish import core, _naive


def test_naive_energy_matches_vectorized():
    img = (np.random.default_rng(0).random((12, 10, 3)) * 255).astype(np.uint8)
    assert np.allclose(_naive.energy_map_naive(img), core.energy_map(img))


def test_naive_cumulative_matches_vectorized():
    e = np.random.default_rng(1).random((12, 10))
    M_naive = _naive.cumulative_energy_naive(e)
    M_vec, _ = core.cumulative_energy(e)
    assert np.allclose(M_naive, M_vec)
