# vanish/_naive.py
"""Naive per-pixel reference implementation. Exists ONLY as the slow baseline
for the benchmark and as a correctness oracle in tests. Not part of the API."""
import math
import numpy as np
from vanish.core import to_grayscale, SOBEL_X, SOBEL_Y


def energy_map_naive(img):
    gray = to_grayscale(img)
    h, w = gray.shape
    out = np.zeros((h, w))
    for r in range(h):
        for c in range(w):
            gx = gy = 0.0
            for i in range(3):
                for j in range(3):
                    rr = min(max(r + i - 1, 0), h - 1)
                    cc = min(max(c + j - 1, 0), w - 1)
                    gx += SOBEL_X[i, j] * gray[rr, cc]
                    gy += SOBEL_Y[i, j] * gray[rr, cc]
            out[r, c] = math.sqrt(gx * gx + gy * gy)
    return out


def cumulative_energy_naive(energy):
    h, w = energy.shape
    M = energy.astype(float).copy()
    for r in range(1, h):
        for c in range(w):
            best = M[r - 1, c]
            if c > 0:
                best = min(best, M[r - 1, c - 1])
            if c < w - 1:
                best = min(best, M[r - 1, c + 1])
            M[r, c] += best
    return M
