# vanish/core.py
"""Pure NumPy seam-carving algorithm. No I/O, no state."""
import numpy as np

_LUMA = np.array([0.2126, 0.7152, 0.0722])

SOBEL_X = np.array([[-1.0, 0.0, 1.0],
                    [-2.0, 0.0, 2.0],
                    [-1.0, 0.0, 1.0]])
SOBEL_Y = SOBEL_X.T


def to_grayscale(img):
    """(H,W,3) image -> (H,W) float64 luminance."""
    return img.astype(np.float64) @ _LUMA


def _convolve3x3(a, kernel):
    """Vectorized 3x3 convolution with edge padding. The 9-iteration loop is
    over kernel taps (constant), not over pixels."""
    p = np.pad(a, 1, mode="edge")
    h, w = a.shape
    out = np.zeros((h, w), dtype=np.float64)
    for i in range(3):
        for j in range(3):
            out += kernel[i, j] * p[i:i + h, j:j + w]
    return out


def energy_map(img):
    """Gradient-magnitude energy: bright = important."""
    gray = to_grayscale(img)
    gx = _convolve3x3(gray, SOBEL_X)
    gy = _convolve3x3(gray, SOBEL_Y)
    return np.sqrt(gx * gx + gy * gy)
