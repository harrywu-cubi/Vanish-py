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


def cumulative_energy(energy):
    """Dynamic-programming table. M[r,c] = energy[r,c] + min of the three
    parents in row r-1 (columns c-1, c, c+1). Vectorized across columns within
    a row; the loop over rows is inherently sequential.

    Returns (M, backtrack) where backtrack[r,c] is the parent column OFFSET
    (-1, 0, or +1) chosen for cell (r,c)."""
    h, w = energy.shape
    M = energy.astype(np.float64).copy()
    backtrack = np.zeros((h, w), dtype=np.int64)
    for r in range(1, h):
        prev = M[r - 1]
        left = np.empty(w)
        left[0] = np.inf
        left[1:] = prev[:-1]
        up = prev
        right = np.empty(w)
        right[-1] = np.inf
        right[:-1] = prev[1:]
        stacked = np.stack([left, up, right])          # (3, w)
        choice = np.argmin(stacked, axis=0)            # 0=left,1=up,2=right
        M[r] += np.min(stacked, axis=0)
        backtrack[r] = choice - 1                      # map to offset -1/0/+1
    return M, backtrack


def find_vertical_seam(energy):
    """Lowest-energy top-to-bottom seam as an array of column indices."""
    M, backtrack = cumulative_energy(energy)
    h, w = M.shape
    seam = np.zeros(h, dtype=np.int64)
    c = int(np.argmin(M[-1]))
    seam[-1] = c
    for r in range(h - 2, -1, -1):
        c = c + int(backtrack[r + 1, c])
        c = max(0, min(w - 1, c))
        seam[r] = c
    return seam
