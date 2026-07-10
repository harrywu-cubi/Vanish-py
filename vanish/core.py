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


def remove_seam(img, seam):
    """Delete one pixel per row along the seam. Works for (H,W,C) images and
    (H,W) 2D arrays (used to track indices/masks). Returns a copy one column
    narrower."""
    h, w = img.shape[:2]
    mask = np.ones((h, w), dtype=bool)
    mask[np.arange(h), seam] = False
    if img.ndim == 3:
        return img[mask].reshape(h, w - 1, img.shape[2])
    return img[mask].reshape(h, w - 1)


def insert_seam(img, seam):
    """Duplicate one pixel per row along the seam; the inserted pixel is the
    average of the seam pixel and its right neighbor (or the seam pixel itself
    at the right edge). Returns a copy one column wider."""
    h, w, c = img.shape
    out = np.zeros((h, w + 1, c), dtype=img.dtype)
    for r in range(h):
        col = int(seam[r])
        out[r, :col + 1] = img[r, :col + 1]
        if col + 1 < w:
            new_px = img[r, col:col + 2].mean(axis=0)
        else:
            new_px = img[r, col]
        out[r, col + 1] = new_px.astype(img.dtype)
        out[r, col + 2:] = img[r, col + 1:]
    return out


def compute_seams(img, num):
    """Find `num` lowest-energy seams, returned as column indices in the
    ORIGINAL image's coordinate system, in removal order. Removing on a working
    copy while tracking original indices prevents picking the same seam twice."""
    tmp = img.copy()
    h, w = img.shape[:2]
    index = np.tile(np.arange(w), (h, 1))
    seams = []
    for _ in range(num):
        seam = find_vertical_seam(energy_map(tmp))
        seams.append(index[np.arange(h), seam].copy())
        tmp = remove_seam(tmp, seam)
        index = remove_seam(index, seam)
    return seams
