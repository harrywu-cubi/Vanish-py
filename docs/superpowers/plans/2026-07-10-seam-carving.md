# Seam Carving (Vanish-py) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a content-aware image resizer (shrink, enlarge, object-removal) on a shared vectorized NumPy seam-carving core, exposed as an importable library and a CLI.

**Architecture:** A pure-functions core (`vanish/core.py`) operates on NumPy arrays only (energy map → DP cumulative table → seam find → seam remove/insert). A stateful `Carver` driver streams removals. A feature layer composes the core into `resize`/`enlarge`/`remove_object`. An I/O layer isolates Pillow. A thin argparse CLI dispatches subcommands. A quarantined naive reference (`_naive.py`) exists only to produce a benchmark speedup.

**Tech Stack:** Python 3.11+, NumPy, Pillow (I/O), imageio (GIF, examples only), pytest.

---

## File Structure

- `pyproject.toml` — packaging + deps + pytest config
- `vanish/__init__.py` — public API re-exports
- `vanish/core.py` — pure algorithm: `to_grayscale`, `energy_map`, `cumulative_energy`, `find_vertical_seam`, `remove_seam`, `insert_seam`, `compute_seams`
- `vanish/carver.py` — `Carver` driver (iteration + frame streaming)
- `vanish/features.py` — `resize`, `enlarge`, `remove_object` (+ transpose handling)
- `vanish/io.py` — `load_image`, `save_image`, `load_mask`
- `vanish/cli.py` — argparse subcommands
- `vanish/__main__.py` — entry point
- `vanish/_naive.py` — naive-loop reference (benchmark only)
- `examples/*.py` + `examples/assets/` — demo scripts + bundled image/mask
- `tests/test_*.py` — pytest suites

**Convention used throughout:** images are `np.ndarray` of shape `(H, W, 3)`, dtype `uint8`. Seams are `np.ndarray` of shape `(H,)`, dtype `int64`, one column index per row. Energy maps are `float64` of shape `(H, W)`.

---

## Task 0: Project scaffolding

**Files:**
- Create: `pyproject.toml`
- Create: `vanish/__init__.py` (empty for now)
- Create: `tests/__init__.py` (empty)

- [ ] **Step 1: Create `pyproject.toml`**

```toml
[project]
name = "vanish"
version = "0.1.0"
description = "Content-aware image resizing via seam carving"
requires-python = ">=3.11"
dependencies = ["numpy>=1.24", "pillow>=10.0"]

[project.optional-dependencies]
examples = ["imageio>=2.31"]
dev = ["pytest>=7.4", "imageio>=2.31"]

[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[tool.setuptools.packages.find]
include = ["vanish*"]

[tool.pytest.ini_options]
testpaths = ["tests"]
```

- [ ] **Step 2: Create empty package files**

```bash
mkdir -p vanish tests examples/assets
: > vanish/__init__.py
: > tests/__init__.py
```

- [ ] **Step 3: Install in editable mode**

Run: `pip install -e ".[dev]"`
Expected: ends with `Successfully installed vanish-0.1.0` (plus numpy/pillow/pytest/imageio).

- [ ] **Step 4: Verify pytest runs (no tests yet)**

Run: `pytest -q`
Expected: `no tests ran` (exit code 5) — confirms pytest is wired up.

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml vanish/__init__.py tests/__init__.py
git commit -m "chore: scaffold vanish package"
```

---

## Task 1: `to_grayscale` and `energy_map`

**Files:**
- Create: `vanish/core.py`
- Test: `tests/test_core.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_core.py
import numpy as np
from vanish import core


def test_to_grayscale_shape_and_values():
    img = np.zeros((2, 3, 3), dtype=np.uint8)
    img[..., 0] = 255  # pure red
    gray = core.to_grayscale(img)
    assert gray.shape == (2, 3)
    # luminance weight of red is 0.2126 -> 0.2126*255
    assert np.allclose(gray, 0.2126 * 255)


def test_energy_uniform_image_is_zero():
    img = np.full((5, 5, 3), 120, dtype=np.uint8)
    e = core.energy_map(img)
    assert e.shape == (5, 5)
    assert np.allclose(e, 0.0)


def test_energy_peaks_at_vertical_edge():
    # left half black, right half white -> highest energy at the boundary columns
    img = np.zeros((5, 6, 3), dtype=np.uint8)
    img[:, 3:, :] = 255
    e = core.energy_map(img)
    # boundary columns (2 and 3) carry more energy than the flat interior column 0
    assert e[:, 2:4].mean() > e[:, 0].mean()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_core.py -q`
Expected: FAIL — `AttributeError: module 'vanish.core' has no attribute 'to_grayscale'`.

- [ ] **Step 3: Implement `core.py` grayscale + energy**

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_core.py -q`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add vanish/core.py tests/test_core.py
git commit -m "feat: energy map via vectorized Sobel gradients"
```

---

## Task 2: `cumulative_energy` (the DP table + backtrack)

**Files:**
- Modify: `vanish/core.py`
- Test: `tests/test_core.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_core.py`:

```python
def test_cumulative_energy_small_known_table():
    # 3x3 energy where the minimum path is hand-verifiable
    e = np.array([[1.0, 4.0, 3.0],
                  [2.0, 1.0, 5.0],
                  [4.0, 2.0, 1.0]])
    M, back = core.cumulative_energy(e)
    # row 0 unchanged
    assert np.allclose(M[0], [1, 4, 3])
    # row 1: each cell + min of the three parents above
    # M[1,0] = 2 + min(1,4) = 3 ; M[1,1] = 1 + min(1,4,3) = 2 ; M[1,2] = 5 + min(4,3) = 8
    assert np.allclose(M[1], [3, 2, 8])
    # row 2: M[2,0]=4+min(3,2)=6 ; M[2,1]=2+min(3,2,8)=4 ; M[2,2]=1+min(2,8)=3
    assert np.allclose(M[2], [6, 4, 3])
    # backtrack stores parent column OFFSET (-1, 0, +1); row 0 is 0
    assert back.shape == (3, 3)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_core.py::test_cumulative_energy_small_known_table -q`
Expected: FAIL — `has no attribute 'cumulative_energy'`.

- [ ] **Step 3: Implement `cumulative_energy`**

Append to `vanish/core.py`:

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_core.py::test_cumulative_energy_small_known_table -q`
Expected: 1 passed.

- [ ] **Step 5: Commit**

```bash
git add vanish/core.py tests/test_core.py
git commit -m "feat: cumulative-energy DP table with backtrack"
```

---

## Task 3: `find_vertical_seam`

**Files:**
- Modify: `vanish/core.py`
- Test: `tests/test_core.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_core.py`:

```python
def test_find_vertical_seam_on_known_table():
    e = np.array([[1.0, 4.0, 3.0],
                  [2.0, 1.0, 5.0],
                  [4.0, 2.0, 1.0]])
    seam = core.find_vertical_seam(e)
    # cheapest end is M[2,2]=3; backtracking: col2 <- col1 (M[1,1]) <- col0 (M[0,0])
    assert seam.tolist() == [0, 1, 2]


def test_seam_is_connected_and_full_height():
    rng = np.random.default_rng(0)
    e = rng.random((20, 15))
    seam = core.find_vertical_seam(e)
    assert seam.shape == (20,)
    assert np.all(np.abs(np.diff(seam)) <= 1)   # 8-connected
    assert seam.min() >= 0 and seam.max() < 15
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_core.py -k find_vertical_seam -q`
Expected: FAIL — `has no attribute 'find_vertical_seam'`.

- [ ] **Step 3: Implement `find_vertical_seam`**

Append to `vanish/core.py`:

```python
def find_vertical_seam(energy):
    """Lowest-energy top-to-bottom seam as an array of column indices."""
    M, backtrack = cumulative_energy(energy)
    h, w = M.shape
    seam = np.zeros(h, dtype=np.int64)
    c = int(np.argmin(M[-1]))
    seam[-1] = c
    for r in range(h - 2, -1, -1):
        c = c + int(backtrack[r + 1, c])   # parent column of row r+1's cell c
        c = max(0, min(w - 1, c))
        seam[r] = c
    return seam
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_core.py -k find_vertical_seam -q`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add vanish/core.py tests/test_core.py
git commit -m "feat: recover lowest-energy vertical seam via backtracking"
```

---

## Task 4: `remove_seam`

**Files:**
- Modify: `vanish/core.py`
- Test: `tests/test_core.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_core.py`:

```python
def test_remove_seam_rgb_shape():
    img = np.arange(4 * 4 * 3, dtype=np.uint8).reshape(4, 4, 3)
    seam = np.array([1, 2, 1, 0])
    out = core.remove_seam(img, seam)
    assert out.shape == (4, 3, 3)


def test_remove_seam_drops_the_seam_pixels():
    img = np.zeros((3, 3, 3), dtype=np.uint8)
    img[np.arange(3), np.array([0, 1, 2])] = 255   # mark the seam pixels
    out = core.remove_seam(img, np.array([0, 1, 2]))
    assert out.shape == (3, 2, 3)
    assert out.max() == 0                          # every marked pixel removed


def test_remove_seam_2d_index_array():
    idx = np.tile(np.arange(4), (2, 1))            # (2,4) int grid
    out = core.remove_seam(idx, np.array([1, 2]))
    assert out.shape == (2, 3)
    assert out[0].tolist() == [0, 2, 3]
    assert out[1].tolist() == [0, 1, 3]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_core.py -k remove_seam -q`
Expected: FAIL — `has no attribute 'remove_seam'`.

- [ ] **Step 3: Implement `remove_seam`**

Append to `vanish/core.py`:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_core.py -k remove_seam -q`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add vanish/core.py tests/test_core.py
git commit -m "feat: remove_seam for image and index arrays"
```

---

## Task 5: `insert_seam` and `compute_seams`

**Files:**
- Modify: `vanish/core.py`
- Test: `tests/test_core.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_core.py`:

```python
def test_insert_seam_widens_by_one():
    img = np.arange(3 * 4 * 3, dtype=np.uint8).reshape(3, 4, 3)
    seam = np.array([1, 2, 0])
    out = core.insert_seam(img, seam)
    assert out.shape == (3, 5, 3)


def test_insert_seam_preserves_uniform_image():
    img = np.full((4, 5, 3), 77, dtype=np.uint8)
    out = core.insert_seam(img, np.array([2, 2, 2, 2]))
    assert out.shape == (4, 6, 3)
    assert np.all(out == 77)   # averaging duplicates of equal pixels stays flat


def test_compute_seams_returns_original_coordinates():
    rng = np.random.default_rng(1)
    img = (rng.random((10, 12, 3)) * 255).astype(np.uint8)
    seams = core.compute_seams(img, 3)
    assert len(seams) == 3
    for s in seams:
        assert s.shape == (10,)
        assert s.min() >= 0 and s.max() < 12   # indices valid in the ORIGINAL
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_core.py -k "insert_seam or compute_seams" -q`
Expected: FAIL — `has no attribute 'insert_seam'`.

- [ ] **Step 3: Implement `insert_seam` and `compute_seams`**

Append to `vanish/core.py`:

```python
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
    index = np.tile(np.arange(w), (h, 1))     # index[r,c] = original column
    seams = []
    for _ in range(num):
        seam = find_vertical_seam(energy_map(tmp))
        seams.append(index[np.arange(h), seam].copy())
        tmp = remove_seam(tmp, seam)
        index = remove_seam(index, seam)
    return seams
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_core.py -k "insert_seam or compute_seams" -q`
Expected: 3 passed.

- [ ] **Step 5: Run the whole core suite**

Run: `pytest tests/test_core.py -q`
Expected: all core tests pass (12 total).

- [ ] **Step 6: Commit**

```bash
git add vanish/core.py tests/test_core.py
git commit -m "feat: insert_seam and multi-seam computation in original coords"
```

---

## Task 6: I/O layer

**Files:**
- Create: `vanish/io.py`
- Test: `tests/test_io.py`

- [ ] **Step 1: Write the failing tests**

```python
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
    assert mask.sum() == 4          # the white 2x2 block


def test_load_mask_shape_mismatch_raises(tmp_path):
    p = tmp_path / "mask.png"
    Image.fromarray(np.zeros((2, 2), dtype=np.uint8), mode="L").save(p)
    with pytest.raises(ValueError):
        io.load_mask(str(p), (4, 4))


def test_load_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        io.load_image("does_not_exist_1234.png")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_io.py -q`
Expected: FAIL — `No module named 'vanish.io'`.

- [ ] **Step 3: Implement `io.py`**

```python
# vanish/io.py
"""Pillow-backed image and mask I/O. Normalizes everything the core sees."""
import os
import numpy as np
from PIL import Image


def load_image(path):
    """Load an image as a uint8 (H,W,3) RGB array."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Image not found: {path}")
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"), dtype=np.uint8).copy()


def save_image(path, img):
    """Save a uint8 (H,W,3) array as an image."""
    arr = np.clip(img, 0, 255).astype(np.uint8)
    Image.fromarray(arr, mode="RGB").save(path)


def load_mask(path, shape):
    """Load a mask PNG as a boolean (H,W) array (True where pixel > 127).
    `shape` is the expected (H, W) of the target image."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Mask not found: {path}")
    with Image.open(path) as im:
        mask = np.asarray(im.convert("L"))
    if mask.shape != tuple(shape):
        raise ValueError(
            f"Mask shape {mask.shape} does not match image shape {tuple(shape)}")
    return mask > 127
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_io.py -q`
Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add vanish/io.py tests/test_io.py
git commit -m "feat: Pillow I/O with RGB normalization and mask loading"
```

---

## Task 7: `Carver` driver

**Files:**
- Create: `vanish/carver.py`
- Test: `tests/test_carver.py`

- [ ] **Step 1: Write the failing tests**

```python
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
    assert np.array_equal(img, original)   # input untouched
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_carver.py -q`
Expected: FAIL — `No module named 'vanish.carver'`.

- [ ] **Step 3: Implement `carver.py`**

```python
# vanish/carver.py
"""Stateful driver over the pure core: find -> remove -> repeat, with optional
per-step frame streaming for GIF/overlay demos."""
from vanish import core


class Carver:
    def __init__(self, img):
        self.img = img.copy()

    def iter_removals(self, n):
        """Yield (current_image, removed_seam) after each of n removals."""
        for _ in range(n):
            seam = core.find_vertical_seam(core.energy_map(self.img))
            self.img = core.remove_seam(self.img, seam)
            yield self.img, seam

    def remove(self, n):
        """Remove n vertical seams and return the final image."""
        for _ in self.iter_removals(n):
            pass
        return self.img
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_carver.py -q`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add vanish/carver.py tests/test_carver.py
git commit -m "feat: Carver driver with frame streaming"
```

---

## Task 8: `features.resize` (shrink + enlarge + transpose for height)

**Files:**
- Create: `vanish/features.py`
- Test: `tests/test_features.py`

- [ ] **Step 1: Write the failing tests**

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_features.py -q`
Expected: FAIL — `No module named 'vanish.features'`.

- [ ] **Step 3: Implement `features.py` (resize + enlarge)**

```python
# vanish/features.py
"""User-facing operations composed from the pure core."""
import numpy as np
from vanish import core
from vanish.carver import Carver


def _enlarge_width(img, num):
    """Insert `num` seams. Seams are computed on the original up front, then
    inserted with index shifting so duplicates don't stack on one seam (which
    would smear)."""
    if num == 0:
        return img.copy()
    seams = [s.copy() for s in core.compute_seams(img, num)]
    out = img.copy()
    for i in range(len(seams)):
        seam = seams[i]
        out = core.insert_seam(out, seam)
        for j in range(i + 1, len(seams)):
            seams[j][seams[j] >= seam] += 2   # +2 keeps later seams clear of the pair
    return out


def _resize_width(img, target):
    w = img.shape[1]
    if target < w:
        return Carver(img).remove(w - target)
    if target > w:
        return _enlarge_width(img, target - w)
    return img.copy()


def resize(img, width=None, height=None):
    """Content-aware resize to an absolute target width and/or height."""
    out = img
    if width is not None:
        out = _resize_width(out, width)
    if height is not None:
        out = _resize_width(out.transpose(1, 0, 2), height).transpose(1, 0, 2)
    return out


def enlarge(img, dwidth=0, dheight=0):
    """Grow width/height by the given deltas."""
    out = img
    if dwidth:
        out = _enlarge_width(out, dwidth)
    if dheight:
        out = _enlarge_width(out.transpose(1, 0, 2), dheight).transpose(1, 0, 2)
    return out
```

Note: `resize` returns `img` itself only via the width==target `.copy()` path inside `_resize_width`; the top-level no-op test passes because `_resize_width` returns a copy equal to the input.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_features.py -q`
Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add vanish/features.py tests/test_features.py
git commit -m "feat: resize and enlarge with transpose-based height handling"
```

---

## Task 9: `features.remove_object`

**Files:**
- Modify: `vanish/features.py`
- Test: `tests/test_features.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_features.py`:

```python
def test_remove_object_erases_masked_blob():
    # flat gray background with a bright vertical stripe (the "object")
    img = np.full((12, 16, 3), 100, dtype=np.uint8)
    img[:, 7:9, :] = 250
    mask = np.zeros((12, 16), dtype=bool)
    mask[:, 7:9] = True
    out = features.remove_object(img, mask, shrink=True)
    # the two stripe columns are gone; result is narrower and has no bright pixels
    assert out.shape == (12, 14, 3)
    assert out.max() < 200


def test_remove_object_restores_width_when_not_shrinking():
    img = np.full((12, 16, 3), 100, dtype=np.uint8)
    img[:, 7:9, :] = 250
    mask = np.zeros((12, 16), dtype=bool)
    mask[:, 7:9] = True
    out = features.remove_object(img, mask, shrink=False)
    assert out.shape == (12, 16, 3)   # width restored via insertion
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_features.py -k remove_object -q`
Expected: FAIL — `has no attribute 'remove_object'`.

- [ ] **Step 3: Implement `remove_object`**

Append to `vanish/features.py`:

```python
def remove_object(img, mask, shrink=False):
    """Remove the masked region by biasing its energy strongly negative so
    seams route through it. Removes vertical seams until the mask is empty.
    If shrink is False, re-inserts the removed columns to restore width."""
    out = img.copy()
    m = mask.copy()
    removed = 0
    while m.any():
        energy = core.energy_map(out)
        energy[m] = -1e9                      # force the seam through the mask
        seam = core.find_vertical_seam(energy)
        out = core.remove_seam(out, seam)
        m = core.remove_seam(m, seam)
        removed += 1
    if not shrink and removed:
        out = _enlarge_width(out, removed)
    return out
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_features.py -k remove_object -q`
Expected: 2 passed.

- [ ] **Step 5: Run full feature suite**

Run: `pytest tests/test_features.py -q`
Expected: 8 passed.

- [ ] **Step 6: Commit**

```bash
git add vanish/features.py tests/test_features.py
git commit -m "feat: object removal via negative-energy mask routing"
```

---

## Task 10: Public API re-exports

**Files:**
- Modify: `vanish/__init__.py`
- Test: `tests/test_api.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_api.py
import numpy as np
import vanish


def test_public_api_exposed():
    img = (np.random.default_rng(0).random((8, 10, 3)) * 255).astype(np.uint8)
    assert vanish.resize(img, width=8).shape == (8, 8, 3)
    assert vanish.enlarge(img, dwidth=2).shape == (8, 12, 3)
    assert hasattr(vanish, "remove_object")
    assert hasattr(vanish, "Carver")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_api.py -q`
Expected: FAIL — `module 'vanish' has no attribute 'resize'`.

- [ ] **Step 3: Implement re-exports**

```python
# vanish/__init__.py
"""Vanish-py: content-aware image resizing via seam carving."""
from vanish.features import resize, enlarge, remove_object
from vanish.carver import Carver

__all__ = ["resize", "enlarge", "remove_object", "Carver"]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_api.py -q`
Expected: 1 passed.

- [ ] **Step 5: Commit**

```bash
git add vanish/__init__.py tests/test_api.py
git commit -m "feat: expose public API from vanish package root"
```

---

## Task 11: CLI

**Files:**
- Create: `vanish/cli.py`
- Create: `vanish/__main__.py`
- Test: `tests/test_cli.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_cli.py
import numpy as np
import pytest
from PIL import Image
from vanish import cli, io


def _write_img(path, h, w):
    img = (np.random.default_rng(0).random((h, w, 3)) * 255).astype(np.uint8)
    io.save_image(str(path), img)


def test_cli_shrink_width(tmp_path):
    src, dst = tmp_path / "in.png", tmp_path / "out.png"
    _write_img(src, 8, 12)
    cli.main(["shrink", str(src), str(dst), "--width", "9"])
    assert io.load_image(str(dst)).shape == (8, 9, 3)


def test_cli_enlarge_dw(tmp_path):
    src, dst = tmp_path / "in.png", tmp_path / "out.png"
    _write_img(src, 8, 10)
    cli.main(["enlarge", str(src), str(dst), "--dw", "3"])
    assert io.load_image(str(dst)).shape == (8, 13, 3)


def test_cli_remove_with_mask(tmp_path):
    src, dst, mpath = tmp_path / "in.png", tmp_path / "out.png", tmp_path / "m.png"
    img = np.full((12, 16, 3), 100, dtype=np.uint8)
    img[:, 7:9, :] = 250
    io.save_image(str(src), img)
    m = np.zeros((12, 16), dtype=np.uint8)
    m[:, 7:9] = 255
    Image.fromarray(m, mode="L").save(mpath)
    cli.main(["remove", str(src), str(dst), "--mask", str(mpath), "--shrink"])
    assert io.load_image(str(dst)).shape == (12, 14, 3)


def test_cli_shrink_rejects_width_not_smaller(tmp_path):
    src, dst = tmp_path / "in.png", tmp_path / "out.png"
    _write_img(src, 8, 12)
    with pytest.raises(SystemExit):
        cli.main(["shrink", str(src), str(dst), "--width", "20"])
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_cli.py -q`
Expected: FAIL — `No module named 'vanish.cli'`.

- [ ] **Step 3: Implement `cli.py`**

```python
# vanish/cli.py
"""Argparse CLI. Parses args, calls io + features, writes output. No algorithm
logic lives here."""
import argparse
import sys
from vanish import io, features


def _target(current, absolute, delta):
    """Resolve an absolute target or a delta into an absolute dimension."""
    if absolute is not None:
        return absolute
    if delta is not None:
        return current + delta
    return None


def _build_parser():
    p = argparse.ArgumentParser(prog="vanish", description="Seam-carving resizer")
    sub = p.add_subparsers(dest="cmd", required=True)

    for name in ("shrink", "enlarge"):
        sp = sub.add_parser(name)
        sp.add_argument("input")
        sp.add_argument("output")
        sp.add_argument("--width", type=int)
        sp.add_argument("--height", type=int)
        sp.add_argument("--dw", type=int, help="delta width (+enlarge/-shrink)")
        sp.add_argument("--dh", type=int, help="delta height")

    rm = sub.add_parser("remove")
    rm.add_argument("input")
    rm.add_argument("output")
    rm.add_argument("--mask", required=True)
    rm.add_argument("--shrink", action="store_true")

    en = sub.add_parser("energy")
    en.add_argument("input")
    en.add_argument("output")

    sm = sub.add_parser("seams")
    sm.add_argument("input")
    sm.add_argument("output")
    sm.add_argument("--count", type=int, required=True)
    return p


def main(argv=None):
    args = _build_parser().parse_args(argv)
    img = io.load_image(args.input)
    h, w = img.shape[:2]

    if args.cmd in ("shrink", "enlarge"):
        tw = _target(w, args.width, args.dw)
        th = _target(h, args.height, args.dh)
        if tw is None and th is None:
            sys.exit("Specify --width/--height or --dw/--dh")
        if args.cmd == "shrink":
            if tw is not None and tw >= w:
                sys.exit(f"shrink width must be < {w}")
            if th is not None and th >= h:
                sys.exit(f"shrink height must be < {h}")
        if args.cmd == "enlarge":
            if tw is not None and tw <= w:
                sys.exit(f"enlarge width must be > {w}")
            if th is not None and th <= h:
                sys.exit(f"enlarge height must be > {h}")
        out = features.resize(img, width=tw, height=th)

    elif args.cmd == "remove":
        mask = io.load_mask(args.mask, (h, w))
        out = features.remove_object(img, mask, shrink=args.shrink)

    elif args.cmd == "energy":
        from vanish import core
        import numpy as np
        e = core.energy_map(img)
        e = (255 * e / e.max()).astype("uint8") if e.max() > 0 else e.astype("uint8")
        out = np.stack([e, e, e], axis=-1)

    elif args.cmd == "seams":
        out = _overlay_seams(img, args.count)

    io.save_image(args.output, out)


def _overlay_seams(img, count):
    """Draw `count` lowest-energy seams in red over a copy of the image."""
    from vanish import core
    out = img.copy()
    for seam in core.compute_seams(img, count):
        out[range(out.shape[0]), seam] = [255, 0, 0]
    return out
```

- [ ] **Step 4: Create `__main__.py`**

```python
# vanish/__main__.py
from vanish.cli import main

if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_cli.py -q`
Expected: 4 passed.

- [ ] **Step 6: Smoke-test the CLI entry point**

Run: `python -m vanish --help`
Expected: usage text listing subcommands `shrink`, `enlarge`, `remove`, `energy`, `seams`.

- [ ] **Step 7: Commit**

```bash
git add vanish/cli.py vanish/__main__.py tests/test_cli.py
git commit -m "feat: argparse CLI with shrink/enlarge/remove/energy/seams"
```

---

## Task 12: Naive reference + benchmark

**Files:**
- Create: `vanish/_naive.py`
- Create: `examples/benchmark.py`
- Test: `tests/test_naive.py`

- [ ] **Step 1: Write the failing test (naive matches vectorized)**

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_naive.py -q`
Expected: FAIL — `No module named 'vanish._naive'`.

- [ ] **Step 3: Implement `_naive.py`**

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_naive.py -q`
Expected: 2 passed.

- [ ] **Step 5: Write `examples/benchmark.py`**

```python
# examples/benchmark.py
"""Benchmark naive per-pixel loops vs the vectorized NumPy pipeline."""
import time
import numpy as np
from vanish import core, _naive


def _bench(fn, *args, repeat=1):
    start = time.perf_counter()
    for _ in range(repeat):
        fn(*args)
    return (time.perf_counter() - start) / repeat


def main():
    rng = np.random.default_rng(0)
    img = (rng.random((200, 300, 3)) * 255).astype(np.uint8)
    energy = core.energy_map(img)

    t_naive_e = _bench(_naive.energy_map_naive, img)
    t_vec_e = _bench(core.energy_map, img, repeat=10)
    t_naive_dp = _bench(_naive.cumulative_energy_naive, energy)
    t_vec_dp = _bench(lambda e: core.cumulative_energy(e), energy, repeat=10)

    print(f"energy map:  naive {t_naive_e:8.4f}s  vec {t_vec_e:8.4f}s  "
          f"speedup {t_naive_e / t_vec_e:6.1f}x")
    print(f"DP table:    naive {t_naive_dp:8.4f}s  vec {t_vec_dp:8.4f}s  "
          f"speedup {t_naive_dp / t_vec_dp:6.1f}x")


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Run the benchmark**

Run: `python examples/benchmark.py`
Expected: two lines printing naive/vec times and a speedup > 1x for each. Record the numbers for the README.

- [ ] **Step 7: Commit**

```bash
git add vanish/_naive.py examples/benchmark.py tests/test_naive.py
git commit -m "feat: naive reference and naive-vs-vectorized benchmark"
```

---

## Task 13: Demo scripts + bundled asset

**Files:**
- Create: `examples/assets/` (add one demo image `beach.jpg` and, for the removal demo, `beach_mask.png`)
- Create: `examples/make_hero_gif.py`
- Create: `examples/make_comparison.py`
- Create: `examples/make_removal_demo.py`
- Create: `examples/make_seam_overlay.py`

- [ ] **Step 1: Add a demo asset**

Place a landscape photo with a clear subject and a large low-detail background at `examples/assets/beach.jpg` (e.g. a wide beach with one or two people). Paint a white-on-black mask over one subject and save as `examples/assets/beach_mask.png` (same H×W as the photo). If no photo is handy, generate a synthetic stand-in:

```python
# examples/assets/_make_synthetic.py  (optional, only if no real photo)
import numpy as np
from vanish import io
h, w = 300, 500
img = np.zeros((h, w, 3), dtype=np.uint8)
img[:, :, 2] = np.linspace(60, 200, w).astype(np.uint8)      # blue gradient sky/sea
img[200:, :, :] = [210, 200, 150]                            # sandy foreground
img[120:210, 120:150] = [230, 60, 60]                        # "subject" A
img[140:210, 360:385] = [60, 200, 80]                        # "subject" B
io.save_image("examples/assets/beach.jpg", img)
mask = np.zeros((h, w), dtype=np.uint8)
mask[120:210, 120:150] = 255
from PIL import Image; Image.fromarray(mask, "L").save("examples/assets/beach_mask.png")
```

Run (only if using the synthetic fallback): `python examples/assets/_make_synthetic.py`
Expected: `beach.jpg` and `beach_mask.png` created under `examples/assets/`.

- [ ] **Step 2: Write `examples/make_hero_gif.py`**

```python
# examples/make_hero_gif.py
"""Hero demo: an image continuously narrowing with subjects intact."""
import imageio.v2 as imageio
import numpy as np
from vanish import io
from vanish.carver import Carver

SRC = "examples/assets/beach.jpg"
OUT = "examples/assets/hero.gif"


def main():
    img = io.load_image(SRC)
    frames = [img]
    carver = Carver(img)
    for frame, _seam in carver.iter_removals(img.shape[1] // 2):
        # pad back to original width so the GIF canvas is stable
        pad = np.zeros((frame.shape[0], img.shape[1] - frame.shape[1], 3), np.uint8)
        frames.append(np.concatenate([frame, pad], axis=1))
    imageio.mimsave(OUT, frames[::2], duration=0.05)   # every other frame
    print(f"wrote {OUT} ({len(frames)} frames)")


if __name__ == "__main__":
    main()
```

Run: `python examples/make_hero_gif.py`
Expected: `examples/assets/hero.gif` written; opening it shows the image narrowing with subjects holding shape.

- [ ] **Step 3: Write `examples/make_comparison.py`**

```python
# examples/make_comparison.py
"""Side-by-side: seam carving vs naive resize at the same target width."""
import numpy as np
from PIL import Image
from vanish import io, features

SRC = "examples/assets/beach.jpg"
OUT = "examples/assets/comparison.png"


def main():
    img = io.load_image(SRC)
    target = img.shape[1] // 2
    carved = features.resize(img, width=target)
    naive = np.asarray(
        Image.fromarray(img).resize((target, img.shape[0]), Image.LANCZOS))
    stacked = np.concatenate([carved, naive], axis=0)   # carved on top
    io.save_image(OUT, stacked)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
```

Run: `python examples/make_comparison.py`
Expected: `comparison.png` — top (seam-carved) keeps subject proportions; bottom (naive) squishes them.

- [ ] **Step 4: Write `examples/make_removal_demo.py`**

```python
# examples/make_removal_demo.py
"""Before/after object removal."""
import numpy as np
from vanish import io, features

SRC = "examples/assets/beach.jpg"
MASK = "examples/assets/beach_mask.png"
OUT = "examples/assets/removal.png"


def main():
    img = io.load_image(SRC)
    mask = io.load_mask(MASK, img.shape[:2])
    removed = features.remove_object(img, mask, shrink=False)  # keep width
    stacked = np.concatenate([img, removed], axis=0)
    io.save_image(OUT, stacked)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
```

Run: `python examples/make_removal_demo.py`
Expected: `removal.png` — top has the masked subject, bottom does not, same dimensions.

- [ ] **Step 5: Write `examples/make_seam_overlay.py`**

```python
# examples/make_seam_overlay.py
"""Draw the lowest-energy seams in red for explanation/debugging."""
from vanish import io
from vanish.cli import _overlay_seams

SRC = "examples/assets/beach.jpg"
OUT = "examples/assets/seams.png"


def main():
    img = io.load_image(SRC)
    io.save_image(OUT, _overlay_seams(img, 40))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
```

Run: `python examples/make_seam_overlay.py`
Expected: `seams.png` — red seams cluster in low-energy background, avoiding subjects.

- [ ] **Step 6: Commit**

```bash
git add examples/
git commit -m "feat: hero GIF, comparison, removal, and seam-overlay demos"
```

---

## Task 14: README

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Write the README**

Replace `README.md` with content covering, in this order:
1. One-line pitch + the hero GIF (`examples/assets/hero.gif`).
2. What seam carving is (2-3 sentences) and the energy → DP → remove/repeat flow.
3. The comparison image and the object-removal before/after.
4. Install (`pip install -e .`) and CLI usage for all five subcommands (copy the exact invocations from `tests/test_cli.py`).
5. Library usage (`import vanish; vanish.resize(...)`, `vanish.remove_object(...)`).
6. The DP recurrence explained, and why greedy row-by-row fails (it can't see that a locally-cheap step forces an expensive one later — only the DP over all paths is optimal).
7. Performance: paste the actual speedup numbers printed by `examples/benchmark.py`, and note that vectorization is across columns within a row while the row loop stays sequential.
8. The seam-overlay image.
9. Limitations / non-goals: no GUI, no forward-energy variant, no video, no GPU, full energy recompute each removal (local recompute is a possible optimization).

- [ ] **Step 2: Verify all referenced asset files exist**

Run: `ls examples/assets/`
Expected: `beach.jpg`, `beach_mask.png`, `hero.gif`, `comparison.png`, `removal.png`, `seams.png`.

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "docs: README with demos, algorithm explanation, and benchmark"
```

---

## Final verification

- [ ] **Step 1: Run the entire test suite**

Run: `pytest -q`
Expected: all tests pass across `test_core`, `test_io`, `test_carver`, `test_features`, `test_api`, `test_cli`, `test_naive`.

- [ ] **Step 2: Full CLI round-trip smoke test**

Run:
```bash
python -m vanish shrink examples/assets/beach.jpg /tmp/s.png --dw -80
python -m vanish enlarge examples/assets/beach.jpg /tmp/e.png --dw 80
python -m vanish energy examples/assets/beach.jpg /tmp/en.png
python -m vanish seams examples/assets/beach.jpg /tmp/sm.png --count 30
python -m vanish remove examples/assets/beach.jpg /tmp/rm.png --mask examples/assets/beach_mask.png --shrink
```
Expected: five output files written with no errors.

---

## Self-Review Notes (checked against the spec)

- **Spec coverage:** energy map (T1), DP + backtrack (T2), seam find (T3), remove (T4), insert + multi-seam (T5), I/O + mask + RGB/RGBA normalization (T6), Carver streaming (T7), resize/enlarge/transpose (T8), object removal via negative-energy mask (T9), library API (T10), CLI all five subcommands + boundary validation (T11), naive reference + benchmark (T12), all four demos (T13), README with algorithm explanation + non-goals (T14). No spec requirement left without a task.
- **Type consistency:** images `(H,W,3)` uint8; seams `(H,)` int64; energy `(H,W)` float64; `backtrack` stores parent-column offsets; `compute_seams` returns original-coordinate column arrays used by both `_enlarge_width` and `_overlay_seams`; `remove_seam` handles both 3D images and 2D index/mask arrays (relied on by `compute_seams` and `remove_object`).
- **No placeholders:** every code step contains complete, runnable code; the only optional branch is the synthetic-image fallback in T13, fully specified.
