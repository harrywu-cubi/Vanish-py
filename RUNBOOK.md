# Vanish-py Runbook

A step-by-step guide to set up, run, test, and use this project from a fresh clone.
No prior knowledge of the codebase is assumed.

---

## 1. Prerequisites

- **Python 3.11 or newer** (developed and tested on 3.14). Check with:
  ```bash
  python --version
  ```
  If `python` isn't found, try `python3`. On Windows, `py -3` also works.
- **git** (to clone), and **pip** (ships with Python).

That's it — all other dependencies (NumPy, Pillow, imageio, pytest) install automatically
in the next step.

---

## 2. Get the code and set up an isolated environment

```bash
git clone <repository-url> Vanish-py
cd Vanish-py

# Create and activate a virtual environment (keeps deps isolated from your system)
python -m venv .venv

# Activate it:
#   macOS / Linux:
source .venv/bin/activate
#   Windows (Git Bash):
source .venv/Scripts/activate
#   Windows (PowerShell):
#   .venv\Scripts\Activate.ps1
```

Your shell prompt should now show `(.venv)`.

---

## 3. Install

Install the package in editable mode with its dev + example dependencies:

```bash
pip install -e ".[dev]"
```

This installs `vanish` plus NumPy, Pillow, imageio, and pytest. You should see a line
ending in `Successfully installed ... vanish-0.1.0`.

---

## 4. Verify it works (run the tests)

```bash
python -m pytest -q
```

Expected: **`49 passed`**. If every test passes, the install is healthy.

---

## 5. Generate the demo image (optional but recommended)

The repo ships with pre-generated demo assets under `examples/assets/`. To (re)create the
synthetic demo image and its object-removal mask from scratch:

```bash
python examples/assets/_make_synthetic.py
# -> wrote beach.jpg and beach_mask.png
```

`beach.jpg` is a synthetic scene: two colored "subjects" on a gradient sky over a sandy
foreground — chosen because the effect is most visible on a clear subject + large
low-detail background. Swap in any real photo of the same nature to try it on a real image.

---

## 6. Use the command-line tool

All commands take an input image and an output path. Run against the bundled demo image:

```bash
# Shrink to an absolute target width (content-aware):
python -m vanish shrink examples/assets/beach.jpg out_shrink.png --width 400

# Shrink by a delta instead (negative = narrower). --dh does the same for height:
python -m vanish shrink examples/assets/beach.jpg out_shrink.png --dw -100

# Enlarge (seam insertion):
python -m vanish enlarge examples/assets/beach.jpg out_enlarge.png --dw 100

# Erase a masked object (white pixels in the mask = remove).
#   --shrink narrows the image to fill the gap; omit it to keep the original width.
python -m vanish remove examples/assets/beach.jpg out_removed.png \
    --mask examples/assets/beach_mask.png --shrink

# Visualize the energy map (bright = important):
python -m vanish energy examples/assets/beach.jpg out_energy.png

# Overlay the 40 lowest-energy seams in red:
python -m vanish seams examples/assets/beach.jpg out_seams.png --count 40
```

See all options:

```bash
python -m vanish --help
python -m vanish shrink --help
```

**Notes / guardrails:**
- `--width`/`--height` are absolute targets; `--dw`/`--dh` are deltas from the current size.
- `shrink` rejects a target that isn't smaller; `enlarge` rejects one that isn't larger.
- `seams --count` must be between 1 and (image width − 1).
- A `remove` mask must have the same height and width as the input image.

---

## 7. Use it as a library

```python
import vanish
from vanish import io

img = io.load_image("examples/assets/beach.jpg")      # -> (H, W, 3) uint8 array

narrow = vanish.resize(img, width=400)                # content-aware shrink
tall   = vanish.resize(img, height=250, width=400)    # both dimensions at once
wide   = vanish.enlarge(img, dwidth=100)              # seam insertion

mask   = io.load_mask("examples/assets/beach_mask.png", img.shape[:2])
erased = vanish.remove_object(img, mask, shrink=False)  # subject vanishes, width kept

io.save_image("result.png", narrow)
```

---

## 8. Regenerate the demo assets

```bash
python examples/make_hero_gif.py       # -> examples/assets/hero.gif (narrowing animation)
python examples/make_comparison.py     # -> comparison.png (seam carve vs. naive resize)
python examples/make_removal_demo.py   # -> removal.png (before/after object removal)
python examples/make_seam_overlay.py   # -> seams.png (seams drawn in red)
```

---

## 9. Reproduce the performance benchmark

```bash
python examples/benchmark.py
```

Prints the speedup of the vectorized NumPy pipeline over a naive per-pixel implementation,
for both the energy map and the DP table. Numbers vary by machine.

---

## 10. Troubleshooting

| Symptom | Fix |
|---|---|
| `python: command not found` | Use `python3` (or `py -3` on Windows). |
| `No module named vanish` | Activate the venv and re-run `pip install -e ".[dev]"`. |
| `pytest: command not found` | Run it as `python -m pytest` (used throughout this runbook). |
| CLI writes nothing / "Image not found" on read-back | Make sure the output directory exists and you're reading back the exact path you wrote. |
| Mask error on `remove` | The mask PNG must match the image's height × width exactly. |

---

## Project layout (orientation)

```
vanish/
  core.py       pure NumPy algorithm: energy map, DP table, seam find/remove/insert
  carver.py     stateful driver that streams seam removals
  features.py   resize / enlarge / remove_object (composed from core)
  io.py         image + mask loading/saving (Pillow)
  cli.py        the `python -m vanish` command-line interface
  _naive.py     slow reference implementation (benchmark + test oracle only)
examples/       demo scripts + bundled assets + benchmark
tests/          pytest suite (run with `python -m pytest`)
docs/           design spec and implementation plan
```
