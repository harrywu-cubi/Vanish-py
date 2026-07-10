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
            # Columns equal to the inserted seam shift +2 (skip the new pair).
            # Columns strictly to the right shift +1 (one pixel was inserted).
            seams[j][seams[j] == seam] += 2   # +2 keeps later seams clear of the pair
            seams[j][seams[j] >  seam] += 1   # +1 for columns beyond the insert
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
        carved = _resize_width(out.transpose(1, 0, 2), height)
        out = np.ascontiguousarray(carved.transpose(1, 0, 2))
    return out


def enlarge(img, dwidth=0, dheight=0):
    """Grow width/height by the given deltas."""
    out = img
    if dwidth:
        out = _enlarge_width(out, dwidth)
    if dheight:
        grown = _enlarge_width(out.transpose(1, 0, 2), dheight)
        out = np.ascontiguousarray(grown.transpose(1, 0, 2))
    return out


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
