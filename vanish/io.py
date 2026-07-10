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
