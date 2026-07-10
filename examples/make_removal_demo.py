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
    removed = features.remove_object(img, mask, shrink=False)
    stacked = np.concatenate([img, removed], axis=0)
    io.save_image(OUT, stacked)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
