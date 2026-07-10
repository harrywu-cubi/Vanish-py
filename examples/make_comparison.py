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
    stacked = np.concatenate([carved, naive], axis=0)
    io.save_image(OUT, stacked)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
