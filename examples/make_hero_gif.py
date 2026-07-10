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
        pad = np.zeros((frame.shape[0], img.shape[1] - frame.shape[1], 3), np.uint8)
        frames.append(np.concatenate([frame, pad], axis=1))
    imageio.mimsave(OUT, frames[::2], duration=0.05)
    print(f"wrote {OUT} ({len(frames)} frames)")


if __name__ == "__main__":
    main()
