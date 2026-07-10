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
