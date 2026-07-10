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
