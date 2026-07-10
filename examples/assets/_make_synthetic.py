# examples/assets/_make_synthetic.py
"""Generate a synthetic demo image (clear subjects + large low-detail
background) plus a mask over one subject, for the demo scripts."""
import numpy as np
from PIL import Image
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
Image.fromarray(mask, "L").save("examples/assets/beach_mask.png")
print("wrote beach.jpg and beach_mask.png")
