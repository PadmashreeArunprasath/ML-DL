#Global pooling
import numpy as np
from PIL import Image

# ------------------------------------
# LOAD COLOR IMAGE
# ------------------------------------
img = Image.open("art.png").resize((16, 16))
img = np.array(img) / 255.0  # normalize to [0,1]

# ------------------------------------
# GLOBAL MAX POOLING
# ------------------------------------
global_max = np.max(img, axis=(0, 1))  # max over height & width for each channel


# ------------------------------------
# GLOBAL AVERAGE POOLING
# ------------------------------------
global_avg = np.mean(img, axis=(0, 1))  # average over height & width for each channel

# ------------------------------------
# DISPLAY RESULTS
# ------------------------------------
print("Global Max Pooling (RGB):", global_max)
print("Global Average Pooling (RGB):", global_avg)
