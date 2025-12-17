#Padding
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# -------------------------------------------------
# STEP 1: LOAD COLOR IMAGE (RGB)
# -------------------------------------------------
img = Image.open("art.png").resize((16, 16))
img = np.array(img) / 255.0   # Normalize to [0,1]

# -------------------------------------------------
# STEP 2: DEFINE 3x3 BLUR KERNEL
# -------------------------------------------------
kernel = np.array([
    [1, 1, 1],
    [1, 1, 1],
    [1, 1, 1]
]) / 9

kh, kw = kernel.shape

# -------------------------------------------------
# STEP 3: VALID CONVOLUTION (NO PADDING)
# -------------------------------------------------
h, w, c = img.shape
valid_output = np.zeros((h - kh + 1, w - kw + 1, c))

for ch in range(c):  # For each RGB channel
    for i in range(valid_output.shape[0]):
        for j in range(valid_output.shape[1]):
            region = img[i:i+kh, j:j+kw, ch]
            valid_output[i, j, ch] = np.sum(region * kernel)

# -------------------------------------------------
# STEP 4: FULL PADDING
# -------------------------------------------------
pad = kh - 1

# Pad RGB image
padded_img = np.pad(
    img,
    ((pad, pad), (pad, pad), (0, 0)),
    mode='constant'
)

ph, pw, _ = padded_img.shape
full_output = np.zeros((ph - kh + 1, pw - kw + 1, c))

for ch in range(c):
    for i in range(full_output.shape[0]):
        for j in range(full_output.shape[1]):
            region = padded_img[i:i+kh, j:j+kw, ch]
            full_output[i, j, ch] = np.sum(region * kernel)

# -------------------------------------------------
# STEP 5: DISPLAY RESULTS
# -------------------------------------------------
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.imshow(img)
plt.title("Original RGB Image")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(valid_output)
plt.title("VALID Convolution (No Padding)")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(full_output)
plt.title("FULL Padding Convolution")
plt.axis("off")

plt.show()