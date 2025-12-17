#image in pixels
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageFilter


# Load, resize, and blur image
img = Image.open("art.png").resize((16, 20))
img = img.filter(ImageFilter.GaussianBlur(radius=1))

# img = Image.open("art.png")

# # Step 1: downscale (controls block size)
# small = img.resize((16, 20), Image.Resampling.BILINEAR)

# # Step 2: upscale using nearest neighbor (pixelated look)
# minecraft_img = small.resize(img.size, Image.Resampling.NEAREST)

# minecraft_img.show()
# minecraft_img.save("minecraft_art.png")


# # Convert to array
# img = np.array(img)

# Display as square boxes
plt.figure(figsize=(6, 6))
plt.imshow(img, interpolation="nearest")
plt.grid(True)

plt.xticks([])
plt.yticks([])
plt.title("Image of Pixels")

plt.show()