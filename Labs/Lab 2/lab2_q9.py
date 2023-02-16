import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from skimage import img_as_ubyte
from skimage.filters import threshold_minimum
from skimage.morphology import opening
from skimage.morphology import disk

original_image = Image.open("blood-cells.jpg")
image = np.array(original_image.convert("L"))

thresh_min = threshold_minimum(image)
binary_min = image > thresh_min

opened = opening(binary_min, img_as_ubyte(disk(3)))

# Plotting the result
fig, ax = plt.subplots(nrows=1, ncols=2)

ax[0].imshow(original_image, cmap="gray")
ax[0].axis('off')
ax[0].title.set_text('Original Image')

ax[1].imshow(opened, cmap="gray")
ax[1].axis('off')
ax[1].title.set_text('Seperated Neutrophils')

plt.show()
