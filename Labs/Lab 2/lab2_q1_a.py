import math
import matplotlib.pyplot as plt
from PIL import ImageDraw, Image
import numpy as np

from skimage import transform

tform = transform.SimilarityTransform(scale=0.5, rotation=3 * math.pi / 8,
                                      translation=(20, 30))

print("\nTransformation matrix = \n")
print(tform.params)

rgb_image = Image.open("suburb.jpg")
rgb_image_array = np.array(rgb_image)
rotated_image = transform.warp(rgb_image_array, tform.inverse)

tform2 = transform.SimilarityTransform(scale=0.5, rotation=3 * math.pi / 8,
                                       translation=(650, 30))
rotated_image2 = transform.warp(rgb_image_array, tform2.inverse)

fig, ax = plt.subplots(nrows=3, ncols=1)

ax[0].imshow(rgb_image, cmap="gray")
ax[0].axis('off')
ax[0].title.set_text('Original Image')

ax[1].imshow(rotated_image, cmap="gray")
ax[1].axis('off')
ax[1].title.set_text('Rotated Image (translation = (20, 30))')

ax[2].imshow(rotated_image2, cmap="gray")
ax[2].axis('off')
ax[2].title.set_text('Rotated Image with additional translation (translation = (650, 30))')
plt.show()
