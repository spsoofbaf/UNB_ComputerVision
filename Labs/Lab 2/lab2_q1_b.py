import matplotlib.pyplot as plt
import numpy as np

from skimage import data
from skimage import transform

image = data.text()
h = np.array([[1, -0.5, 100], [0.1, 0.9, 50], [0.0015, 0.0015, 1]])

tform = transform.ProjectiveTransform(matrix=h)
projected_image = transform.warp(image, tform.inverse)

fig, ax = plt.subplots(nrows=2, ncols=1)

ax[0].imshow(image, cmap="gray")
ax[0].axis('off')
ax[0].title.set_text('Original Image')

ax[1].imshow(projected_image, cmap="gray")
ax[1].axis('off')
ax[1].title.set_text('Projected Image')

plt.show()