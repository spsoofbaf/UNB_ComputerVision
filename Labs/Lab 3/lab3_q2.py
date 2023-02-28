import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from skimage.morphology import disk
from skimage.morphology import opening
from skimage.filters import gaussian
from skimage.segmentation import (morphological_geodesic_active_contour,
                                  inverse_gaussian_gradient,
                                  )
from skimage.filters import sobel
from skimage.segmentation import felzenszwalb, slic, watershed
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float

fig, axes = plt.subplots(2, 2, figsize=(8, 8))
ax = axes.flatten()

# Geodesic Active Contour
original_image = Image.open("violin-and-hand.jpg")
image = np.array(original_image.convert("L"))
init_ls = np.zeros(image.shape, dtype=np.int8)
init_ls[10:-10, 10:-10] = 1
image = opening(image, disk(2))
image = inverse_gaussian_gradient(image)

ls = morphological_geodesic_active_contour(image, num_iter=300, smoothing=4, balloon=-1, threshold=0.69,
                                           init_level_set=init_ls
                                           )

ax[0].imshow(original_image, cmap="gray")
ax[0].set_axis_off()
ax[0].contour(ls, [0.5], colors='r')
ax[0].set_title("Geodesic Active Contour", fontsize=12)

# Watershed
image2 = np.array(original_image.convert("L"))
gradient = sobel(image2)
segments_watershed = watershed(gradient, markers=120, compactness=0.001)

ax[1].imshow(mark_boundaries(img_as_float(original_image), segments_watershed), cmap="nipy_spectral")
ax[1].set_axis_off()
ax[1].set_title("Watershed", fontsize=12)

# SLIC
smoothed_image = gaussian(np.array(original_image), 1, preserve_range=False, multichannel=False)
segments_slic = slic(smoothed_image, n_segments=140, compactness=10, sigma=2,
                     start_label=1)
ax[2].imshow(mark_boundaries(img_as_float(original_image), segments_slic), cmap="nipy_spectral")
ax[2].set_axis_off()
ax[2].set_title("SLIC", fontsize=12)

# Felzenszwalb
segments_fz = felzenszwalb(smoothed_image, scale=800, sigma=0.5, min_size=200)

ax[3].imshow(mark_boundaries(img_as_float(original_image), segments_fz), cmap="nipy_spectral")
ax[3].set_axis_off()
ax[3].set_title("Felzenszwalb", fontsize=12)

fig.tight_layout()
plt.show()
