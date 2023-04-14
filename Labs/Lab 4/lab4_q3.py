from math import sqrt
from skimage import data
from skimage.feature import blob_dog, blob_log, blob_doh
from skimage.color import rgb2gray
import timeit

import matplotlib.pyplot as plt

image = data.hubble_deep_field()[0:500, 0:500]
image_gray = rgb2gray(image)

start = timeit.default_timer()
blobs_log = blob_log(image_gray, max_sigma=30, num_sigma=10, threshold=.1)
blobs_log[:, 2] = blobs_log[:, 2] * sqrt(2)
end = timeit.default_timer()
print("\nexecution time of LoG = " + str(end - start) + " seconds")

start = timeit.default_timer()
blobs_dog = blob_dog(image_gray, max_sigma=30, threshold=.1)
blobs_dog[:, 2] = blobs_dog[:, 2] * sqrt(2)
end = timeit.default_timer()
print("\nexecution time of DoG = " + str(end - start) + " seconds")

start = timeit.default_timer()
blobs_doh = blob_doh(image_gray, max_sigma=30, threshold=.01)
blobs_list = [blobs_log, blobs_dog, blobs_doh]
end = timeit.default_timer()
print("\nexecution time of DoH = " + str(end - start) + " seconds")

colors = ['yellow', 'lime', 'red']
titles = ['Laplacian of Gaussian', 'Difference of Gaussian',
          'Determinant of Hessian']
sequence = zip(blobs_list, colors, titles)

fig, axes = plt.subplots(1, 3, figsize=(9, 3), sharex=True, sharey=True)
ax = axes.ravel()

for idx, (blobs, color, title) in enumerate(sequence):
    ax[idx].set_title(title)
    ax[idx].imshow(image)
    for blob in blobs:
        y, x, r = blob
        c = plt.Circle((x, y), r, color=color, linewidth=2, fill=False)
        ax[idx].add_patch(c)
    ax[idx].set_axis_off()

plt.tight_layout()
plt.show()

# Laplacian of Gaussian (LoG)
# This is the most accurate and slowest approach. This is because the LoG approach is a multi-scale method that applies a Gaussian smoothing filter to the image before computing the Laplacian.
# It computes the Laplacian of Gaussian images with successively increasing standard deviation and stacks them up in a cube.
# Blobs are local maximas in this cube.
# Detecting larger blobs is especially slower because of larger kernel sizes during convolution.
# Only bright blobs on dark backgrounds are detected

# Difference of Gaussian (DoG)
# This is a faster approximation of LoG approach.
# In this case the image is blurred with increasing standard deviations and the difference between two successively blurred images are stacked up in a cube.
# This method suffers from the same disadvantage as LoG approach for detecting larger blobs.
# Blobs are again assumed to be bright on dark

# Determinant of Hessian (DoH)
# This is the fastest approach.
# It detects blobs by finding maximas in the matrix of the Determinant of Hessian of the image.
# The detection speed is independent of the size of blobs as internally the implementation uses box filters instead of convolutions.
# Bright on dark as well as dark on bright blobs are detected.
# The downside is that small blobs (<3px) are not detected accurately.

# b) The DoH is the fastest method and has the least execution time.
# This is because the DoH method can be efficiently implemented using the box filter, which has a linear time complexity, whereas the LoG and DoG methods require Gaussian filtering, which has a higher time complexity.