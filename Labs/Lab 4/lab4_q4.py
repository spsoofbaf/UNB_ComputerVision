from matplotlib import pyplot as plt

from skimage import data
from skimage.feature import corner_harris, corner_subpix, corner_peaks, corner_shi_tomasi

image = data.camera()

coords = corner_peaks(corner_harris(image), min_distance=5, threshold_rel=0.02)
coords_subpix = corner_subpix(image, coords, window_size=13)

coords2 = corner_peaks(corner_shi_tomasi(image), min_distance=5, threshold_rel=0.02)
coords_subpix2 = corner_subpix(image, coords2, window_size=13)

fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(10, 5))

ax1.imshow(image, cmap=plt.cm.gray)
ax1.plot(coords[:, 1], coords[:, 0], color='cyan', marker='o', linestyle='None', markersize=6)
ax1.plot(coords_subpix[:, 1], coords_subpix[:, 0], '+r', markersize=15)

ax2.imshow(image, cmap=plt.cm.gray)
ax2.plot(coords2[:, 1], coords2[:, 0], color='cyan', marker='o', linestyle='None', markersize=6)
ax2.plot(coords_subpix2[:, 1], coords_subpix2[:, 0], '+r', markersize=15)

plt.show()

#The Harris corner detection algorithm is a method for detecting corners in an image by analyzing variations in local intensity gradients.
# The Harris corner detector uses the eigenvalues of the second moment matrix to determine whether a point in the image corresponds to a corner, an edge or a flat area.

# The Shi-Tomasi corner detection algorithm is an improvement over the Harris corner detector, which uses the minimum eigenvalue of the second moment matrix as a corner measure.
# The Shi-Tomasi algorithm instead uses the minimum of the two eigenvalues as a measure of corner quality, which makes it less sensitive to noise than the Harris detector.
# The algorithm selects corners with the highest corner measure and discards corners with a low corner measure.

# In summary, both the Harris and Shi-Tomasi corner detection algorithms compute corner measures based on the eigenvalues of the second moment matrix, but Shi-Tomasi is an improvement over Harris as it is less sensitive to noise and can detect more accurate corners.
