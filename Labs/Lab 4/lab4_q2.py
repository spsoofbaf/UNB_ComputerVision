import numpy as np
import matplotlib.pyplot as plt

from skimage import filters
from skimage.data import astronaut
from skimage.color import rgb2gray

image = rgb2gray(astronaut())

edge_sobel = filters.sobel(image)
g_x = filters.sobel_h(image)
g_y = filters.sobel_v(image)

columns = image.shape[0]
rows = image.shape[1]

magnitude = rgb2gray(astronaut())
phase = rgb2gray(astronaut())


def calculate_magnitude(x, y):
    return abs(g_x[x, y]) + abs(g_y[x, y])


def calculate_phase(x, y):
    return np.arctan2(g_y[x, y], g_x[x, y])


for x in range(columns):
    for y in range(rows):
        magnitude[x, y] = calculate_magnitude(x, y)
        phase[x, y] = calculate_phase(x, y)

fig, axes = plt.subplots(ncols=3, nrows=2, sharex=True, sharey=True,
                         figsize=(8, 4))

axes[0][0].imshow(image, cmap=plt.cm.gray)
axes[0][0].set_title('Original Image')
axes[0][0].axis('off')

axes[0][1].imshow(g_x, cmap=plt.cm.gray)
axes[0][1].set_title('Sobel-X')
axes[0][1].axis('off')

axes[0][2].imshow(magnitude, cmap=plt.cm.gray)
axes[0][2].set_title('Magnitude')
axes[0][2].axis('off')

axes[1][0].imshow(edge_sobel, cmap=plt.cm.gray)
axes[1][0].set_title('Sobel Edge Detection')
axes[1][0].axis('off')

axes[1][1].imshow(g_y, cmap=plt.cm.gray)
axes[1][1].set_title('Sobel-Y')
axes[1][1].axis('off')

axes[1][2].imshow(phase, cmap=plt.cm.gray)
axes[1][2].set_title('Phase')
axes[1][2].axis('off')

plt.tight_layout()
plt.show()
