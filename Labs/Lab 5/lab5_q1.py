import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from skimage import color
from skimage.feature import greycomatrix, greycoprops

# open the image
image = Image.open('desert-rocks.jpg').convert('L')
# image = color.rgb2gray(image)

# select some patches from desert areas of the image
desert_locations = [(630, 166), (761, 173), (754, 513), (732, 918)]
desert_patches = []
patch_size = 21
for loc in desert_locations:
    patch = np.array(image.crop((loc[1], loc[0], loc[1] + patch_size, loc[0] + patch_size)))
    desert_patches.append(patch)

# select some patches from rock areas of the image
rock_locations = [(486, 217), (415, 190), (501, 531), (248, 1054)]
rock_patches = []
for loc in rock_locations:
    patch = np.array(image.crop((loc[1], loc[0], loc[1] + patch_size, loc[0] + patch_size)))
    rock_patches.append(patch)

# compute GLCM properties for each patch
xs = []
ys = []
colors = []
for patch in (desert_patches + rock_patches):
    glcm = greycomatrix(patch, distances=[5], angles=[0], levels=256,
                        symmetric=True, normed=True)
    contrast = greycoprops(glcm, 'contrast')[0, 0]
    homogeneity = greycoprops(glcm, 'homogeneity')[0, 0]
    energy = greycoprops(glcm, 'energy')[0, 0]
    xs.append(contrast)
    ys.append(homogeneity)
    colors.append(energy)

# create the figure
fig = plt.figure(figsize=(8, 8))

# display original image with locations of patches
ax = fig.add_subplot(3, 2, 1)
ax.imshow(image, cmap=plt.cm.gray,
          vmin=0, vmax=255)
for (y, x) in desert_locations:
    ax.plot(x + patch_size / 2, y + patch_size / 2, 'gs')
for (y, x) in rock_locations:
    ax.plot(x + patch_size / 2, y + patch_size / 2, 'bs')
ax.set_xlabel('Original Image')
ax.set_xticks([])
ax.set_yticks([])
ax.axis('image')

# for each patch, plot (dissimilarity, correlation)
ax = fig.add_subplot(3, 2, 2)
ax.plot(xs[:len(desert_patches)], ys[:len(desert_patches)], 'go',
        label='Desert')
ax.plot(xs[len(desert_patches):], ys[len(desert_patches):], 'bo',
        label='Rock')
ax.set_xlabel('GLCM Dissimilarity')
ax.set_ylabel('GLCM Correlation')
ax.legend()

# display the image patches
for i, patch in enumerate(desert_patches):
    ax = fig.add_subplot(3, len(desert_patches), len(desert_patches) * 1 + i + 1)
    ax.imshow(patch, cmap=plt.cm.gray,
              vmin=0, vmax=255)
    ax.set_xlabel(f"Desert {i + 1}")

for i, patch in enumerate(rock_patches):
    ax = fig.add_subplot(3, len(rock_patches), len(rock_patches) * 2 + i + 1)
    ax.imshow(patch, cmap=plt.cm.gray,
              vmin=0, vmax=255)
    ax.set_xlabel(f"Rock {i + 1}")

# display the patches and plot
fig.suptitle('Grey level co-occurrence matrix features', fontsize=14, y=1.05)
plt.tight_layout()
plt.show()
