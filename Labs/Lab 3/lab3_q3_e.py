import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from PIL import Image
from skimage import data
from skimage.metrics import (adapted_rand_error,
                              variation_of_information)
from skimage.filters import sobel
from skimage.measure import label
from skimage.util import img_as_float
from skimage.feature import canny
from skimage.morphology import remove_small_objects
from skimage.segmentation import (morphological_geodesic_active_contour,
                                  inverse_gaussian_gradient,
                                  watershed,
                                  mark_boundaries)

# image = data.coins()
original_image = Image.open("fruit_gray.PNG")
image = np.array(original_image.convert("L"))

elevation_map = sobel(image)

markers = np.zeros_like(image)
markers[image < 60] = 1
markers[image > 122] = 2
print("Watershed Parameters: markers:", markers)

im_true = watershed(elevation_map, markers)
im_true = ndi.label(ndi.binary_fill_holes(im_true - 1))[0]

edges = sobel(image)
print("Watershed Parameters: edges:", edges)
print("Watershed Parameters: markers=130, compactness=0.0015")

im_test1 = watershed(edges, markers=130, compactness=0.0015)

image = img_as_float(image)
gradient = inverse_gaussian_gradient(image, sigma=1.5)
init_ls = np.zeros(image.shape, dtype=np.int8)
init_ls[10:-10, 10:-10] = 1
print("MGAC Parameters: gradient:", gradient)
print("MGAC Parameters: num_iter=400")
print("MGAC Parameters: smoothing=2")
print("MGAC Parameters: threshold=0.58")
im_test3 = morphological_geodesic_active_contour(gradient, num_iter=400,
                                                 init_level_set=init_ls,
                                                 smoothing=2, balloon=-1,
                                                 threshold=0.58)
im_test3 = label(im_test3)

method_names = ['Compact watershed', 'Canny filter',
                'Morphological Geodesic Active Contours']
short_method_names = ['Compact WS', 'Canny', 'GAC']

precision_list = []
recall_list = []
split_list = []
merge_list = []
for name, im_test in zip(method_names, [im_test1, im_test3]):
    error, precision, recall = adapted_rand_error(im_true, im_test)
    splits, merges = variation_of_information(im_true, im_test)
    split_list.append(splits)
    merge_list.append(merges)
    precision_list.append(precision)
    recall_list.append(recall)
    print(f'\n## Method: {name}')
    print(f'Adapted Rand error: {error}')
    print(f'Adapted Rand precision: {precision}')
    print(f'Adapted Rand recall: {recall}')
    print(f'False Splits: {splits}')
    print(f'False Merges: {merges}')

fig, axes = plt.subplots(1, 3, figsize=(9, 6), constrained_layout=True)
ax = axes.ravel()

# ax[0].scatter(merge_list, split_list)
# for i, txt in enumerate(short_method_names):
#     ax[0].annotate(txt, (merge_list[i], split_list[i]),
#                    verticalalignment='center')
# ax[0].set_xlabel('False Merges (bits)')
# ax[0].set_ylabel('False Splits (bits)')
# ax[0].set_title('Split Variation of Information')

# ax[1].scatter(precision_list, recall_list)
# for i, txt in enumerate(short_method_names):
#     ax[1].annotate(txt, (precision_list[i], recall_list[i]),
#                    verticalalignment='center')
# ax[1].set_xlabel('Precision')
# ax[1].set_ylabel('Recall')
# ax[1].set_title('Adapted Rand precision vs. recall')
# ax[1].set_xlim(0, 1)
# ax[1].set_ylim(0, 1)

ax[0].imshow(mark_boundaries(image, im_true))
ax[0].set_title('True Segmentation')
ax[0].set_axis_off()

ax[1].imshow(mark_boundaries(image, im_test1))
ax[1].set_title('Compact Watershed')
ax[1].set_axis_off()

# ax[4].imshow(mark_boundaries(image, im_test2))
# ax[4].set_title('Edge Detection')
# ax[4].set_axis_off()

ax[2].imshow(mark_boundaries(image, im_test3))
ax[2].set_title('Morphological GAC')
ax[2].set_axis_off()

plt.show()