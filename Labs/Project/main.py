import skimage
from skimage import feature
from skimage import filters
from skimage import io
from skimage import color
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from skimage.metrics import adapted_rand_error, variation_of_information


# guided filter implementation
def guided_filter(p, I, r, e, return_intermediates=False):
    # convert p to tensor
    p = torch.from_numpy(p).float()
    # convert I to tensor
    I = torch.from_numpy(I).float()

    # reshape to (x,y, 1)
    p = p.reshape(1, p.shape[0], p.shape[1])
    kernel = torch.ones(p.shape[0], 1, r, r) / (r ** 2)
    pad = r // 2

    # reshape I
    I = I.reshape(1, I.shape[0], I.shape[1])

    # pad p and I
    # p = F.pad(p, (pad, pad, pad, pad))
    # I = F.pad(I, (pad, pad, pad, pad))

    mean_p = F.conv2d(p, kernel, padding=pad, groups=1)
    mean_I = F.conv2d(I, kernel, padding=pad, groups=1)
    corr_I = F.conv2d(I * I, kernel, padding=pad, groups=1)
    corr_Ip = F.conv2d(I * p, kernel, padding=pad, groups=1)

    var_I = corr_I - mean_I * mean_I
    cov_Ip = corr_Ip - mean_I * mean_p

    a = cov_Ip / (var_I + e)
    b = mean_p - a * mean_I
    mean_a = F.conv2d(a, kernel, padding=pad, groups=1)
    mean_b = F.conv2d(b, kernel, padding=pad, groups=1)

    # pad mean_b by 2
    # mean_b = F.pad(mean_b, (pad, pad, pad, pad))
    print(mean_b.shape)
    print(mean_a.shape)
    print(I.shape)

    # mean B : torch.Size([1, 442, 702])
    # mean A : torch.Size([1, 442, 702])
    # I Size : torch.Size([1, 440, 700])

    # reshape I to torch.Size([1, 442, 702]) from torch.Size([1, 440, 700])
    # I = F.pad(I, (1, 1, 1, 1))

    q = mean_a * I + mean_b
    if return_intermediates:
        return mean_p, mean_I, corr_I, corr_Ip, \
            var_I, cov_Ip, a, b, mean_a, mean_b, q
    else:
        return q


def main():
    # Load image
    image = io.imread("Leaf.jpg")
    # Convert image to grayscale
    image = color.rgb2gray(image)

    org = image

    # Apply Canny edge detection without smoothing
    edges1 = feature.canny(image, sigma=0)

    # Apply Canny edge detection with gaussian smoothing (sigma=2)
    edges2 = feature.canny(image, sigma=2)

    # Apply Canny edge detection with gaussian smoothing (sigma=3)
    edges3 = feature.canny(image, sigma=3)

    # Apply bilateral filter
    bilateral = skimage.restoration.denoise_bilateral(image, sigma_spatial=3, win_size=8)
    # Apply Canny edge detection to smooth image with bilateral filter
    edges5 = feature.canny(bilateral, sigma=0)

    # Apply guided filter
    image_smoothed_guided = guided_filter(image, image, 15, 0.01)
    # get image from tensor to numpy
    image_smoothed_guided = image_smoothed_guided.numpy()
    image_smoothed_guided = image_smoothed_guided.reshape(image_smoothed_guided.shape[1],
                                                          image_smoothed_guided.shape[2])
    # Apply Canny edge detection to smoothed image with guided filter
    edges6 = feature.canny(image_smoothed_guided, sigma=0)

    # Apply median filter
    image_smoothed = filters.median(image, mode="reflect")
    # Apply Canny edge detection to smoothed image with median filter
    edges7 = feature.canny(image_smoothed, sigma=0)

    # Display results
    fig, axes = plt.subplots(nrows=4, ncols=2, figsize=(8, 8))
    ax = axes.ravel()
    ax[0].imshow(org, cmap=plt.cm.gray)
    ax[0].set_title("Original image")
    ax[1].imshow(edges1, cmap=plt.cm.gray)
    ax[1].set_title("Canny without smoothing")
    ax[2].imshow(edges2, cmap=plt.cm.gray)
    ax[2].set_title("Canny with gaussian smoothing (sigma=2)")
    ax[3].imshow(edges3, cmap=plt.cm.gray)
    ax[3].set_title("Canny with gaussian smoothing (sigma=3)")
    ax[4].imshow(edges5, cmap=plt.cm.gray)
    ax[4].set_title("Canny with bilateral filter")
    # ax[4].imshow(edges5 + 0.4 * (0.9 - image) + 0.7 * (edges2), cmap=plt.cm.gray)
    ax[5].imshow(edges6, cmap=plt.cm.gray)
    ax[5].set_title("Canny with guided filter")
    ax[6].imshow(edges7, cmap=plt.cm.gray)
    ax[6].set_title("Canny with median filter")

    for a in ax:
        a.axis("off")
    fig.tight_layout()
    plt.show()

    method_names = ['Canny filter without smoothing',
                    'Canny filter with gaussian smoothing (sigma=2)',
                    'Canny filter with gaussian smoothing (sigma=3)',
                    'Canny filter with bilateral filter',
                    'Canny filter with guided filter',
                    'Canny filter with median filter']
    im_test1 = edges1
    im_test2 = edges2
    im_test3 = edges3
    # im_test4 = edges4
    im_test5 = edges5
    im_test6 = edges6
    im_test7 = edges7
    image255 = ((image / np.max(image)) * 255).astype(np.uint8)
    im_true = image255
    split_list = []
    merge_list = []
    precision_list = []
    recall_list = []

    # Compute the adapted Rand error, precision, recall and the false splits and merges
    for name, im_test in zip(method_names, [im_test1, im_test2, im_test3, im_test5, im_test6, im_test7]):
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


if __name__ == "__main__":
    main()
