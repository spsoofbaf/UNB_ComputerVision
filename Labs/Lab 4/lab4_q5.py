import numpy as np
from matplotlib import pyplot as plt
from skimage.color import rgb2gray
from skimage.feature import CENSURE, match_descriptors, ORB, plot_matches
from skimage.transform import warp, AffineTransform
from PIL import Image

tform = AffineTransform(scale=(0.6, 0.9), rotation=0.1, shear=0.15,
                        translation=(110, 30))

original_image = rgb2gray(Image.open("book.jpg"))
original_image = np.asanyarray(original_image)

image = warp(original_image, tform.inverse)
image = Image.fromarray((image * 255).astype(np.uint8))
image.save("transformed.png", "PNG")

detector = CENSURE()
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))
detector.detect(original_image)

ax[0].imshow(original_image, cmap=plt.cm.gray)
ax[0].scatter(detector.keypoints[:, 1], detector.keypoints[:, 0],
              2 ** detector.scales, facecolors='none', edgecolors='r')
ax[0].set_title("Original Image")

detector.detect(image)

ax[1].imshow(image, cmap=plt.cm.gray)
ax[1].scatter(detector.keypoints[:, 1], detector.keypoints[:, 0],
              2 ** detector.scales, facecolors='none', edgecolors='r')
ax[1].set_title('Transformed Image')

for a in ax:
    a.axis('off')

plt.tight_layout()
plt.show()


descriptor_extractor = ORB(n_keypoints=200)

descriptor_extractor.detect_and_extract(original_image)
keypoints1 = descriptor_extractor.keypoints
descriptors1 = descriptor_extractor.descriptors

descriptor_extractor.detect_and_extract(image)
keypoints3 = descriptor_extractor.keypoints
descriptors3 = descriptor_extractor.descriptors

# matches12 = match_descriptors(descriptors1, descriptors1, cross_check=True)
# matches13 = match_descriptors(descriptors1, descriptors3, cross_check=True)

fig, ax = plt.subplots(nrows=2, ncols=1)

# plt.gray()
#
# plot_matches(ax[0], original_image, image, keypoints1, keypoints2, matches12)
# ax[0].axis('off')
# ax[0].set_title("Original Image vs. Transformed Image")
#
# plot_matches(ax[1], img1, img3, keypoints1, keypoints3, matches13)
# ax[1].axis('off')
# ax[1].set_title("Original Image vs. Transformed Image")


plt.show()

