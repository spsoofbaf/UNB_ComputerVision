from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# b)
# reading the original image
rgb_image = Image.open("suburb.jpg")

rgb_image_array = np.array(rgb_image)

# converting to grayscale
grayscale_image = rgb_image.convert("L")

image = np.array(grayscale_image)


def flip(x, y, rows):
    return (x, rows - 1 - y)


def flop(x, y, columns):
    return (columns - 1 - x, y)


def rotate_90(x, y, rows):
    return (rows - y - 1, x)


columns = grayscale_image.size[0]
rows = grayscale_image.size[1]

fliped = image.copy()
floped = image.copy()
inverted = 255 - image
rotated = np.full((columns, rows), 0)
fliped2 = rgb_image_array.copy()
floped2 = rgb_image_array.copy()
inverted2 = 255 - rgb_image_array
rotated2 = np.zeros(([columns, rows, 3]), dtype=np.uint8)

for x in range(columns):
    for y in range(rows):
        (x_fliped, y_fliped) = flip(x, y, rows)
        (x_floped, y_floped) = flop(x, y, columns)
        (x_rotated, y_rotated) = rotate_90(x, y, rows)
        fliped[y][x] = image[y_fliped][x_fliped]
        floped[y][x] = image[y_floped][x_floped]
        rotated[x][y] = image[x_rotated][y_rotated]
        fliped2[y][x] = rgb_image_array[y_fliped][x_fliped]
        floped2[y][x] = rgb_image_array[y_floped][x_floped]
        rotated2[x][y] = rgb_image_array[x_rotated][y_rotated]


flipped_image = Image.fromarray(fliped)
flopped_image = Image.fromarray(floped)
inverted_image = Image.fromarray(inverted)
rotated_image = Image.fromarray(rotated)

flipped_image2 = Image.fromarray(fliped2)
flopped_image2 = Image.fromarray(floped2)
inverted_image2 = Image.fromarray(inverted2)
rotated_image2 = Image.fromarray(rotated2)

# rgb_image.show()
# grayscale_image.show()
# flipped_image.show()
# flopped_image.show()
# inverted_image.show()
# rotated_image.show()
# flipped_image2.show()
# flopped_image2.show()
# inverted_image2.show()
# rotated_image2.show()

fig0, ax0 = plt.subplots(nrows=3, ncols=2, sharey='all')

ax0[0, 0].imshow(rgb_image, cmap="gray")
ax0[0, 0].axis('off')
ax0[0, 0].title.set_text('Original Image')

ax0[0, 1].imshow(grayscale_image, cmap="gray")
ax0[0, 1].axis('off')
ax0[0, 1].title.set_text('Grayscale Image')

ax0[1, 1].imshow(flopped_image, cmap="gray")
ax0[1, 1].axis('off')
ax0[1, 1].title.set_text('Flopped Image')

ax0[1, 0].imshow(flipped_image, cmap="gray")
ax0[1, 0].axis('off')
ax0[1, 0].title.set_text('Flipped Image')

ax0[1, 1].imshow(flopped_image, cmap="gray")
ax0[1, 1].axis('off')
ax0[1, 1].title.set_text('Flopped Image')

ax0[2, 0].imshow(inverted_image, cmap="gray")
ax0[2, 0].axis('off')
ax0[2, 0].title.set_text('Inverted Image')

ax0[2, 1].imshow(rotated_image, cmap="gray")
ax0[2, 1].axis('off')
ax0[2, 1].title.set_text('Rotated Image')

plt.show()

# C)
fig, ax = plt.subplots(nrows=2, ncols=2, sharey='all')

ax[0, 0].imshow(flipped_image2, cmap="gray")
ax[0, 0].axis('off')
ax[0, 0].title.set_text('Flipped Image')

ax[0, 1].imshow(flopped_image2, cmap="gray")
ax[0, 1].axis('off')
ax[0, 1].title.set_text('Flopped Image')

ax[1, 0].imshow(inverted_image2, cmap="gray")
ax[1, 0].axis('off')
ax[1, 0].title.set_text('Inverted Image')

ax[1, 1].imshow(rotated_image2, cmap="gray")
ax[1, 1].axis('off')
ax[1, 1].title.set_text('Rotated Image')

plt.show()

