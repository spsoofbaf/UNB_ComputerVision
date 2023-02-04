from PIL import Image
import numpy as np
import cv2 as cv

arr = np.array([[5, 8, 3, 7],
                [1, 3, 3, 9],
                [6, 8, 2, 7],
                [4, 1, 0, 9]])

# b)
# reading the original image
rgb_image = Image.open("s.png")
rgb_image_array = np.array(rgb_image)

# converting to grayscale
grayscale_image = rgb_image.convert("L")

grayscale_image_array = np.array(grayscale_image)

columns = grayscale_image.size[0]
rows = grayscale_image.size[1]

k = 0
l = rows * columns


def calculate_histogram(image, columns, rows):
    h = np.full((256, 1), 0)
    for x in range(columns):
        for y in range(rows):
            h[image[y][x]] = h[image[y][x]] + 1

    return h


def calculate_normalized_histogram(histogram, n):
    return histogram / n


def calculate_running_sum(a):
    length = len(a)
    s = np.full((length, 1), 0.0)
    s[0] = a[0]
    for k in range(1, length - 1):
        s[k] = s[k - 1] + a[k]
    return s


def calculate_histogram_equalization(image, columns, rows):
    I = np.full((rows, columns), 0)

    histogram = calculate_histogram(image, columns, rows)
    normalized_histogram = calculate_normalized_histogram(histogram, columns * rows)

    for x in range(columns):
        for y in range(rows):
            I[y][x] = np.round(256 * (calculate_running_sum(normalized_histogram)[image[y][x]]))
    return I


histogram_equalization = calculate_histogram_equalization(grayscale_image_array, columns, rows)

new_image = Image.fromarray(histogram_equalization)
new_image.show()

# c) using opencv
src = cv.imread("s.png")
src = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
dst = cv.equalizeHist(src)
cv.imshow('Source image', src)
cv.imshow('Equalized Image', dst)
cv.waitKey()

difference = np.full((rows, columns), 0)
for x in range(columns):
    for y in range(rows):
        if (dst[y][x] != histogram_equalization[y][x]):
            difference[y][x] = 255

difference_image = Image.fromarray(difference)
difference_image.show()
