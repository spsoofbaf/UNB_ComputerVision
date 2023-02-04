from PIL import Image
import numpy as np


def interpolate_bilinear(image, x, y):
    x0 = int(np.floor(x))
    y0 = int(np.floor(y))
    alpha_x = x - x0
    alpha_y = y - y0
    alpha_bar_x = 1 - alpha_x
    alpha_bar_y = 1 - alpha_y

    return (alpha_bar_x * alpha_bar_y * image[y0][x0]) + (
            alpha_x * alpha_bar_y * image[y0][x0 + 1]) + (
            alpha_bar_x * alpha_y * image[y0 + 1][x0]) + alpha_x * alpha_y * image[y0 + 1][x0 + 1]


I = np.array([[232, 177, 82, 7],
              [241, 18, 152, 140],
              [156, 221, 67, 3]])

p1 = (0.1, 0.7)
p2 = (1.2, 0.5)
p3 = (1.3, 1.6)
p4 = (2.8, 1.7)

interpolated_p1 = interpolate_bilinear(I, p1[0], p1[1])
interpolated_p2 = interpolate_bilinear(I, p2[0], p2[1])
interpolated_p3 = interpolate_bilinear(I, p3[0], p3[1])
interpolated_p4 = interpolate_bilinear(I, p4[0], p4[1])

print(interpolated_p1)
print(interpolated_p2)
print(interpolated_p3)
print(interpolated_p4)
