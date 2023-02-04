from PIL import Image
import numpy as np

def calculate_double_difference(image0, image1, image2, threshold):
    columns = image0.shape[0]
    rows = image0.shape[1]
    double_difference = np.full((rows, columns), 0)

    d1 = abs(image0 - image1)
    d2 = abs(image1 - image2)

    for x in range(columns):
        for y in range(rows):
            if (d1[y][x] > threshold and d2[y][x] > threshold):
                # setting 255 for better visual results
                double_difference[y][x] = 255
            else:
                double_difference[y][x] = 0

    return double_difference


def calculate_triple_difference(image0, image1, image2, threshold):
    columns = image0.shape[0]
    rows = image0.shape[1]
    triple_difference = np.full((rows, columns), 0)

    d1 = abs(image0 - image1)
    d2 = abs(image2 - image1)
    d3 = abs(image2 - image0)

    for x in range(columns):
        for y in range(rows):
            if ((d1[y][x] + d2[y][x] - d3[y][x]) > threshold):
                # setting 255 for better visual results
                triple_difference[y][x] = 255
            else:
                triple_difference[y][x] = 0

    return triple_difference


I1 = np.array([[18, 168, 94, 67],
               [120, 97, 78, 198],
               [83, 70, 208, 17],
               [238, 208, 189, 68]])

I2 = np.array([[21, 168, 92, 71],
               [122, 71, 191, 227],
               [83, 212, 16, 187],
               [240, 216, 188, 68]])

I3 = np.array([[20, 171, 92, 70],
               [76, 193, 39, 255],
               [209, 20, 20, 194],
               [241, 210, 190, 73]])

# I1_difference_image = Image.fromarray(I1)
# I1_difference_image.show()
# I2_difference_image = Image.fromarray(I2)
# I2_difference_image.show()
# I3_difference_image = Image.fromarray(I3)
# I3_difference_image.show()

double_difference = calculate_double_difference(I1, I2, I3, 40)
triple_difference = calculate_triple_difference(I1, I2, I3, 40)

print(double_difference)
double_difference_image = Image.fromarray(double_difference)
double_difference_image.show()

print(triple_difference)
triple_difference_image = Image.fromarray(triple_difference)
triple_difference_image.show()
