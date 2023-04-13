import numpy as np
from scipy.signal import convolve2d

# Convolve the 2D image I with the 2D kernel G
# I: 2D image
I = np.array([[5, 4, 0, 3],
              [6, 2, 1, 8],
              [7, 9, 4, 2],
              [8, 3, 6, 1]])
# G: 2D kernel
G = (1.0 / 16) * np.array([[1, 2, 1],
                           [2, 4, 2],
                           [1, 2, 1]])

# Replicate-padding the input image
padded_I = np.pad(I, pad_width=1, mode='edge')

# Convolve the image with the kernel
convolved_I = convolve2d(padded_I, G, mode='valid')

# Print the result
print("\nConvolved image:\n")
print(convolved_I)

# And if we want to convert the result to integers:
output = convolved_I.astype(int)
print("\n\nConvolved image (as integers):\n")
print(output)
