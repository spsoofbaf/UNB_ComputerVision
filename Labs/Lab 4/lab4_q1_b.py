import numpy as np
from scipy.signal import convolve2d

# Convolve the 2D image I with a horizontal and a vertical 1D kernel
# I: 2D image
I = np.array([[5, 4, 0, 3],
              [6, 2, 1, 8],
              [7, 9, 4, 2],
              [8, 3, 6, 1]])
# G: 2 kernels
G1 = (1.0 / 4) * np.array([1, 2, 1])
G2 = (1.0 / 4) * np.array([[1],
                           [2],
                           [1]])
# Replicate-padding the input image
padded_I = np.pad(I, ((1, 1), (1, 1)), mode='edge')

# Convolve the image with the horizontal kernel
convolved_I_horizontal = convolve2d(padded_I, G1[np.newaxis, :], mode='valid')
# Convolve the convolved_I_horizontal with the vertical kernel
convolved_I_vertical = convolve2d(convolved_I_horizontal, G2, mode='valid')

# Print the result
print("\nConvolved image (Horizontal-First)::\n")
print(convolved_I_vertical)

# And if we want to convert the result to integers:
output = convolved_I_vertical.astype(int)
print("\n\nConvolved image (Horizontal-First) (as integers):\n")
print(output)

# Convolve the image with the horizontal kernel
convolved_I_vertical2 = convolve2d(padded_I, G2, mode='valid')
# Convolve the convolved_I_horizontal with the vertical kernel
convolved_I_horizontal2 = convolve2d(convolved_I_vertical2, G1[np.newaxis, :], mode='valid')

# Print the result
print("\nConvolved image (Vertical-First)::\n")
print(convolved_I_horizontal2)

# And if we want to convert the result to integers:
output2 = convolved_I_horizontal2.astype(int)
print("\n\nConvolved image (Vertical-First) (as integers):\n")
print(output2)
