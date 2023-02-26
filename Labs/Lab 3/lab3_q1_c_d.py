import numpy as np
from skimage.filters import threshold_isodata, threshold_otsu, threshold_local
import matplotlib.pyplot as plt
from PIL import Image

# Define the image
image = np.array([[6.0, 5, 8, 7], [4, 2, 3, 8], [1, 8, 6, 1]])

# # Compute the optimal threshold using Ridler-Calvard method
t_rc = threshold_isodata(image)

# Compute the optimal threshold using Otsu's method
t_ot = threshold_otsu(image)

print("Optimal threshold using Ridler-Calvard method = " + str(
    t_rc) + "  =>  Rounded threshold (Ridler-Calvard) = " + str(round(t_rc)))
print("Optimal threshold using Otsu's method = " + str(t_ot) + "  =>  Rounded threshold (Otsu) = " + str(round(t_ot)))

original_image = Image.open("image2.png")
image2 = np.array(original_image.convert("L"))

t_rc2 = threshold_isodata(image2)

# Compute the optimal threshold using Otsu's method
t_ot2 = threshold_otsu(image2)

binary_rc = image2 > t_rc2
binary_ot = image2 > t_ot2

# Plotting the result
fig, ax = plt.subplots(nrows=1, ncols=4)

ax[0].imshow(image2, cmap="gray")
ax[0].axis('off')
ax[0].title.set_text('Original Image')

ax[1].imshow(binary_rc, cmap="gray")
ax[1].axis('off')
ax[1].title.set_text('Ridler-Calvard')

ax[2].imshow(binary_ot, cmap="gray")
ax[2].axis('off')
ax[2].title.set_text('Otsu')

ax[3].imshow(image2 > threshold_local(image2), cmap="gray")
ax[3].axis('off')
ax[3].title.set_text('local')

plt.show()

# Notes
# The Otsu result is better than the Ridler-Calvard in this image, because two regions have different variances.
# The global thresholding has better result in this image because the noise doesn't vary across the image
