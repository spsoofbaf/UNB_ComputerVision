import matplotlib.pyplot as plt
import numpy as np
from skimage import img_as_ubyte
from PIL import Image
from skimage.morphology import (opening, closing)

image = img_as_ubyte(Image.open("fruit.PNG").convert("1"))

opened1 = opening(image)
opened2 = opening(opened1)
opened3 = opening(opened2)

closed1 = closing(image)
closed2 = closing(closed1)
closed3 = closing(closed2)

# Plotting the result
fig, ax = plt.subplots(nrows=3, ncols=2)

ax[0][0].imshow(opened1, cmap="gray")
ax[0][0].axis('off')
ax[0][0].title.set_text('Opening 1')

ax[1][0].imshow(opened2, cmap="gray")
ax[1][0].axis('off')
ax[1][0].title.set_text('Opening 2')

ax[2][0].imshow(opened3, cmap="gray")
ax[2][0].axis('off')
ax[2][0].title.set_text('Opening 3')

ax[0][1].imshow(closed1, cmap="gray")
ax[0][1].axis('off')
ax[0][1].title.set_text('Closing 1')

ax[1][1].imshow(closed2, cmap="gray")
ax[1][1].axis('off')
ax[1][1].title.set_text('Closing 2')

ax[2][1].imshow(closed3, cmap="gray")
ax[2][1].axis('off')
ax[2][1].title.set_text('Closing 3')



plt.show()
