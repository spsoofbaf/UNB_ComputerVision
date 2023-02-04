import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

arr = np.array([[100, 29, 230, 94, 50],
                  [116, 217, 138, 219, 250],
                  [37, 169, 202, 14, 99],
                  [106, 117, 138, 119, 150]])

image = Image.fromarray(arr)

plt.imshow(image)
plt.show()
