from PIL import Image

# a)
# reading the original image
rgb_image = Image.open("suburb.jpg")

# converting to grayscale
grayscale_image = rgb_image.convert("L")

# saving the grayscale image as png
grayscale_image.save("suburb_grayscale.png", "png")

# b)
# resizing the grayscale image
size = grayscale_image.size
grayscale_image_resized = grayscale_image.resize((size[0] // 2, size[1] // 2))
grayscale_image_resized.save("suburb_grayscale_resized.png", "png")

# showing the resized image
grayscale_image_resized.show()
