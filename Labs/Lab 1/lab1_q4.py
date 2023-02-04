columns = 640
rows = 480

(x1, y1) = (38, 52)
(x2, y2) = (592, 241)
(x3, y3) = (33, 0)


def calculate_1d_index(x, y, width):
    return y * width + x


def calculate_coordinate(i, width):
    return (i % width, i // width)


print("1D:")
print("index_1: " + str(calculate_1d_index(x1, y1, columns)))
print("index_2: " + str(calculate_1d_index(x2, y2, columns)))
print("index_3: " + str(calculate_1d_index(x3, y3, columns)))

print("2D:")
print("p1: " + str(calculate_coordinate(8092, columns)))
print("p2: " + str(calculate_coordinate(24061, columns)))
print("p3: " + str(calculate_coordinate(38190, columns)))
