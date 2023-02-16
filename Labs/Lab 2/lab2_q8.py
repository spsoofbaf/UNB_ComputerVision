import numpy as np


# Euclidean
def calculate_euclidean_distance(p, q):
    return np.sqrt(np.power(p[0] - q[0], 2) + np.power(p[1] - q[1], 2))


# Manhattan
def calculate_manhattan_distance(p, q):
    return np.absolute(p[0] - q[0]) + np.absolute(p[1] - q[1])


# Chessboard
def calculate_chessboard_distance(p, q):
    return np.maximum(np.absolute(p[0] - q[0]), np.absolute(p[1] - q[1]))


random_image = np.random.randint(255, size=(5, 5))

columns = 5
rows = 5

euclidean = np.zeros(([columns, rows]), dtype=np.int8)
manhattan = np.zeros(([columns, rows]), dtype=np.int8)
chessboard = np.zeros(([columns, rows]), dtype=np.int8)

center = (2, 2)

for x in range(columns):
    for y in range(rows):
        euclidean[x][y] = calculate_euclidean_distance((x, y), center)
        manhattan[x][y] = calculate_manhattan_distance((x, y), center)
        chessboard[x][y] = calculate_chessboard_distance((x, y), center)

print("\nEuclidean: \n")
print(euclidean)
print("\nManhattan: \n")
print(manhattan)
print("\nChessboard: \n")
print(chessboard)
