import numpy as np

a = np.array([[1, 1, 1, 0],
              [0, 1, 0, 0],
              [1, 1, 1, 0],
              [0, 0, 0, 0]], dtype=bool)

b = np.array([[0, 0, 0, 0],
              [0, 1, 1, 0],
              [0, 1, 1, 1],
              [0, 1, 0, 1]], dtype=bool)

# union
a_union_b = np.logical_or(a, b)
print("\nUnion: \n")
print(a_union_b)

# intersection
a_intersect_b = np.logical_and(a, b)
print("\nIntersection: \n")
print(a_intersect_b)

# reflection
b_reflection = np.fliplr(np.flipud(b))
print("\nReflection: \n")
print(b_reflection)

# complement
a_complement = np.logical_not(a)
print("\nComplement: \n")
print(a_complement)

# difference
a_difference_b = np.logical_and(a, np.logical_not(b))
print("\nDifference: \n")
print(a_difference_b)
