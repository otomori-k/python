import numpy as np
A = np.array([1,2,3,4])
print(A)
np.ndim(A)
A.shape
A.shape[0]

B = np.array([[1, 2], [3, 4], [5, 6]])
print(B)

A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])
np.dot(A, B)
