
import matmul.matmult_cc as matmult
import numpy as np
import time

dim = 1000
# Generate two random dimxdim matrices
A = np.random.rand(dim, dim)
B = np.random.rand(dim, dim)
time1 = time.time()
C1 = A@B
time2 = time.time()
print("Time taken by the @ operator:", time2-time1)
time1 = time.time()
C = matmult.matmult_cc(A, B)
time2 = time.time()
print("Time taken by the Python function:", time2-time1)

# print("Difference between the two methods:", np.linalg.norm(C-C1))