
import matmul.matmult as matmult
import numpy as np
import time
import os
from mpi4py import MPI

# Set the number of threads
os.environ["OMP_NUM_THREADS"] = "8"
os.environ["OPENBLAS_NUM_THREADS"] = "8"
os.environ["NUMEXPR_NUM_THREADS"] = "8"
# os.environ["MKL_NUM_THREADS"] = "8"

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

n_rows_A = 4096
n_cols_A = 4096
n_rows_B = n_cols_A
n_cols_B = 4096
A = None
B = None
C = None
# Generate two random matrices A and B with specified dimensions
if rank == 0:
    A = np.random.rand(n_rows_A, n_cols_A)
    B = np.random.rand(n_rows_B, n_cols_B)



time1 = time.time()
for i in range(1):
    C = matmult.matrixMultiply(A, B, comm, rank, size)
time2 = time.time()
if rank == 0:
    print("Time taken by the Python function:", time2-time1)


time1 = time.time()
if rank == 0:
    C1 = A@B
time2 = time.time()
if rank == 0:
    print("Time taken by the @ operator:", time2-time1)



if rank == 0:
    print("Difference between the two methods:", np.linalg.norm(C-C1))
