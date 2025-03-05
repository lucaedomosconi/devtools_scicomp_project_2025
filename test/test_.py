import matmul.matmult as matmult
import numpy as np
from mpi4py import MPI


def test_1():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    n_rows_A = 4095
    n_cols_A = 4097
    n_rows_B = n_cols_A
    n_cols_B = 4099
    A = None
    B = None
    C = None
    # Generate two random matrices A and B with specified dimensions
    if rank == 0:
        A = np.random.rand(n_rows_A, n_cols_A)
        B = np.random.rand(n_rows_B, n_cols_B)


    
    C = matmult.matrixMultiply(A, B, comm, rank, size, 4)



    if rank == 0:
        C1 = A@B
        C_C1_difference = np.linalg.norm(C-C1)
        assert (C_C1_difference < 0.00001) == True

