import numpy as np
import numba
from mpi4py import MPI
from memory_profiler import profile
# import cProfile


# @numba.jit(nopython=True)
def sub_matrixMultiply(A, B, size, rank):
    n_rows_A, n_cols_A = A.shape
    n_rows_B, n_cols_B = B.shape
    if n_cols_A != n_rows_B:
        raise ValueError("Inner dimensions of matrices don't match.")
    n_rows_C = n_rows_A
    n_cols_C = n_cols_B
    C = np.zeros((n_rows_C, n_cols_C))
    for i in range(rank, n_rows_A, size):
        for j in range(n_cols_B):
            for k in range(n_cols_A):
                C[i, j] += A[i, k] * B[k, j]
    return C


@profile
def matrixMultiply(A_in, B_in, comm, rank, size):
    """
    Multiply two matrices A and B using the naive method.
    """
    if rank == 0:
        print("Multiplying two matrices using MPI.")
        if A_in is None or B_in is None:
            raise ValueError("Input matrices must not be None.")
        if not isinstance(A_in, np.ndarray) or not isinstance(B_in, np.ndarray):
            raise ValueError("Input matrices must be numpy arrays.")
        if A_in.ndim != 2 or B_in.ndim != 2:
            raise ValueError("Input matrices must be 2D.")
        if A_in.shape[1] != B_in.shape[0]:
            raise ValueError("Inner dimensions of matrices don't match.")
        if comm is None:
            raise ValueError("MPI communicator must not be None.")
        if not isinstance(comm, MPI.Comm):
            raise ValueError("MPI communicator must be of type MPI.Comm.")
        if size < 1:
            raise ValueError("Number of processes must be at least 1.")
        if A_in.flags['C_CONTIGUOUS'] == False:
            print("Matrix A is not C contiguous. Will create a copy.")
            A_in = A_in.ascontiguousarray()
        if B_in.flags['C_CONTIGUOUS'] == False:
            print("Matrix B is not C contiguous. Will create a copy.")
            B_in = B_in.ascontiguousarray()
        rows_A, cols_A = A_in.shape
        rows_B, cols_B = B_in.shape
    else:
        A_in = None
        B_in = None
        rows_A = cols_A = rows_B = cols_B = None
    
    rows_A = comm.bcast(rows_A, root=0)
    cols_A = comm.bcast(cols_A, root=0)
    rows_B = comm.bcast(rows_B, root=0)
    cols_B = comm.bcast(cols_B, root=0)

    A_local_rows = np.zeros(size, dtype=int)
    A_local_rows[:] = rows_A // size
    extra_rows = rows_A % size
    A_local_rows[:extra_rows] += 1
    
    A_loc = np.zeros((A_local_rows[rank], cols_A))
    sendcounts = A_local_rows*cols_A
    senddispls = np.insert(np.cumsum(sendcounts), 0, 0)[:-1]
    comm.Scatterv([A_in, sendcounts, senddispls, MPI.DOUBLE], A_loc, root=0)
    C_loc = np.zeros((A_local_rows[rank], cols_B))
    B = comm.bcast(B_in, root=0)
    C_loc = A_loc @ B
    C = None
    if rank == 0:
        C = np.zeros((rows_A, cols_B))
    recvcounts = A_local_rows * cols_B
    recvdispls = np.insert(np.cumsum(recvcounts), 0, 0)[:-1]
    comm.Gatherv(C_loc, [C, recvcounts, recvdispls, MPI.DOUBLE], root=0)
    
    return C