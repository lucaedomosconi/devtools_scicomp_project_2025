import numpy as np
from mpi4py import MPI
from . import matmult_pbcc
from memory_profiler import profile
import os
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

# fp=open(os.environ["logfile"],'w+')
# @profile(stream=fp)
def matrixMultiply(A_in : np.ndarray, B_in : np.ndarray, comm : MPI.Comm, rank : int, size : int , n_split_B = 1) -> np.ndarray:
    """
    Multiply two matrices A and B using MPI.
    
    Parameters:
        A (np.ndarray): a 2D NumPy array of float64. Must be passed by rank 0 processor. The other processors may pass a null object ("None").
        B (np.ndarray): a 2D NumPy array of float64. Must be passed by rank 0 processor. The other processors may pass a null object ("None").
        comm (MPI.comm): MPI communicator.
        rank (int): rank of current processor.
        size (int): number of processors.
        n_split_B (int): number of splits of matrix B. 
    Returns:
        C (np.ndarray): a 2D NumPy array of float64. The result is returned only to rank 0 processor
    Description:
        The function computes the product of two matrices. Rows of A are scattered among the different processors, which compute the corresponding rows of C.
        If n_split_B = 1 each processor will have his own copy of B. If n_split_B > 1, B will be split into n_split_B blocks of rows. This blocks will be passed one by one from the rank 0 processor to the others. In this way we split the computations and avoid making a copy of the whole matrix B into each processor. This will result in a smaller amount of memory required by each processor (apart from rank 0).
        The matrices A and B must be row major contiguous. If they are not they will be converted, and the user will be informed.
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
        if n_split_B < 1:
            raise ValueError("n_split_B must be at least 1")
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
    A_extra_rows = rows_A % size
    A_local_rows[:A_extra_rows] += 1
    if rank > 0:
        A_loc = np.empty((A_local_rows[rank], cols_A))
    sendcounts = A_local_rows*cols_A
    senddispls = np.insert(np.cumsum(sendcounts), 0, 0)[:-1]
    if rank == 0:
        comm.Scatterv([A_in, sendcounts, senddispls, MPI.DOUBLE], MPI.IN_PLACE, root=0)
        A_loc = A_in[:A_local_rows[rank],:]
        C = np.zeros((rows_A, cols_B), dtype=float)
        C_loc = C[:A_local_rows[0],:]
    if rank > 0:
        comm.Scatterv([A_in, sendcounts, senddispls, MPI.DOUBLE], A_loc, root=0)
        C_loc = np.zeros((A_local_rows[rank], cols_B), dtype=float)
    # C_loc = np.zeros((A_local_rows[rank], cols_B))
    if n_split_B == 1:
        if rank > 0:
            B_in = np.empty((rows_B,cols_B), dtype=float)
        comm.Bcast(B_in, root=0)
        matmult_pbcc.matrixMultiply(A_loc, B_in, C_loc)
        # C_loc = A_loc@B_in
    else:
        B_local_rows = np.zeros(n_split_B, dtype=int)
        B_local_rows[:] = rows_B // n_split_B
        B_extra_rows = rows_B % n_split_B
        B_local_rows[:B_extra_rows] += 1
        B_rows_splitters = np.insert(np.cumsum(B_local_rows), 0, 0)
        for B_spl_count in range(n_split_B):
            if rank == 0:
                B_loc = B_in[B_rows_splitters[B_spl_count]:B_rows_splitters[B_spl_count+1],:]
            else:
                B_loc = np.empty((B_rows_splitters[B_spl_count+1]-B_rows_splitters[B_spl_count], cols_B))
            comm.Bcast(B_loc, root=0)
            matmult_pbcc.submatrixMultiply(A_loc, B_loc, C_loc, B_rows_splitters[B_spl_count])
    # C_loc = A_loc @ B
    C = None
    if rank == 0:
        C = np.empty((rows_A, cols_B))
    recvcounts = A_local_rows * cols_B
    recvdispls = np.insert(np.cumsum(recvcounts), 0, 0)[:-1]
    comm.Gatherv(C_loc, [C, recvcounts, recvdispls, MPI.DOUBLE], root=0)
    C_loc = None
    return C