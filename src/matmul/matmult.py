import numpy as np
import numba
from numba.pycc import CC

cc = CC('matmult_cc')




@cc.export('matmult_cc', 'f8[:,:](f8[:,:], f8[:,:])')
@numba.njit(parallel=True)
def matmult(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    rows_A, cols_A = A.shape
    rows_B, cols_B = B.shape
    result = np.zeros((rows_A, cols_B))
    for i in numba.prange(rows_A):
        for j in range(cols_B):
            for k in range(cols_A):
                result[i, j] += A[i, k] * B[k, j]
    
    return result

if __name__ == "__main__":
    cc.compile()

