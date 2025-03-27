

/*
To compile the code, there are different ways: 1 and 2 unsing mkl libraries, 3 using openblas:
1. - First preload the mkl libraries with
    $ export LD_PRELOAD="/usr/lib/x86_64-linux-gnu/libmkl_intel_lp64.so:/usr/lib/x86_64-linux-gnu/libmkl_core.so:/usr/lib/x86_64-linux-gnu/libmkl_gnu_thread.so:/usr/lib/x86_64-linux-gnu/libomp.so.5"
   - Then run the command
    $ g++ -O3 -Wall -shared -fPIC $(python3 -m pybind11 --includes) -static-libstdc++ matmult.cpp -L/usr/lib/x86_64-linux-gnu -Wl,-rpath,/usr/lib/x86_64-linux-gnu -lmkl_intel_lp64 -lmkl_core -lmkl_gnu_thread -lpthread -lm -ldl -fopenmp -o matmult_pbcc$(python3-config --extension-suffix)
2. - Uncomment lines void * handle1..., void * handle2..., void * handle3...
   - Run the command same as above
    $ g++ -O3 -Wall -shared -fPIC $(python3 -m pybind11 --includes) -static-libstdc++ matmult.cpp -L/usr/lib/x86_64-linux-gnu -Wl,-rpath,/usr/lib/x86_64-linux-gnu -lmkl_intel_lp64 -lmkl_core -lmkl_gnu_thread -lpthread -lm -ldl -fopenmp -o matmult_pbcc$(python3-config --extension-suffix)
3. - Use openblas libraries (might be a bit less optimized for intel processors)
    $ g++ -O3 -Wall -shared -fPIC $(python3 -m pybind11 --includes) -static-libstdc++ matmult.cpp -lopenblas -o matmult_pbcc$(python3-config --extension-suffix)
*/


#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <cstdlib>
#include <iostream>
#include <vector>
#include <omp.h>
// #include <mkl/mkl_cblas.h>
// #include <mkl/mkl.h>
#include <cblas.h>
#include <dlfcn.h>

using real = double;
namespace py = pybind11;


void matrixMultiply(py::array_t<real,py::array::c_style> A,
                    py::array_t<real,py::array::c_style> B,
                    py::array_t<real,py::array::c_style> C) {
    py::buffer_info bufA = A.request();
    py::buffer_info bufB = B.request();
    py::buffer_info bufC = C.request();
    py::ssize_t n_rows_A = bufA.shape[0];
    py::ssize_t n_cols_A = bufA.shape[1];
    py::ssize_t n_cols_B = bufB.shape[1];
    real * ptrA = static_cast<real *>(bufA.ptr);
    real * ptrB = static_cast<real *>(bufB.ptr);
    real * ptrC = static_cast<real *>(bufC.ptr);
    // void *handle1 = dlopen("/usr/lib/x86_64-linux-gnu/libmkl_gnu_thread.so", RTLD_LAZY | RTLD_GLOBAL);
    // void *handle2 = dlopen("/usr/lib/x86_64-linux-gnu/libmkl_intel_lp64", RTLD_LAZY | RTLD_GLOBAL);
    // void *handle3 = dlopen("/usr/lib/x86_64-linux-gnu/libmkl_core.so", RTLD_LAZY | RTLD_GLOBAL);


    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                n_rows_A, n_cols_B, n_cols_A,
                1.0, ptrA, n_cols_A, ptrB, n_cols_B,
                0.0, ptrC, n_cols_B);
}

void submatrixMultiply(py::array_t<real,py::array::c_style> A,
                       py::array_t<real,py::array::c_style> B,
                       py::array_t<real,py::array::c_style> C,
                       py::ssize_t start_row_A) {
    py::buffer_info bufA = A.request();
    py::buffer_info bufB = B.request();
    py::buffer_info bufC = C.request();
    // py::ssize_t n_rows_A = bufA.shape[0];
    py::ssize_t n_cols_A = bufA.shape[1];
    py::ssize_t n_rows_B = bufB.shape[0];
    py::ssize_t n_cols_B = bufB.shape[1];
    py::ssize_t n_rows_C = bufC.shape[0];
    py::ssize_t n_cols_C = bufC.shape[1];
    real * ptrA = static_cast<real *>(bufA.ptr);
    real * ptrB = static_cast<real *>(bufB.ptr);
    real * ptrC = static_cast<real *>(bufC.ptr);
    // void * handle1 = dlopen("/usr/lib/x86_64-linux-gnu/libmkl_gnu_thread.so", RTLD_LAZY | RTLD_GLOBAL);
    // void * handle2 = dlopen("/usr/lib/x86_64-linux-gnu/libmkl_intel_lp64", RTLD_LAZY | RTLD_GLOBAL);
    // void * handle3 = dlopen("/usr/lib/x86_64-linux-gnu/libmkl_core.so", RTLD_LAZY | RTLD_GLOBAL);


    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                n_rows_C, n_cols_C, n_rows_B,
                1.0, ptrA+start_row_A, n_cols_A, ptrB, n_cols_B,
                1.0, ptrC, n_cols_C);
}



PYBIND11_MODULE(matmult_pbcc, m) {
    m.doc() = "Compiled matrix multiplication plugin\n";

    m.def("matrixMultiply", &matrixMultiply, 
    "Perform matrix multiplication using CBLAS (dgemm).\n"
    "This function multiplies two 2D NumPy arrays (A and B) and stores\n"
    "the result in a third array (C).\n"
    "It uses the CBLAS `dgemm` function for efficient matrix multiplication.\n\n"
    "Parameters\n"
    "----------\n"
    "    A : np.ndarray\n"
    "        A 2D NumPy array (py::array_t) of type `real` in C-style row-major order.\n"
    "        Represents the left-hand side matrix in the multiplication.\n"
    "    B : np.ndarray\n"
    "        A 2D NumPy array (py::array_t) of type `real` in C-style row-major order.\n"
    "        Represents the right-hand side matrix in the multiplication.\n"
    "    C : np.ndarray\n"
    "        A 2D NumPy array (py::array_t) of type `real` in C-style row-major order.\n"
    "        Represents the output matrix where the result of the multiplication is stored.\n\n"
    "Returns\n"
    "-------\n"
    "    None\n\n"
    "Notes\n"
    "-----\n"
    "    The dimensions of the matrices must satisfy the following conditions:\n"
    "        - The number of columns in A must match the number of rows in B.\n"
    "        - The shape of C must match the resulting matrix dimensions (rows of\n"
    "          A x columns of B).\n\n"
    "Warnings\n"
    "--------\n"
    "    - This function assumes that the input arrays are properly allocated and have\n"
    "      valid shapes.\n"
    "    - No checks are performed to validate the input dimensions.\n\n");


    m.def("submatrixMultiply", &submatrixMultiply,
    "Perform submatrix multiplication using CBLAS (dgemm).\n"
    "This function multiplies a submatrix of A (starting from a specified row)\n"
    "with matrix B and adds the result to matrix C.\n"
    "It uses the CBLAS `dgemm` function for efficient matrix multiplication.\n\n"
    "Parameters\n"
    "----------\n"
    "    A : np.ndarray\n"
    "        A 2D NumPy array (py::array_t) of type `real` in C-style row-major order.\n"
    "        Represents the left-hand side matrix in the multiplication.\n"
    "    B : np.ndarray\n"
    "        A 2D NumPy array (py::array_t) of type `real` in C-style row-major order.\n"
    "        Represents the right-hand side matrix in the multiplication.\n"
    "    C : np.ndarray\n"
    "        A 2D NumPy array (py::array_t) of type `real` in C-style row-major order.\n"
    "        Represents the output matrix where the result of the multiplication is added.\n"
    "    start_row_A : int\n"
    "        The starting row index in matrix A from which the submatrix is taken.\n\n"
    "Returns\n"
    "-------\n"
    "    None\n\n"
    "Notes\n"
    "-----\n"
    "    The dimensions of the matrices must satisfy the following conditions:\n"
    "        - The number of columns in A must match the number of rows in B.\n"
    "        - The shape of C must match the resulting matrix dimensions (rows of\n"
    "          the submatrix of A x columns of B).\n\n"
    "Warnings\n"
    "--------\n"
    "    - This function assumes that the input arrays are properly allocated and have\n"
    "      valid shapes.\n"
    "    - No checks are performed to validate the input dimensions.\n\n");
}