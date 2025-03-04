

/*
To compile the code, there are two ways:
1. - First preload the mkl libraries with
    $ export LD_PRELOAD="/usr/lib/x86_64-linux-gnu/libmkl_intel_lp64.so:/usr/lib/x86_64-linux-gnu/libmkl_core.so:/usr/lib/x86_64-linux-gnu/libmkl_gnu_thread.so:/usr/lib/x86_64-linux-gnu/libomp.so.5"
   - Then run the command
    $ g++ -O3 -Wall -shared -fPIC $(python3 -m pybind11 --includes) matmult.cpp -L/usr/lib/x86_64-linux-gnu -Wl,-rpath,/usr/lib/x86_64-linux-gnu -lmkl_intel_lp64 -lmkl_core -lmkl_gnu_thread -lpthread -lm -ldl -fopenmp -o matmult_pbcc$(python3-config --extension-suffix)
2. - Uncomment lines void * handle1..., void * handle2..., void * handle3...
   - Run the command same as above
    $ g++ -O3 -Wall -shared -fPIC $(python3 -m pybind11 --includes) matmult.cpp -L/usr/lib/x86_64-linux-gnu -Wl,-rpath,/usr/lib/x86_64-linux-gnu -lmkl_intel_lp64 -lmkl_core -lmkl_gnu_thread -lpthread -lm -ldl -fopenmp -o matmult_pbcc$(python3-config --extension-suffix)
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


py::array_t<real,py::array::c_style> matrixMultiply(py::array_t<real,py::array::c_style> A,
                                                    py::array_t<real,py::array::c_style> B) {
    py::buffer_info bufA = A.request();
    py::buffer_info bufB = B.request();
    py::ssize_t n_rows_A = bufA.shape[0];
    py::ssize_t n_cols_A = bufA.shape[1];
    py::ssize_t n_cols_B = bufB.shape[1];
    // std::cout << "n_rows_A: " << n_rows_A << std::endl;
    // std::cout << "n_cols_A: " << n_cols_A << std::endl;
    // std::cout << "n_rows_B: " << n_rows_B << std::endl;
    // std::cout << "n_cols_B: " << n_cols_B << std::endl;
    py::array_t<real,py::array::c_style> C({n_rows_A,n_cols_B});
    py::buffer_info bufC = C.request();
    real * ptrA = static_cast<real *>(bufA.ptr);
    real * ptrB = static_cast<real *>(bufB.ptr);
    real * ptrC = static_cast<real *>(bufC.ptr);
    void *handle1 = dlopen("/usr/lib/x86_64-linux-gnu/libmkl_gnu_thread.so", RTLD_LAZY | RTLD_GLOBAL);
    void *handle2 = dlopen("/usr/lib/x86_64-linux-gnu/libmkl_intel_lp64", RTLD_LAZY | RTLD_GLOBAL);
    void *handle3 = dlopen("/usr/lib/x86_64-linux-gnu/libmkl_core.so", RTLD_LAZY | RTLD_GLOBAL);


    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                n_rows_A, n_cols_B, n_cols_A,
                1.0,
                ptrA, n_cols_A,ptrB,n_cols_B,
                0.0,ptrC,n_cols_B);
    // real result = cblas_ddot(n_rows_A, ptrA, 1, ptrB, 1);


    // std::cout << result << std::endl;
    return C;
}

PYBIND11_MODULE(matmult_pbcc, m) {
    m.doc() = "pybind11 matrix multiplication plugin"; // optional module docstring

    m.def("matrixMultiply", &matrixMultiply, "A function that multiplies two matrices");
}


