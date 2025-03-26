Luca Edoardo Mosconi

lmosconi@sissa.it

Course: Mathematical Analysis, Modelling and Applications

# Matrix Multiplication with MPI
The project implements a function to multiply two real matrices $A(m,k)$ and $B(k,n)$ to obtain $C(m,n)$ using MPI, balancing memory usage and performance according to user needs.


## Parallelization strategy
- **Distribution of matrix A**:
Each rank is assigned approximately m/\<number of ranks\> contiguous rows of A (the case where the number of rows is not a multiple of the number of ranks is handled), needed to compute the corresponding rows of $C$.
- **Handling of matrix B**:
    - *Challenge*: The whole matrix is required by each rank. However, broadcasting the entire matrix can consume significant memory when $B$ is very large, especially when we are using many ranks on a single machine.
    - *Solution*: The function accepts a further parameter to split $B$ into smaller chunks of (contiguous) rows. Only rank 0 will contain the whole $B$ during the whole program execution, the other ranks will receive only a part of $B$ at a time, and use it for the required computations. Memory profiling demonstrated the benefits of this approach with minimal performance cost. Also one can set a suitable number of OpenMP threads on local machine avoiding any memory duplication.
## Implementation details
- **C++ Library with Pybind11**:
In order to obtain the maximum performance, a simple C++ library was written and bound to the Python module with [pybind11](https://github.com/pybind/pybind11).
- **Optimized BLAS Libraries**:
Inside the compiled plugin cblas routines are invoked. In the proposed install process [OpenBlas](https://github.com/OpenMathLib/OpenBLAS) is adopted for its portability, but to improve performances one could use CPU brand mathematical libraries (mkl of aocl) exploiting available vector extended instructions set at best.

**Note**: The implemented function exploits the benefits of numpy array contiguity (in Row-major style by default). When this requirement is not met by input matrices a copy is carried out.

## Install
In order to correctly install using a python virtual environment is suggested (as in the GitHub workflow ```main.yml```).

After cloning the repository in directory `devtools_scicomp_project_2025` create a virtual environment:
```bash
cd devtools_scicomp_project_2025
python -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
```
Then, install required package and compile the C++ library with
```bash
python -m pip install --upgrade pip setuptools wheel
sudo apt-get update
sudo apt-get install -y build-essential gfortran \
libopenblas-dev openmpi-bin libopenmpi-dev
python -m pip install pybind11
g++ -O3 -Wall -shared -fPIC $(python3 -m pybind11 --includes) \
-static-libstdc++ src/matmul/matmult.cpp -lopenblas \
-o src/matmul/matmult_pbcc$(python3-config --extension-suffix)
python -m pip install .
```
## Test
It is possible to verify the project was correctly installed by running the tests with `pytest` command:
```bash
python -m pip install pytest
mpiexec -n 4 python -m pytest
```
## Performance & Memory Profiling
### Memory profiling
Before proceeding make sure the memory profiler is installed:
```bash
python -m pip install -U memory-profiler
```
Run the script `shell/mem_prof.sh` to visualize memory consumption for 12 different combinations of number of ranks and slices of $B$. The output pictures in `logs/ps<number of processes>spl<number of blocks of B>` show that while memory consumption of rank 0 is constant across simulations, when we use multiprocessing, increasing the number of splits (-ns 4) reduces the amount of memory required by other ranks. To better visualize this, compare for example `logs/ps4spl1/memprof.png` and `logs/ps4spl4/memprof.png`.

**Note**: As it is, the script generates only the png pictures in the relative folders. In order to visualize line by line profiling of memory consumption for each rank you should insert (or uncomment) the following lines before function `matrixMultiply` in `src/matmul/matmult.py`:
```python
fp=open(os.environ["logfile"],'w+')
@profile(stream=fp)
```

### Performance for different matrix sizes and rank numbers
Run the script `shell/perform.sh` to compare performance of implemented function with different number of ranks and different dimensions of square matrices (in this case $B$ is not split during the process). The output is saved by default in a csv file `logs/performance.log`.
Note that for small matrices the overhead of mpi causes the code to be less efficient than its numpy @ counterpart (using 1 thread). However as we consider bigger matrices a significant improvement can be observed.