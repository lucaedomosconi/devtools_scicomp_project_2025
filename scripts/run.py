

import numpy as np
import time
import os
from mpi4py import MPI
from argparse import ArgumentParser
import yaml
# Set the number of threads
# os.environ["OMP_NUM_THREADS"] = "8"
# os.environ["OPENBLAS_NUM_THREADS"] = "8"
# os.environ["NUMEXPR_NUM_THREADS"] = "8"
# os.environ["MKL_NUM_THREADS"] = "8"

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()



parser = ArgumentParser()
parser.add_argument("-ns","--numsplitB", type=int, default=1, help="number of (horizontal) blocks to split B")
parser.add_argument("--time", action="store_true", help="time execution of the function")
parser.add_argument("--logfolder", type=str, help="memory usage logs will be saved in 'logs/<logfolder>/mem_prof_rank_<mpi rank>'")
parser.add_argument("--test", action="store_true", help="compare implemented function with @ operator")
parser.add_argument("-f","--file", type=str, help="read parameters in yaml file '<file>'")
parser.add_argument("--onlytime", action="store_true", help="only plot time of the execution of the function")

args = parser.parse_args()
if args.logfolder is not None:
    os.makedirs(args.logfolder, exist_ok=True)
    os.environ["logfile"] = args.logfolder + f"/mem_prof_rank_{rank}.txt"
else:
    os.environ["logfile"] = f"logs/mem_prof_rank_{rank}.txt"
import matmul.matmult as matmult


n_rows_A = 2560
n_cols_A = 2560
n_rows_B = n_cols_A
n_cols_B = 256
if args.file is not None:
    with open(args.file,"r") as conf_file:
        config = yaml.safe_load(conf_file)
        n_rows_A = config["n_rows_A"]
        n_cols_A = config["n_cols_A"]
        n_rows_B = n_cols_A
        n_cols_B = config["n_cols_B"]
A = None
B = None
C = None
# Generate two random matrices A and B with specified dimensions
if rank == 0:
    A = np.random.rand(n_rows_A, n_cols_A)
    B = np.random.rand(n_rows_B, n_cols_B)

if (args.time or args.onlytime) and rank == 0:
    time1 = time.time()

C = matmult.matrixMultiply(A, B, comm, rank, size, args.numsplitB)
if args.time and rank == 0:
    time2 = time.time()
    print("Time taken by the implemented function:  ", time2-time1)
if args.onlytime and rank == 0:
    time2 = time.time()
    print(time2-time1, end="")
    if not args.test:
        print()

# to obtain better plot of memory profiler
if not args.time:
    time.sleep(0.2)
if args.test and rank == 0:
    if args.time or args.onlytime:
        time1 = time.time()
    C1 = A@B
    if args.time:
        time2 = time.time()
        print("Time taken by the @ operator:            ", time2-time1)
    elif args.onlytime:
        time2 = time.time()
        print(f",{time2-time1}")

if not args.onlytime and args.test and rank == 0:
    print("Difference between the two methods:      ", np.linalg.norm(C-C1))
