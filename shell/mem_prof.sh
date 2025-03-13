
# memory profiling for square matrices

# dim A = 4096 x 10000
# dim B = 10000 x 4096
export OMP_NUM_THREADS=1

mprof run --interval 0.02 --multiprocess mpirun -n 1 python scripts/run.py -ns 1 --logfolder=ps1spl1 -f mem_prof_config.yaml
mprof plot -o logs/ps1spl1/memprof.png

sleep 0.5

mprof run --interval 0.02 --multiprocess mpirun -n 2 python scripts/run.py -ns 1 --logfolder=ps2spl1 -f mem_prof_config.yaml
mprof plot -o logs/ps2spl1/memprof.png

sleep 0.5

mprof run --interval 0.02 --multiprocess mpirun -n 4 python scripts/run.py -ns 1 --logfolder=ps4spl1 -f mem_prof_config.yaml
mprof plot -o logs/ps4spl1/memprof.png

sleep 0.5

mprof run --interval 0.02 --multiprocess mpirun -n 1 python scripts/run.py -ns 2 --logfolder=ps1spl2 -f mem_prof_config.yaml
mprof plot -o logs/ps1spl2/memprof.png

sleep 0.5

mprof run --interval 0.02 --multiprocess mpirun -n 2 python scripts/run.py -ns 2 --logfolder=ps2spl2 -f mem_prof_config.yaml
mprof plot -o logs/ps2spl2/memprof.png

sleep 0.5

mprof run --interval 0.02 --multiprocess mpirun -n 4 python scripts/run.py -ns 2 --logfolder=ps4spl2 -f mem_prof_config.yaml
mprof plot -o logs/ps4spl2/memprof.png

sleep 0.5

mprof run --interval 0.02 --multiprocess mpirun -n 1 python scripts/run.py -ns 4 --logfolder=ps1spl4 -f mem_prof_config.yaml
mprof plot -o logs/ps1spl4/memprof.png

sleep 0.5

mprof run --interval 0.02 --multiprocess mpirun -n 2 python scripts/run.py -ns 4 --logfolder=ps2spl4 -f mem_prof_config.yaml
mprof plot -o logs/ps2spl4/memprof.png

sleep 0.5

mprof run --interval 0.02 --multiprocess mpirun -n 4 python scripts/run.py -ns 4 --logfolder=ps4spl4 -f mem_prof_config.yaml
mprof plot -o logs/ps4spl4/memprof.png

rm mprofile*.dat