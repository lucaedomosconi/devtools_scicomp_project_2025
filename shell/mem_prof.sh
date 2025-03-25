
# memory profiling for square matrices

# dim A = 4096 x 10000
# dim B = 10000 x 4096
export OMP_NUM_THREADS=1
outputfolder="logs_b"
mkdir -p $outputfolder


process_counts=(1 2 4)
splits=(1 2 4)


for split in "${splits[@]}"; do
  for proc in "${process_counts[@]}"; do
    testname="ps${proc}spl${split}"
    mkdir -p "${outputfolder}/${testname}"
    mprof run --interval 0.02 --multiprocess mpirun -n $proc python scripts/run.py -ns $split --logfolder="${outputfolder}/${testname}" -f experiments/mem_prof_config.yaml
    mprof plot -o "${outputfolder}/${testname}/memprof.png"
    sleep 0.5
  done
done


rm mprofile*.dat