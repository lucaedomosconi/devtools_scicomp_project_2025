
# performance profiling for square matrices

export OMP_NUM_THREADS=1
outputfolder="logs"
mkdir -p $outputfolder
output="${outputfolder}/performance.csv"
touch $output

echo "m_dim,ranks,nsplits,time_matmul,time_reference_@" > $output

matrix_dims=(256 512 1024 2048 4096)
process_counts=(1 2 4 8)

for dim in "${matrix_dims[@]}"; do
  for proc in "${process_counts[@]}"; do
    echo -n "${dim},${proc},1," >> $output
    mpirun -n $proc python scripts/run.py --onlytime --test -ns 1 -f experiments/square_${dim}_matrices.yaml >> $output
  done
done
