#!/bin/bash

#SBATCH --reservation=fri
#SBATCH --partition=gpu
#SBATCH --job-name=lennard-jones
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --output=lj_out.log
#SBATCH --time=03:00:00 

#LOAD MODULES 
module load CUDA

#BUILD
make

#RUN
echo "===== CPU ====="
for N in 1000 2000 4000; do
    echo "--- N=$N ---"
    for RUN in 1 2 3 4 5; do
        srun ./lj.out $N 5000 0
    done
done

echo "--- N=8000 (1 run) ---"
srun ./lj.out 8000 5000 0

echo "===== GPU ====="
for N in 1000 2000 4000 8000; do
    echo "--- N=$N ---"
    for RUN in 1 2 3 4 5; do
        srun ./lj.out $N 5000 1
    done
done

