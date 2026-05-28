#!/bin/bash

#SBATCH --reservation=fri
#SBATCH --job-name=lenia
#SBATCH --ntasks-per-node=2
#SBATCH --nodes=1
#SBATCH --output=lenia_out.log
#SBATCH --hint=nomultithread

#Load MPI module 
module load OpenMPI

#Build
make

#Run
SIZES=(128 512 1024 2048 4096)
REPEATS=5

for SIZE in "${SIZES[@]}"; do
    echo -n "PROCS=$SLURM_NTASKS SIZE=${SIZE}x${SIZE} "
    TOTAL=0
    for i in $(seq 1 $REPEATS); do
        TIME=$(mpirun -np $SLURM_NTASKS ./lenia.out $SIZE | grep "Execution time" | awk '{print $3}')
        TOTAL=$(echo "$TOTAL + $TIME" | bc)
    done
    AVG=$(echo "scale=3; $TOTAL / $REPEATS" | bc)
    echo "AVG_TIME=$AVG"
done

