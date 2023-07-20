#!/bin/bash

# Remember:
# Submit via: sbatch ...
# Check via: squeue -u $USER

#SBATCH -J Python

#SBATCH --account=fc_cosi
#SBATCH --partition=savio3_gpu
#SBATCH --qos=savio_lowprio

#SBATCH -t 2:00:00

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:V100:2

#SBATCH --signal=2@60

# --> CHANGE TO YOUR EMAIL

##SBATCH --akotamraju@berkeley.edu

##SBATCH --mail-type=ALL

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

echo "Starting analysis on host ${HOSTNAME} with job ID ${SLURM_JOB_ID}..."

echo "Loading modules..."
module load nano
module load gcc
module load python

echo "Starting execution..."

python3 predict_cone.py

echo "Waiting for all processes to end..."
wait