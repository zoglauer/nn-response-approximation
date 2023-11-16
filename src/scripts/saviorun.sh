#!/bin/bash

# Remember:
# Submit via: sbatch ...
# Check via: squeue -u $USER

#SBATCH -J Python

#SBATCH --account=fc_cosi
#SBATCH --partition=savio4_gpu
#SBATCH --qos=a5k_gpu4_normal

#SBATCH -t 24:00:00

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:A5000:8

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