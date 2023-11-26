#!/bin/bash
#SBATCH -A m3863_g
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t 2:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=4
#SBATCH -c 32
#SBATCH --gpus-per-task=1

export HF_HOME=$SCRATCH
export HF_DATASETS_CACHE=$SCRATCH
export TRANSFORMERS_CACHE=$SCRATCH

cd $SCRATCH/smr/fl-minillm

module load conda
conda activate sysml
srun bash fl/fl_train.sh 1