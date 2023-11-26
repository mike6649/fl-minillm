export HF_HOME=$SCRATCH
export HF_DATASETS_CACHE=$SCRATCH
export TRANSFORMERS_CACHE=$SCRATCH

cd $SCRATCH/smr/fl-minillm

module load conda
conda activate sysml
srun bash fl/fl_train.sh 1