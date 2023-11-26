#! /bin/bash

MASTER_ADDR=localhost
# MASTER_PORT=${3-2012}
NNODES=1
NODE_RANK=0
GPUS_PER_NODE=${1-2}

PID_FILE="slurm_pids_$SLURM_JOB_ID.txt"

> "$PID_FILE"
sleep 1

# Write the current task's PID to the file
echo $SLURM_TASK_PID >> "$PID_FILE"

# Wait to allow all tasks to write their PIDs
sleep 10

# Read all PIDs and find the minimum
MIN_PID=$(sort -n "$PID_FILE" | head -n 1)

NEW_MASTER_PORT=$(($SLURM_TASK_PID % 65536))
NEW_RANK=$(($SLURM_TASK_PID % $MIN_PID))

# Read all PIDs, sort them, and find the rank of the current task's PID
# Save the sorted PIDs in an array
readarray -t sorted_pids < <(sort -n "$PID_FILE")

# Find the rank of the current task
for i in "${!sorted_pids[@]}"; do
    if [[ "${sorted_pids[$i]}" == "$SLURM_TASK_PID" ]]; then
        NEW_RANK=$i
        break
    fi
done

# echo $NEW_RANK

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE \
                  --nnodes $NNODES \
                  --node_rank $NODE_RANK \
                  --master_addr $MASTER_ADDR \
                  --master_port $NEW_MASTER_PORT"

# model
# default to working directory
BASE_PATH=${2-"."}

CKPT_NAME="flminillm-student"
CKPT=${3-"${BASE_PATH}/checkpoints/gpt2/"}

TEACHER_CKPT_NAME="flminillm-teacher"
TEACHER_CKPT=${4-"${BASE_PATH}/checkpoints/gpt2-large/"}

# data
PROMPT_DATA_DIR="${BASE_PATH}/processed_data/dolly/prompt/gpt2/"
LM_DATA_DIR="${BASE_PATH}/processed_data/openwebtext/gpt2/512/1M/"
# runtime
SAVE_PATH="${BASE_PATH}/results/gpt2/train/fl-minillm/"
# hp
GRAD_ACC=1
BATCH_SIZE=4
CHUNK_SIZE=16


OPTS=""
OPTS+=" --fl-rank ${NEW_RANK}"
OPTS+=" --num-clients 3"
OPTS+=" --fl-rounds 10"
OPTS+=" --epochs 10"
OPTS+=" --fine-tune-epochs 10"
# model
OPTS+=" --base-path ${BASE_PATH}"
OPTS+=" --model-path ${CKPT}"
OPTS+=" --teacher-model-path ${TEACHER_CKPT}"
OPTS+=" --ckpt-name ${CKPT_NAME}"
OPTS+=" --teacher-ckpt-name ${TEACHER_CKPT_NAME}"
OPTS+=" --n-gpu ${GPUS_PER_NODE}"
OPTS+=" --teacher-model-fp16"
# OPTS+=" --gradient-checkpointing"
# data
OPTS+=" --prompt-data-dir ${PROMPT_DATA_DIR}"
OPTS+=" --lm-data-dir ${LM_DATA_DIR}"
OPTS+=" --dev-num 1000"
OPTS+=" --num-workers 0"
# hp
OPTS+=" --total-iters 5000"
OPTS+=" --kd-ratio 0.5"
OPTS+=" --batch-size ${BATCH_SIZE}"
OPTS+=" --lr 5e-6"
OPTS+=" --lr-min 5e-6"
OPTS+=" --gradient-accumulation-steps ${GRAD_ACC}"
OPTS+=" --max-length 512"
OPTS+=" --max-prompt-length 256"
OPTS+=" --warmup-iters 100"
# runtime
OPTS+=" --save ${SAVE_PATH}"
OPTS+=" --seed 10"
OPTS+=" --seed-ppo 42"
OPTS+=" --seed-lm 7"
OPTS+=" --save-interval 500"
OPTS+=" --eval-interval 100"
OPTS+=" --log-interval 16"
OPTS+=" --mid-log-num 1"
# ppo
OPTS+=" --type minillm"
OPTS+=" --ppo-epochs 4"
OPTS+=" --num-rollouts 16"
OPTS+=" --chunk-size ${CHUNK_SIZE}"
# minillm
OPTS+=" --length-norm"
OPTS+=" --single-step-reg"
OPTS+=" --teacher-mixed-alpha 0.2"
# reward
OPTS+=" --reward-scaling 0.5"
OPTS+=" --cliprange-reward 100"
# gen
OPTS+=" --do-sample"
OPTS+=" --top-k 0"
OPTS+=" --top-p 1.0"
OPTS+=" --temperature 1.0"
# deepspeed
OPTS+=" --deepspeed"
OPTS+=" --deepspeed_config ${BASE_PATH}/configs/deepspeed/ds_config.json"

export NCCL_DEBUG=""
export WANDB_DISABLED=True
export TF_CPP_MIN_LOG_LEVEL=3
export PYTHONPATH=${BASE_PATH}

CMD="torchrun ${DISTRIBUTED_ARGS} ${BASE_PATH}/train_fl_minillm.py ${OPTS} $@"
# CMD="python test.py"

echo ${CMD}
echo "PYTHONPATH=${PYTHONPATH}"
mkdir -p ${SAVE_PATH}

${CMD}
