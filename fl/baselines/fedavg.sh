#! /bin/bash

NUM_CLIENTS=2
FINE_TUNE_EPOCHS=2
FL_ROUNDS=2

PID_FILE="slurm_pids_$SLURM_JOB_ID.txt"

> "$PID_FILE"
sleep 1

# Write the current task's PID to the file
echo $SLURM_TASK_PID >> "$PID_FILE"

# Wait to allow all tasks to write their PIDs
sleep 10

# Read all PIDs and find the minimum
MIN_PID=$(sort -n "$PID_FILE" | head -n 1)

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

# model
# default to working directory
BASE_PATH=${1-"."}

CKPT_NAME="distilgpt-student"
CKPT=${2-"gpt2"}

# data
PROMPT_DATA_DIR="${BASE_PATH}/fl_processed_data/dolly/prompt/gpt2/"
# runtime
SAVE_PATH="${BASE_PATH}/results/distilgpt2/train/fl-minillm/"
# hp
GRAD_ACC=1
BATCH_SIZE=4
CHUNK_SIZE=16


OPTS=""
OPTS+=" --fl-rank ${NEW_RANK}"
OPTS+=" --num-clients ${NUM_CLIENTS}"
OPTS+=" --fine-tune-epochs ${FINE_TUNE_EPOCHS}"
OPTS+=" --fl-rounds ${FL_ROUNDS}"
# model
OPTS+=" --base-path ${BASE_PATH}"
OPTS+=" --model-path ${CKPT}"
OPTS+=" --ckpt-name ${CKPT_NAME}"
# OPTS+=" --gradient-checkpointing"
# data
OPTS+=" --prompt-data-dir ${PROMPT_DATA_DIR}"
OPTS+=" --dev-num 1000"
OPTS+=" --num-workers 0"
# hp
OPTS+=" --total-iters 5000"
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

CMD="python3 ${BASE_PATH}/fl/baselines/fedavg_client.py ${OPTS} $@"

echo ${CMD}
echo "PYTHONPATH=${PYTHONPATH}"
mkdir -p ${SAVE_PATH}

${CMD}
