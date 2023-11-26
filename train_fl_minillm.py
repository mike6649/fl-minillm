import torch
import deepspeed
import logging
import os
import time
import json
import torch.distributed as dist
from accelerate import init_empty_weights
import sys
import shutil
from pynvml import *

from transformers import (
    AutoModelForCausalLM,
    AutoConfig,
    ParallelOPTForCausalLM,
    ParallelGPTJForCausalLM,
    ParallelGPT2LMHeadModel,
    ParallelLlamaForCausalLM,
    )

import finetune
from fl.combined_clients import CombinedClients
from fl.fl_arguments import get_args
from fl.fl_fine_tuning import my_finetune, setup_fine_tuning
from fl.fl_minillm import train_minillm

parallel_model_map = {
    "opt": ParallelOPTForCausalLM,
    "gpt2": ParallelGPT2LMHeadModel,
    "gptj": ParallelGPTJForCausalLM,
    "llama": ParallelLlamaForCausalLM
}

from utils import print_args, initialize, load_parallel, get_tokenizer, print_rank

from minillm import train, Reward

deepspeed.utils.logger.setLevel(logging.WARNING)
# nvmlInit()
# h = nvmlDeviceGetHandleByIndex(0)
# info = nvmlDeviceGetMemoryInfo(h)
# print(f'total    : {info.total}')
# print(f'free     : {info.free}')
# print(f'used     : {info.used}')
            
def get_model(model_path, model_type, model_parallel, device):
    config = AutoConfig.from_pretrained(model_path)
    if model_parallel:
        config.is_model_parallel = True
        with init_empty_weights():
            model = parallel_model_map[model_type](config).half()
        load_parallel(model, model_path)
        model = model.to(device)
    else:
        config.is_model_parallel = False
        model = AutoModelForCausalLM.from_pretrained(model_path, config=config, device_map={"": device}, torch_dtype=torch.float16)
    model.eval()
    return model

def get_teacher_model(args, device):
    return get_model(args.teacher_model_path, args.model_type, args.model_parallel, device)

def get_student_model(args, device):
    return get_model(args.model_path, args.model_type, args.model_parallel, device)

def get_student_models(args, device, fl_round):
    return [get_model(os.path.join(args.save, str(student + 1), "_" + str(fl_round)), args.model_type, args.model_parallel, device) for student in range(args.num_clients)]

def fine_tune(model, args, tokenizer, finetune_dataset, ds_config):
    return my_finetune(args, model, tokenizer, finetune_dataset, ds_config)

def student2teacher_kd(students, teacher_model, args, tokenizer, ds_config, fl_round):
    print_rank("*" * 100)
    print_rank(f"Running student2teacher_kd...")
    # create megastudent
    megastudent = CombinedClients(students)
    reward = Reward(args, tokenizer, megastudent)

    prev_save = args.save
    args.save = os.path.join(args.save, str(0), str(fl_round))

    train_minillm(
        args=args,
        tokenizer=tokenizer,
        reward_fn=reward.reward_fn,
        teacher_model=megastudent,
        student_model=teacher_model,
        ds_config=ds_config,
        prompt_data=args.prompt_data_dir + str(0) + "/",
        eval_prompt_data=args.prompt_data_dir + str(0) + "/",
        lm_data=args.lm_data_dir,
        eval_lm_data=args.lm_data_dir,
    )

    args.save = prev_save

def teacher2student_kd(student_model, teacher_model, args, tokenizer, ds_config, fl_round, rank):

    reward = Reward(args, tokenizer, teacher_model)
    print_rank(f"teacher2student_kd to student @ rank {rank}...")

    prev_save = args.save
    args.save = os.path.join(args.save, str(rank), str(fl_round))

    train_minillm(
        args=args,
        tokenizer=tokenizer,
        reward_fn=reward.reward_fn,
        teacher_model=teacher_model,
        student_model=student_model,
        ds_config=ds_config,
        prompt_data=args.prompt_data_dir + str(0) + "/",
        eval_prompt_data=args.prompt_data_dir + str(0) + "/",
        lm_data=args.lm_data_dir,
        eval_lm_data=args.lm_data_dir,
    )

    args.save = prev_save

def setup_ds(args):
    with open(args.deepspeed_config, "r") as f:
        ds_config = json.load(f)

    ds_config["gradient_accumulation_steps"] = args.gradient_accumulation_steps
    ds_config["train_micro_batch_size_per_gpu"] = args.batch_size
    ds_config["gradient_clipping"] = args.clip_grad
    ds_config["steps_per_print"] = 10000000
    return ds_config

def setup_args():
    args = get_args()
    initialize(args)
    os.makedirs(args.save, exist_ok=True)
    if dist.get_rank() == 0:
        # print_args(args)
        with open(os.path.join(args.save, "args.json"), "w") as f:
            json.dump(vars(args), f)
            
    ds_config = setup_ds(args)
    args.deepspeed_config = None
    if args.teacher_model_type is None:
        args.teacher_model_type = args.model_type

    # do not save model
    args.save_interval = float("inf")
    # do not generate answers during eval
    args.eval_gen = False
    # do not eval during fine-tuning
    args.eval_interval = float("inf")
    args.log_interval = 1000
    args.mid_log_num = 0
    return ds_config, args

def waitFor(path):
    while not os.path.exists(path):
        time.sleep(5)

    while os.stat(path).st_size == 0:
        time.sleep(5)

    time.sleep(10)

def waitForStep(base_path, rank, step, fl_round):
    while not os.path.exists(os.path.join(base_path, str(rank), str(fl_round), f"completed_step_{step}.txt")):
        time.sleep(5)

    time.sleep(5)

def completeStep(base_path, rank, step, fl_round):
    file_path = os.path.join(base_path, str(rank), str(fl_round), f"completed_step_{step}.txt")
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w') as file:
        file.write("Step completed")

def removeDir(dir_to_remove):
    if os.path.exists(dir_to_remove) and os.path.isdir(dir_to_remove):
        shutil.rmtree(dir_to_remove)

def main():
    # sys.stdout.write("Launched process %d of %d on %s.\n" % (rank, size, name))

    ds_config, args = setup_args()
    device = torch.cuda.current_device()

    rank = args.fl_rank
    size = args.num_clients

    print_rank(f"Launched process {rank}\n")
    
    tokenizer = get_tokenizer(args)
    finetuning_args, fine_tune_dataset = setup_fine_tuning(args, tokenizer, rank)

    # Set to -1 if no previous checkpoint
    start_at = 0
    if start_at > -1:
        args.teacher_model_path = os.path.join(args.save, str(0), str(start_at))
        if rank > 0 : args.model_path = os.path.join(args.save, str(rank), str(start_at))

    for fl_round in range(start_at + 1, args.fl_rounds):
        if rank == 0 : print_rank("*" * 100)
        if rank == 0 : print_rank(f"FL MINILLM ROUND {fl_round + 1} / {args.fl_rounds}")
        if rank == 0 : print_rank("*" * 100)

        # Step 0: Load respective model

        student_model = get_student_model(args, device) if rank > 0 else None
        teacher_model = get_teacher_model(args, device)

        print_rank(f"STEP 0 COMPLETE @ RANK {rank} in ROUND {fl_round}")
        completeStep(args.save, rank, 0, fl_round)

        # Step 1: Fine tune seperately and save to results/rank/fl_round/

        finetuning_args.save = os.path.join(args.save, str(rank), "_" + str(fl_round))
        os.makedirs(finetuning_args.save, exist_ok=True)

        if rank > 0 : student_model = fine_tune(student_model, finetuning_args, tokenizer, fine_tune_dataset, ds_config)
        if rank < 1 : teacher_model = fine_tune(teacher_model, finetuning_args, tokenizer, fine_tune_dataset, ds_config)

        print_rank(f"STEP 1 COMPLETE @ RANK {rank} in ROUND {fl_round}")
        completeStep(args.save, rank, 1, fl_round)

        # Step 2: Load all clients when they are ready onto rank 0

        if rank == 0:
            for student in range(size):
                waitForStep(args.save, student + 1, 1, fl_round)

        print_rank(f"STEP 2 COMPLETE @ RANK {rank} in ROUND {fl_round}")
        completeStep(args.save, rank, 2, fl_round)

        # Step 3: Ensemble and Train teacher using MiniLLM

        if rank == 0 :
            student_models = get_student_models(args, device, fl_round)
            student2teacher_kd(student_models, teacher_model, args, tokenizer, ds_config, fl_round)

        print_rank(f"STEP 3 COMPLETE @ RANK {rank} in ROUND {fl_round}")
        completeStep(args.save, rank, 3, fl_round)

        # Step 4: Once teacher is ready, load teacher on all ranks

        args.teacher_model_path = os.path.join(args.save, str(0), str(fl_round))
        waitForStep(args.save, 0, 3, fl_round)
        teacher_model = get_teacher_model(args, device)

        print_rank(f"STEP 4 COMPLETE @ RANK {rank} in ROUND {fl_round}")
        completeStep(args.save, rank, 4, fl_round)

        # Step 5: Train Students seperately using MiniLLM and update path

        if rank > 0 : 
            teacher2student_kd(student_model, teacher_model, args, tokenizer, ds_config, fl_round, rank)
            args.model_path = os.path.join(args.save, str(rank), str(fl_round))

        print_rank(f"STEP 5 COMPLETE @ RANK {rank} in ROUND {fl_round}")
        completeStep(args.save, rank, 5, fl_round)

    # if (rank > 0): waitForStep(args.save, 0, 4, args.fl_rounds - 1)
    # for fl_round in range(args.fl_rounds - 1):
    #     removeDir(os.path.join(args.save, str(rank), str(fl_round)))
    #     removeDir(os.path.join(args.save, str(rank), "_" + str(fl_round)))

    # removeDir(os.path.join(args.save, str(rank), "_" + str(args.fl_rounds - 1)))

    # if rank == 0 : print_rank("*" * 100)
    # if rank == 0 : print_rank(f"FL MINILLM COMPLETE")
    # if rank == 0 : print_rank("*" * 100)

    print_rank(str(rank) + " is DONE")
    exit()
        
def test():
    ds_config, args = setup_args()
    device = torch.cuda.current_device()

    rank = args.fl_rank
    size = args.num_clients

    print_rank(f"Launched process {rank}\n")
    
    tokenizer = get_tokenizer(args)
    finetuning_args, fine_tune_dataset = setup_fine_tuning(args, tokenizer, rank)

    student_model = get_student_model(args, device) if rank > 0 else None
    print_rank(f"Done process {rank}\n")
    exit()
    
if __name__ == "__main__":
    main()

