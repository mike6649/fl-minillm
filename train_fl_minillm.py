import torch
import os
import json
import torch.distributed as dist
from accelerate import init_empty_weights

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

from utils import print_args, initialize, load_parallel, get_tokenizer

from minillm import train, Reward

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

def get_server_model(args, device):
    return get_model(args.teacher_model_path, args.model_type, args.model_parallel, device)

def get_client_models(args, device):
    return [get_model(args.model_path, args.model_type, args.model_parallel, device) for _ in range(args.num_clients)]

def client_fine_tune(clients, args, tokenizer, finetune_dataset, ds_config):
    # fine tune sequentially because i dont care
    for client_model in clients:
        my_finetune(args, client_model, tokenizer, finetune_dataset, ds_config)


def client2server_kd(clients, server_model, args, tokenizer, ds_config):
    # create megaclient
    megaclient = CombinedClients(clients)

    reward = Reward(args, tokenizer, megaclient)
    
    train_minillm(
        args=args,
        tokenizer=tokenizer,
        reward_fn=reward.reward_fn,
        teacher_model=megaclient,
        student_model=server_model,
        ds_config=ds_config,
        prompt_data=args.prompt_data_dir,
        eval_prompt_data=args.prompt_data_dir,
        lm_data=args.lm_data_dir,
        eval_lm_data=args.lm_data_dir,
    )

def server2client_kd(client_models, server_model, args, tokenizer, ds_config):
    reward = Reward(args, tokenizer, server_model)
    
    for client_model in client_models:
        train_minillm(
            args=args,
            tokenizer=tokenizer,
            reward_fn=reward.reward_fn,
            teacher_model=server_model,
            student_model=client_model,
            ds_config=ds_config,
            prompt_data=args.prompt_data_dir,
            eval_prompt_data=args.prompt_data_dir,
            lm_data=args.lm_data_dir,
            eval_lm_data=args.lm_data_dir,
        )

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
        print_args(args)
        with open(os.path.join(args.save, "args.json"), "w") as f:
            json.dump(vars(args), f)
            
    ds_config = setup_ds(args)
    args.deepspeed_config = None
    if args.teacher_model_type is None:
        args.teacher_model_type = args.model_type

    # do not save model
    args.save_interval = None
    # do not generate answers during eval
    args.eval_gen = False
    # do not eval during fine-tuning
    args.eval_interval = None

    return ds_config, args

def main():
    ds_config, args = setup_args()
    device = torch.cuda.current_device()
    
    server_model = get_server_model(args, device)
    client_models = get_client_models(args, device)
    
    tokenizer = get_tokenizer(args)
    finetuning_args, fine_tune_dataset = setup_fine_tuning(args, tokenizer)

    for round in range(args.fl_rounds):
        # first fine-tune the clients
        client_fine_tune(client_models, args, tokenizer, fine_tune_dataset, ds_config)

        # then KD to the server
        client2server_kd(client_models, server_model, args, tokenizer, ds_config)

        # then server KD to the clients
        server2client_kd(client_models, server_model, args, tokenizer, ds_config)
    
    # TODO some kind of evaluation

if __name__ == "__main__":
    main()
