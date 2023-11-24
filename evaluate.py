import time
import os

import torch
import torch.distributed as dist
from data_utils.prompt_datasets import PromptDataset
import deepspeed

import json

from transformers import AutoConfig, AutoModelForCausalLM, ParallelGPT2LMHeadModel, AutoTokenizer

from arguments import get_args
from gpt4_evaluate import get_metrics_gpt
from rouge_metric import compute_metrics

from utils import initialize, print_args
from utils import print_rank
from utils import save_rank
from utils import get_tokenizer, get_model, load_parallel

from evaluate_main import evaluate_main, prepare_dataset_main, run_model
from accelerate import load_checkpoint_and_dispatch, init_empty_weights

torch.set_num_threads(4)


def load_model(model_path, device, model_parallel=True):
    config = AutoConfig.from_pretrained(model_path)
    if model_parallel:
        config.is_model_parallel = True
        with init_empty_weights():
            model = ParallelGPT2LMHeadModel(config).half()
        load_parallel(model, model_path)
    else:
        config.is_model_parallel = False
        model = AutoModelForCausalLM.from_pretrained(model_path, config=config, device_map={"": device}, torch_dtype=torch.float16)
    return model

def load_tokenizer(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    return tokenizer

def _inner_evaluate(args, tokenizer, model, dataset: PromptDataset, device):
    lm_loss, query_ids, response_ids = run_model(args, tokenizer, model, dataset, device)
    query_strs = tokenizer.batch_decode(query_ids, skip_special_tokens=True)
    response_strs = tokenizer.batch_decode(response_ids, skip_special_tokens=True)
    all_preds = [[]]
    for q, r in zip(query_strs, response_strs):
        all_preds[0].append((q, q + r))
    all_responses = [] 
    for p in all_preds[0]:
        q, r = p
        r = r[len(q):]
        idx = r.find("<|endoftext|>")
        if idx >= 0:
            r = r[:idx]
        all_responses.append(r.replace("<n>", "\n").strip())
    gen_res: dict = compute_metrics(all_responses, dataset.answers)
    # {"exact_match": em, "rougeL": rougeL}
    gen_res["loss"] = lm_loss
    # TODO gpt4 score
    # get_metrics_gpt()
    gen_res = {k: round(v, 4) for k, v in gen_res.items()}
    return gen_res


def evaluate(model_path, device, data_dir, args, tokenizer=None):
    if tokenizer is None:
        tokenizer = load_tokenizer(model_path)
    dataset = PromptDataset(args, tokenizer, "valid", data_dir, args.dev_num)
    model = load_model(model_path, device, args.model_parallel)
    return _inner_evaluate(args, tokenizer, model, dataset, device)

def main():
    torch.backends.cudnn.enabled = False
    
    args = get_args()
    initialize(args)
    
    if dist.get_rank() == 0:
        print_args(args)
        with open(os.path.join(args.save, "args.json"), "w") as f:
            json.dump(vars(args), f)
    
    device = torch.cuda.current_device()
    cur_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    save_rank("\n\n" + "="*30 + f" EXP at {cur_time} " + "="*30, os.path.join(args.save, "log.txt"))
    print("OK")
    with open(args.deepspeed_config, "r") as f:
        ds_config = json.load(f)

    ds_config["gradient_accumulation_steps"] = args.gradient_accumulation_steps
    ds_config["train_micro_batch_size_per_gpu"] = args.batch_size
    ds_config["gradient_clipping"] = args.clip_grad
    ds_config["steps_per_print"] = args.gradient_accumulation_steps
    
    if not args.do_train:
        ds_config["zero_optimization"]["stage"] = 0

    args.deepspeed_config = None

    # get the tokenizer
    tokenizer = get_tokenizer(args)
    dataset = PromptDataset(args, tokenizer, "valid", args.data_dir, args.dev_num)
    model = load_model(args.model_path, device, args.model_parallel)
    
    if args.type == "eval_main":
        evaluate_main(args, tokenizer, model, dataset, device)
    else:
        raise NotImplementedError
    
    
if __name__ == "__main__":
    main()