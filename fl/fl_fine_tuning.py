import copy
import finetune
import arguments
import deepspeed
import torch
import torch.distributed as dist
from transformers import AutoTokenizer, mpu
from utils import get_model, print_rank, print_args


def setup_optimizer(model, args, ds_config):
    optimizer = finetune.get_optimizer(args, model)
    lr_scheduler = finetune.get_learning_rate_scheduler(args, optimizer)
    model, optimizer, _, lr_scheduler = deepspeed.initialize(
        model=model,
        optimizer=optimizer,
        args=args,
        lr_scheduler=lr_scheduler,
        config=ds_config,
        mpu=mpu if args.model_parallel else None,
    )
    return model, optimizer, lr_scheduler


def get_tokenizer(model_path, model_type):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if model_type in ["gpt2", "opt", "llama", "gptj"]:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return tokenizer


def my_finetune(args, model, tokenizer, dataset, ds_config=None):
    device = torch.cuda.current_device()
    model, optimizer, lr_scheduler = setup_optimizer(model, args, ds_config)
    model = finetune.finetune(args, tokenizer, model, optimizer, lr_scheduler, dataset, device)
    return model


def setup_fine_tuning(all_args, tokenizer):
    args = copy.deepcopy(all_args)
    args.epochs = args.fine_tune_epochs
    args.do_train = True
    
    args.data_dir = args.prompt_data_dir
    dataset = finetune.prepare_dataset(args, tokenizer)   # TODO this should be 10 separate datasets or whathaveyou
    dp_world_size = mpu.get_data_parallel_world_size() if args.model_parallel else dist.get_world_size()
    args.train_iters_per_epoch = int(len(dataset["train"]) / (args.batch_size * dp_world_size * args.gradient_accumulation_steps))
    args.total_iters = args.train_iters_per_epoch * args.epochs
    return args, dataset

def main():
    args = arguments.get_args()
    print_args(args)
    dp_world_size = mpu.get_data_parallel_world_size() if args.model_parallel else dist.get_world_size()
    args.train_iters_per_epoch = int(len(dataset["train"]) / (args.batch_size * dp_world_size * args.gradient_accumulation_steps))
    args.total_iters = args.train_iters_per_epoch * args.epochs
    
    # mpu setup
    finetune.initialize(args)
    tokenizer = get_tokenizer(args.model_path, args.model_type)
    dataset = finetune.prepare_dataset(args, tokenizer)
    
    # do not save model
    args.save_interval = float("inf")
    # do not generate answers during eval
    args.eval_gen = False
    args.eval_interval = float("inf")

    model = get_model(args, torch.cuda.current_device())
    model = my_finetune(args, model, tokenizer, dataset)

if __name__ == "__main__":
    main()
