import copy
import finetune
import arguments
import deepspeed
import torch
import torch.distributed as dist
from transformers import AutoTokenizer, mpu
from utils import get_model, print_rank, print_args


def setup_optimizer(model, args):
    optimizer = finetune.get_optimizer(args, model)
    lr_scheduler = finetune.get_learning_rate_scheduler(args, optimizer)
    model, optimizer, _, lr_scheduler = deepspeed.initialize(
        model=model,
        optimizer=optimizer,
        args=args,
        lr_scheduler=lr_scheduler,
        mpu=mpu if args.model_parallel else None,
    )
    return model, optimizer, lr_scheduler


def get_tokenizer(model_path, model_type):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if model_type in ["gpt2", "opt", "llama", "gptj"]:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return tokenizer


def my_finetune(args, model, tokenizer, dataset):
    """
    arguments:
    epoch, lr
    """
    print_rank("total_iters", args.total_iters)

    device = torch.cuda.current_device()
    model, optimizer, lr_scheduler = setup_optimizer(model, args)
    model = finetune.finetune(args, tokenizer, model, optimizer, lr_scheduler, dataset, device)
    return model


def setup_fine_tuning_args(args, dataset):
    fine_tuning_args = copy.deepcopy(args)
    args.epochs = args.fine_tune_epochs
    dp_world_size = mpu.get_data_parallel_world_size() if args.model_parallel else dist.get_world_size()
    args.train_iters_per_epoch = int(len(dataset["train"]) / (args.batch_size * dp_world_size * args.gradient_accumulation_steps))
    args.total_iters = args.train_iters_per_epoch * args.epochs
    return fine_tuning_args

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
    args.save_interval = None
    # do not generate answers during eval
    args.eval_gen = False
    args.eval_interval = None

    model = get_model(args, torch.cuda.current_device())
    model = my_finetune(args, model, tokenizer, dataset)

if __name__ == "__main__":
    main()
