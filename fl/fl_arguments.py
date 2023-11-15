import argparse
from arguments import *

def add_fl_args(parser):
    group = parser.add_argument_group('fl', 'federated learning config')
    group.add_argument("--fine-tune-epochs", type=int, default=1)
    group.add_argument("--num-clients", type=int, default=2)
    group.add_argument("--fl-rounds", type=int, default=2)
    return parser

def get_args():
    parser = argparse.ArgumentParser()
    parser = add_model_args(parser)
    parser = add_runtime_args(parser)
    parser = add_data_args(parser)
    parser = add_hp_args(parser)
    parser = add_ppo_args(parser)
    parser = add_minillm_args(parser)
    parser = add_gen_args(parser)
    parser = add_peft_args(parser)
    parser = add_fl_args(parser)
    parser = deepspeed.add_config_arguments(parser)
    
    args, unknown = parser.parse_known_args()
    
    assert all(["--" not in x for x in unknown]), unknown
    
    args.local_rank = int(os.getenv("LOCAL_RANK", "0"))
        
    args.n_gpu = args.n_gpu * args.n_nodes
        
    if args.type == "eval_main":
        ckpt_name = None
        if args.ckpt_name is not None:
            ckpt_name = args.ckpt_name
        if args.peft_name is not None:
            ckpt_name = args.peft_name

        if ckpt_name is not None:
            tmp = ckpt_name.split("/")
            if tmp[-1].isdigit():
                ckpt_name = "_".join(tmp[:-1]) + "/" + tmp[-1]
            else:
                ckpt_name = "_".join(tmp)

        save_path = os.path.join(
            args.save,
            f"{args.data_names}-{args.max_length}" + (f"-mp{args.model_parallel_size}" if args.model_parallel > 0 else ""),
            ckpt_name,
            f"{args.seed}",
        )
        args.save = save_path
    elif args.type == "lm":
        save_path = os.path.join(
            args.save,
            (f"{args.ckpt_name}" + f"-{args.peft_name}" if args.peft_name is not None else ""),
            (f"e{args.epochs}-bs{args.batch_size}-lr{args.lr}-G{args.gradient_accumulation_steps}-N{args.n_gpu}-NN{args.n_nodes}") + \
            (f"-mp{args.model_parallel_size}" if args.model_parallel > 0 else "") + \
            (f"-lora-{args.peft_lora_r}-{args.peft_lora_alpha}-{args.peft_lora_dropout}" if args.peft == "lora" else "") + \
            args.save_additional_suffix
        )
        args.save = save_path
    elif args.type == "kd":
        save_path = os.path.join(
            args.save,
            (f"{args.ckpt_name}" + f"-{args.peft_name}" if args.peft_name is not None else "" + \
             f"-{args.teacher_ckpt_name}" + f"-{args.teacher_peft_name}" if args.teacher_peft_name is not None else ""),
            (f"e{args.epochs}-bs{args.batch_size}-lr{args.lr}-G{args.gradient_accumulation_steps}-N{args.n_gpu}-NN{args.n_nodes}-kd{args.kd_ratio}") + \
            (f"-mp{args.model_parallel_size}" if args.model_parallel > 0 else "") + \
            (f"-lora-{args.peft_lora_r}-{args.peft_lora_alpha}-{args.peft_lora_dropout}" if args.peft == "lora" else "") + \
            args.save_additional_suffix
        )
        args.save = save_path
    elif args.type == "gen":
        save_path = os.path.join(
            args.save,
            (f"{args.ckpt_name}"),
            (f"t{args.temperature}-l{args.max_length}"),
        )
        args.save = save_path
    elif args.type == "minillm":
        ppo_prefix = f"pe{args.ppo_epochs}" + \
                     (f"_rs{args.reward_scaling}" if args.ppo_epochs is not None else "") + \
                     (f"_nr{args.num_rollouts}" if args.num_rollouts is not None else "") + \
                     (f"_ln" if args.length_norm else "") + \
                     (f"_sr" if args.single_step_reg else "") + \
                     (f"_tm{args.teacher_mixed_alpha}" if args.teacher_mixed_alpha is not None else "")
        save_path = os.path.join(
            args.save,
            (f"{args.ckpt_name}" + f"-{args.peft_name}" if args.peft_name is not None else "" + \
             f"-{args.teacher_ckpt_name}" + f"-{args.teacher_peft_lora_name}" if hasattr(args, "teacher_peft_lora_name") else ""),
            (f"bs{args.batch_size}-lr{args.lr}-G{args.gradient_accumulation_steps}-N{args.n_gpu}-NN{args.n_nodes}-lm{args.lm_coef}-len{args.max_length}" + \
                (f"-mp{args.model_parallel_size}" if args.model_parallel > 0 else "")) + \
            (f"-lora-{args.peft_lora_r}-{args.peft_lora_alpha}-{args.peft_lora_dropout}" if args.peft == "lora" else ""),
            ppo_prefix + args.save_additional_suffix
        )
        args.save = save_path
        
        if args.warmup_iters > 0:
            assert args.scheduler_name is not None

    return args
