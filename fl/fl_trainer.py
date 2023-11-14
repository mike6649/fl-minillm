import os
import torch
import torch.distributed as dist

from transformers import (
    AutoTokenizer,
    GenerationConfig,
    mpu)

from minillm.trainer import PPOTrainer
from minillm.storages import PPORolloutStorage
from minillm.losses import Loss


class FLPPOTrainer(PPOTrainer):
    """
    RL model trainer with an `accelerate` based backend
    """

    def __init__(self, args, tokenizer: AutoTokenizer, reward_fn, ds_config, teacher_model, student_model):
        self.args = args
        self.max_length = args.max_length
        self.ds_config = ds_config
        self.reward_fn = reward_fn
        self.device = torch.cuda.current_device()

        if int(os.environ.get("WORLD_SIZE", 1)) > 1:
            dist.barrier(device_ids=[int(os.environ.get("LOCAL_RANK", 0))])

        if args.model_parallel:
            self.dp_world_size = mpu.get_data_parallel_world_size()
            self.dp_rank = mpu.get_data_parallel_rank()
            self.dp_group = mpu.get_data_parallel_group()
        else:
            self.dp_world_size = dist.get_world_size()
            self.dp_rank = dist.get_rank()
            self.dp_group = None

        self.model = student_model
        if args.model_parallel:
            if mpu.get_data_parallel_rank() == 0:
                print(' > number of parameters on model parallel rank {}: {}M'.format(
                    mpu.get_model_parallel_rank(),
                    int(sum([p.nelement() for p in self.model.parameters()]) / 1e6)), flush=True)
        else:
            if dist.get_rank() == 0:
                print(' > number of parameters: {}M'.format(
                    int(sum([p.nelement() for p in self.model.parameters()]) / 1e6)), flush=True)

        self.sampler = None
        self.teacher_model = teacher_model
        self.opt = self.setup_optimizer()
        self.scheduler = self.setup_scheduler()
        self.model, self.opt, self.scheduler = self.setup_ds(self.model, self.opt, self.scheduler)
        
        self.tokenizer = tokenizer
        self.store = PPORolloutStorage(self.tokenizer.pad_token_id, self.args.seed_ppo + self.dp_rank)
        self.store.clear_history()
        
        self.losses = Loss(args, self)
        self.generate_kwargs = dict(
            do_sample=args.do_sample,
            top_p=args.top_p,
            top_k=args.top_k,
            temperature=args.temperature,
            max_length=args.max_length,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
        )
