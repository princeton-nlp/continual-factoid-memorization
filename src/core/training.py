import os
import math
import uuid
import yaml
import time
import random
import argparse
import functools
from datetime import datetime

import transformers
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    FullStateDictConfig,
    StateDictType,
)

from pynvml import (
    nvmlInit,
    nvmlDeviceGetHandleByIndex,
    nvmlDeviceGetMemoryInfo,
)

import wandb
import numpy as np
from rich import print


def print_gpu_usage():
    nvmlInit()
    n_gpus = torch.cuda.device_count()

    print('========== GPU Utilization ==========')
    for gpu_id in range(n_gpus):
        h = nvmlDeviceGetHandleByIndex(gpu_id)
        info = nvmlDeviceGetMemoryInfo(h)
        print(f'GPU {gpu_id}')
        print(f'- Used:       {info.used / 1024 ** 3:>8.2f} B ({info.used / info.total * 100:.1f}%)')
        print(f'- Available:  {info.free / 1024 ** 3:>8.2f} B ({info.free / info.total * 100:.1f}%)')
        print(f'- Total:      {info.total / 1024 ** 3:>8.2f} B')
    print('=====================================')


def print_rank0(msg):
    if dist.get_rank() == 0:
        print(msg)


def setup_model(model_name, max_seq_length):
    config = transformers.AutoConfig.from_pretrained(
        model_name,
        token=os.environ['HF_TOKEN'],
    )
    config.use_cache = False

    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_name,
        token=os.environ['HF_TOKEN'],
        config=config,
        torch_dtype=torch.bfloat16,
        attn_implementation='eager', #'flash_attention_2',
    )

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_name,
        model_max_seq_length=max_seq_length,
        padding_side='right',
        use_fast=False,
        token=os.environ['HF_TOKEN'],
        trust_remote_code=True,
        add_prefix_space=True,
    )
    return model, tokenizer


@torch.inference_mode()
def run_evaluation(model, eval_dataloader, local_rank):
    """
    Returns the averaged loss reduced across all processes.
    """
    model.eval()
    print_rank0('Running eval for loss...')

    losses = 0
    for step, batch in enumerate(eval_dataloader):
        inputs = {k: v.to(model.device) for k, v in batch.items() if k in ('input_ids', 'attention_mask', 'labels')}
        outputs = model(**inputs)
        loss = outputs.loss
        losses += loss.float()
    losses = losses / len(eval_dataloader)
    eval_loss = get_all_reduce_mean(losses.clone()).item()
    return eval_loss


def get_dataloader(dataset_split, dataset, world_size, local_rank, seed, collator, batch_size):
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=local_rank,
        shuffle=True if dataset_split == 'train' else False,
        seed=seed,
    )
    loader = DataLoader(
        dataset,
        shuffle=False,
        pin_memory=True,
        drop_last=True if dataset_split == 'train' else False,
        batch_size=batch_size,
        collate_fn=collator,
        sampler=sampler,
    )
    return sampler, loader


def get_parameter_names(model, forbidden_layer_types):
    result = []
    for name, child in model.named_children():
        result += [
            f'{name}.{n}'
            for n in get_parameter_names(child, forbidden_layer_types)
            if not isinstance(child, tuple(forbidden_layer_types))
        ]
    result += list(model._parameters.keys())
    return result


def get_optimizer(model, lr, weight_decay):
    decay_parameters = get_parameter_names(model, [torch.nn.LayerNorm])
    decay_parameters = [name for name in decay_parameters if 'bias' not in name]
    optimizer_grouped_parameters = [
        {
            'params': [p for n, p in model.named_parameters() if (n in decay_parameters and p.requires_grad)],
            'weight_decay': weight_decay,
        },
        {
            'params': [p for n, p in model.named_parameters() if (n not in decay_parameters and p.requires_grad)],
            'weight_decay': 0.0,
        },
    ]
    return torch.optim.AdamW(
        params=optimizer_grouped_parameters,
        lr=lr,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=weight_decay,
    )


def log_stats(global_step, total_training_steps, epoch, loss, grad_norm, scheduler, use_wandb):
    lr = scheduler.get_last_lr()[0]
    if use_wandb:
        wandb.log(dict(loss=loss, global_step=global_step, lr=lr, grad_norm=grad_norm))
    print_rank0(f'\[train] global_step={global_step} / {total_training_steps} | epoch={epoch} | loss={loss:.6f} | lr={lr:.10f} | grad_norm={grad_norm:.6f}')


def get_all_reduce_mean(tensor):
    torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
    tensor = tensor / torch.distributed.get_world_size()
    return tensor


def save_model(local_rank, model, tokenizer, run_dir, ckpt_name):
    save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, save_policy):
        cpu_state = model.state_dict()

    ckpt_dir = os.path.join(run_dir, ckpt_name)
    print_rank0(f'Saving model to: {ckpt_dir}')
    if local_rank == 0:
        model.save_pretrained(ckpt_dir, state_dict=cpu_state)
        tokenizer.save_pretrained(ckpt_dir)


def get_args():
    from src.config import RUN_BASE_DIR

    parser = argparse.ArgumentParser()
    parser.add_argument('--run_base_dir', type=str, default=RUN_BASE_DIR)
    parser.add_argument('--run_name', type=str, default='try')
    parser.add_argument('--run_dir', type=str, default=None)

    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--use_wandb', action='store_true')
    parser.add_argument('--show_gpu_usage', action='store_true')
    parser.add_argument('--eval_every_n_steps', type=int, default=1000000000)
    parser.add_argument('--save_every_n_steps', type=int, default=1000000000)

    parser.add_argument('--model_name', type=str, default='meta-llama/Llama-2-7b-hf')
    parser.add_argument('--base_model_name', type=str, default=None)
    parser.add_argument('--max_seq_length', type=int, default=2048)

    parser.add_argument('--lr', type=float, default=2e-05)
    parser.add_argument('--train_batch_size', type=int, default=16)
    parser.add_argument('--eval_batch_size', type=int, default=16)
    parser.add_argument('--scheduler_type', type=str, default='cosine')
    parser.add_argument('--warmup_ratio', type=float, default=0.05)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--gradient_clipping', type=float, default=1.0)
    parser.add_argument('--gradient_checkpointing', action='store_true')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)

    parser.add_argument('--dataset_name', type=str, default='cot')
    parser.add_argument('--dataset_module_path', type=str, default='core.data.datasets')
    parser.add_argument('--num_epochs', type=int, default=1)
    parser.add_argument('--train_num_examples', type=int, default=1000)
    parser.add_argument('--dev_num_examples', type=int, default=100)
    parser.add_argument('--use_train_for_dev', action='store_true')
    args = parser.parse_args()
    return args
