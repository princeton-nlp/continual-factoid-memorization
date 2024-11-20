import os
import math
import yaml
import time
import copy
import random
import argparse
import functools
from datetime import datetime
from string import ascii_uppercase

import torch
import torch.distributed as dist
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.distributed.fsdp.fully_sharded_data_parallel import ShardingStrategy

import transformers
from transformers.models.mistral.modeling_mistral import MistralDecoderLayer
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from transformers.models.gemma.modeling_gemma import GemmaDecoderLayer
from transformers.models.gemma2.modeling_gemma2 import Gemma2DecoderLayer

import wandb
import numpy as np
from tqdm import tqdm
from rich import print
from src.core.training import (
    print_gpu_usage,
    print_rank0,
    setup_model,
    run_evaluation,
    get_dataloader,
    get_parameter_names,
    get_optimizer,
    log_stats,
    get_all_reduce_mean,
    save_model,
)
from src.core.data.sft import FinetuningDataManager
from src.datasets import get_collator
from src.config import RUN_BASE_DIR, DATA_DIR

os.environ['WANDB_MODE'] = 'dryrun'
os.environ['WANDB_PROJECT'] = 'ft'



def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_base_dir', type=str, default=RUN_BASE_DIR)
    parser.add_argument('--run_name', type=str, default='try')
    parser.add_argument('--run_dir', type=str, default=None)

    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--wandb_name', type=str, default='none')
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
    parser.add_argument('--mix_ratio', type=float, default=None)

    parser.add_argument('--dataset_name', type=str, default='cot')
    parser.add_argument('--task_mode', type=str, default='none')
    parser.add_argument('--dataset_module_path', type=str, default='core.data.datasets')
    parser.add_argument('--num_epochs', type=int, default=1)
    parser.add_argument('--train_num_examples', type=int, default=1000)
    parser.add_argument('--dev_num_examples', type=int, default=100)
    parser.add_argument('--use_train_for_dev', action='store_true')
    parser.add_argument('--stopping_condition', type=str, default='sc=fixed', help='If sc=loss, stop training when training loss < 0.0001 (3 times) where max epoch is set to 100.')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    start_time = time.time()
    local_rank = int(os.environ['LOCAL_RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.cuda.set_device(local_rank)
    dist.init_process_group('nccl', rank=local_rank, world_size=world_size)

    args = get_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    transformers.set_seed(args.seed)

    if args.base_model_name is None:
        args.base_model_name = args.model_name
    
    # Early stopping condition ################################################
    if args.stopping_condition == 'sc=loss':
        stopping_counter = 0
        max_stopping_counts = 10
        stopping_loss = 0.0001
        stop_flag = torch.zeros(1).to(torch.cuda.current_device())
    ###########################################################################

    if 'llama' in args.base_model_name.lower():
        transformer_layer_cls = {LlamaDecoderLayer}
    elif 'mistral' in args.base_model_name.lower():
        transformer_layer_cls = {MistralDecoderLayer}
    elif 'gemma' in args.base_model_name.lower():
        transformer_layer_cls = {Gemma2DecoderLayer}
    else:
        raise ValueError(f'transformer_layer_cls of {args.model_name} not specified.')

    if args.run_dir is None:
        args.run_dir = os.path.join(args.run_base_dir, args.run_name)
    else:
        assert os.path.join(args.run_base_dir, args.run_name) == args.run_dir
    os.makedirs(args.run_dir, exist_ok=True)
    args.date = datetime.now().strftime('%Y-%m-%d-%I_%M_%S_%p')
    args.total_batch_size = args.train_batch_size * world_size * args.gradient_accumulation_steps
    print_rank0(vars(args))

    model, tokenizer = setup_model(args.model_name, args.max_seq_length)
    num_params = sum([p.numel() for p in model.parameters()])
    auto_wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls=transformer_layer_cls,
    )

    fsdp_config = dict(
        auto_wrap_policy=auto_wrap_policy,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        device_id=torch.cuda.current_device(),
        mixed_precision=MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.bfloat16,
            buffer_dtype=torch.bfloat16,
        ),
        backward_prefetch=None,
        param_init_fn=None,
        cpu_offload=None,
    )

    model = FSDP(model, **fsdp_config)
    optimizer = get_optimizer(model, args.lr, args.weight_decay)

    finetuning_data_manager = FinetuningDataManager(
        dataset_name=args.dataset_name,
        dataset_module_path=args.dataset_module_path,
        tokenizer=tokenizer,
        max_seq_length=args.max_seq_length,
        train_num_examples=args.train_num_examples,
        dev_num_examples=args.dev_num_examples,
        use_train_for_dev=args.use_train_for_dev,
        task_mode=args.task_mode,
        mix_ratio=args.mix_ratio,
        base_model_name=args.base_model_name,
    )
    if local_rank == 0:
        finetuning_data_manager.print_config()

    train_dataset = finetuning_data_manager.get_dataset('train')
    eval_dataset = finetuning_data_manager.get_dataset('dev')

    collator = get_collator(args.task_mode, finetuning_data_manager)

    train_sampler, train_loader = get_dataloader(
        dataset_split='train',
        dataset=train_dataset,
        world_size=world_size,
        local_rank=local_rank,
        seed=args.seed,
        collator=collator,
        batch_size=args.train_batch_size,
    )
    eval_sampler, eval_loader = get_dataloader(
        dataset_split='eval',
        dataset=eval_dataset,
        world_size=world_size,
        local_rank=local_rank,
        seed=args.seed,
        collator=collator,
        batch_size=args.eval_batch_size,
    )

    training_steps_per_epoch = (len(train_loader) // args.gradient_accumulation_steps) + int(len(train_loader) % args.gradient_accumulation_steps > 0)
    total_training_steps = training_steps_per_epoch * args.num_epochs
    scheduler = transformers.get_scheduler(
        name=args.scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=math.ceil(total_training_steps * args.warmup_ratio),
        num_training_steps=total_training_steps,
    )

    args.training_steps_per_epoch = training_steps_per_epoch
    args.total_training_steps = total_training_steps
    args.data_config = finetuning_data_manager.config

    if local_rank == 0:
        with open(os.path.join(args.run_dir, 'config.yaml'), 'w') as f:
            yaml.dump(vars(args), f, default_flow_style=False)

        if args.wandb_name != 'none':
            wandb.init(project=args.wandb_name, name=args.run_name, config=vars(args), dir=RUN_BASE_DIR)

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    model.train()
    dist.barrier()

    global_step = 0
    for epoch in range(args.num_epochs):

        # Early stopping condition ############################################
        if args.stopping_condition == 'sc=loss':
            dist.all_reduce(stop_flag, op=dist.ReduceOp.SUM)
            if stop_flag.item() > 0:
                break
        #######################################################################

        for step, batch in enumerate(train_loader):
            inputs = {k: v.to(model.device) for k, v in batch.items() if k in ('input_ids', 'attention_mask', 'labels')}
            outputs = model(**inputs)

            loss = outputs.loss
            loss = loss / args.gradient_accumulation_steps
            loss.backward()
            if (step + 1) % args.gradient_accumulation_steps == 0 or step == len(train_loader) - 1:
                grad_norm = model.clip_grad_norm_(args.gradient_clipping).item()
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                loss = get_all_reduce_mean(loss.detach())
                if local_rank == 0:
                    log_stats(
                        global_step=global_step,
                        total_training_steps=total_training_steps,
                        epoch=epoch,
                        loss=loss,
                        grad_norm=grad_norm,
                        scheduler=scheduler,
                        use_wandb=args.wandb_name != 'none',
                    )
                    if args.show_gpu_usage:
                        print_gpu_usage()

                if global_step and global_step % args.eval_every_n_steps == 0:
                    eval_loss = run_evaluation(model, eval_loader, local_rank)
                    print_rank0(f'\[eval]  global_step={global_step} / {total_training_steps} | epoch={epoch} | eval_loss={eval_loss:.6f}')
                    if args.wandb_name != 'none' and local_rank == 0:
                        wandb.log(dict(eval_loss=eval_loss))
                    model.train()

                if global_step and global_step % args.save_every_n_steps == 0:
                    save_model(local_rank, model, tokenizer, args.run_dir, ckpt_name=f'checkpoint-{global_step}')

                # Early stopping condition ####################################
                if local_rank == 0 and args.stopping_condition == 'sc=loss':
                    if loss.item() < stopping_loss:
                        stopping_counter += 1

                    if stopping_counter >= max_stopping_counts:
                        stop_flag.fill_(1)

                if args.stopping_condition == 'sc=loss':
                    dist.all_reduce(stop_flag, op=dist.ReduceOp.SUM)
                    if stop_flag.item() > 0:
                        break
                    print_rank0(f'stopping_counter={stopping_counter} | flag={stop_flag.item()}')
                ###############################################################

                global_step += 1
    dist.barrier()

    save_model(local_rank, model, tokenizer, args.run_dir, ckpt_name='final')
    print_rank0(f'Training finished in {(time.time() - start_time) / 60:.2f} minutes')
    dist.destroy_process_group()
