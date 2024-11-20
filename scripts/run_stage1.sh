#!/bin/bash -l

num_gpus=2
export OMP_NUM_THREADS=${num_gpus}
master_addr=localhost
master_port=$(comm -23 <(seq 49152 65535 | sort) <(ss -Htan | awk '{print $4}' | cut -d':' -f2 | sort -u) | shuf | head -n 1)

conda activate py310n

# Main paths
root_dir="put/your/path/here"
run_base_dir="put/your/path/here"
run_name="put/your/run/name/here"

# Common configurations
base_model_name="llama-3-8b"
seed=0
wandb_name="none"

# Task-specific configurations ################################################
dataset_name="kvr"
#dataset_name="popqa"
#dataset_name="triviaqa"

mix_ratio=0.0
mixd="none"
#mixd="knowledge-pile"
#mixd="arxiv-pile"
#mixd="fineweb"
#mixd="random-pretraining"

gradient_accumulation_steps=1
###############################################################################

cd ${root_dir}

if [ "$dataset_name" = "kvr" ]; then
    stopping_condition="sc=fixed"
else
    stopping_condition="sc=loss"
fi

if [ "$mixd" = "none" ]; then
    task_mode="none"
else
    task_mode="mix"
fi

# Common configurations #######################################################
if [[ ${num_gpus} == "4" ]]; then
    train_batch_size=8
elif [[ ${num_gpus} == "2" ]]; then
    train_batch_size=16
else
    echo "Invalid num_gpus. Please set to 2 or 4."
    exit 1
fi

model_name="meta-llama/Meta-Llama-3-8B"
num_epochs=20
train_num_examples=2000
max_seq_length=2048
dev_num_examples=20
eval_every_n_steps=100
eval_batch_size=4
lr=5e-5
dataset_module_path="src.datasets"

if [ "$dataset_name" = "gsm8k" ]; then
    train_num_examples=2000
    num_epochs=10
    gradient_accumulation_steps=1
    stopping_condition="sc=fixed"
    lr=5e-5
fi
###############################################################################

bsz=$(( "${train_batch_size}" * "${gradient_accumulation_steps}" * "${num_gpus}" ))

if [[ ${task_mode} == *"mix"* ]]; then
  train_num_examples=$(echo "${train_num_examples} + ${train_num_examples} * ${mix_ratio}" | bc)
  train_num_examples=$(printf "%.0f" ${train_num_examples})
  task_mode="${task_mode}+${mixd}"
fi

run_dir="${run_base_dir}/${run_name}"
mkdir -p ${run_dir}

command="torchrun --rdzv-backend=c10d --rdzv-endpoint=${master_addr}:${master_port} --nnodes=1 --nproc-per-node=${num_gpus} -m src.train"
args=(
    --run_name ${run_name} \
    --run_base_dir ${run_base_dir} \
    --run_dir ${run_dir} \
    --model_name ${model_name} \
    --base_model_name ${base_model_name} \
    --dataset_name ${dataset_name} \
    --dataset_module_path ${dataset_module_path} \
    --train_num_examples ${train_num_examples} \
    --dev_num_examples ${dev_num_examples} \
    --num_epochs ${num_epochs} \
    --eval_every_n_steps ${eval_every_n_steps} \
    --max_seq_length ${max_seq_length} \
    --lr ${lr} \
    --gradient_checkpointing \
    --train_batch_size ${train_batch_size} \
    --eval_batch_size ${eval_batch_size} \
    --gradient_accumulation_steps ${gradient_accumulation_steps} \
    --seed ${seed} \
    --use_train_for_dev \
    --task_mode ${task_mode} \
    --mix_ratio ${mix_ratio} \
    --stopping_condition ${stopping_condition} \
    --wandb_name ${wandb_name} \
    --show_gpu_usage
)

echo "${command} ${args[@]}" | tee "${run_dir}/command.txt"
${command} "${args[@]}" 2>&1 | tee -a "${run_dir}/log.txt"


# Run eval after the model is trained
eval_command="python -m src.run_eval"
eval_model_name="${run_dir}/final"
eval_args=(
    --seed 0
    --batch_size 20
    --base_model_name "${base_model_name}"
    --dataset_name "${dataset_name}"
    --task_mode none
    --model_name "${eval_model_name}"
    --save_results
)
${eval_command} "${eval_args[@]}"
