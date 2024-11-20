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
model_name="put/your/model/here"

# Common configurations
base_model_name="llama-3-8b"
seed=0
wandb_name="none"

# Task-specific configurations ################################################

# data A config --------------------
dataset_name_A="kvr"
#dataset_name_A="popqa"
#dataset_name_A="triviaqa"

# data B config --------------------
dataset_name_B="lama"
#dataset_name_B="webqa"
#dataset_name_B="entityqa"
#dataset_name_B="gsm8k"
#dataset_name_B="math"
#dataset_name_B="evolcode"
#dataset_name_B="apps"
#dataset_name_B="ultrachat"

# mixing config --------------------
mix_ratio=0.0
mixd="none"
#mix_ratio=2.0
#mixd="knowledge-pile"
#mixd="random-pretraining"

mix_ratio_B=0.0
mixd_B="none"
#mix_ratio_B=2.0
#mixd_B="knowledge-pile"
#mixd_B="random-pretraining"

gradient_accumulation_steps=1  # for controlling total bsz 1: 32, 4: 128
###############################################################################


cd ${root_dir}

task_type_A="con"
task_type_B="con"

if [ "$dataset_name_A" = "kvr" ] || [ "$dataset_name_A" = "kvrB" ]; then
    stopping_condition="sc=fixed"
else
    stopping_condition="sc=loss"
fi

if [ "$mixd_B" = "none" ]; then
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

num_epochs=20
train_num_examples=2000
max_seq_length=2048
dev_num_examples=20
eval_every_n_steps=100
eval_batch_size=4
lr=5e-5
dataset_module_path="src.datasets"
###############################################################################

bsz=$(( "${train_batch_size}" * "${gradient_accumulation_steps}" * "${num_gpus}" ))
bsz_A="${bsz}"

if [[ ${task_mode} == *"mix"* ]]; then
  train_num_examples=$(echo "${train_num_examples} + ${train_num_examples} * ${mix_ratio_B}" | bc)
  train_num_examples=$(printf "%.0f" ${train_num_examples})
  task_mode="${task_mode}+${mixd_B}"
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
    --dataset_name ${dataset_name_B} \
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
    --mix_ratio ${mix_ratio_B} \
    --stopping_condition ${stopping_condition} \
    --wandb_name ${wandb_name} \
    --show_gpu_usage
)

echo "${command} ${args[@]}" | tee "${run_dir}/command.txt"
echo -e "$(date +"%Y-%m-%d %H:%M:%S")\n${run_dir}\n#####\n" >> "global_slurm_log.txt"
${command} "${args[@]}" 2>&1 | tee -a "${run_dir}/log.txt"

# Evaluate on A and B
eval_command="python -m src.run_eval"
eval_model_name="${run_dir}/final"

mc=("medqa commonsenseqa piqa")
inst=("openmathinstruct1" "ultrachat" "evolcode")

# Run eval after the model is trained on dataset A
if [ "$task_type_A" = "qa" ]; then
    eval_task_mode_A="qa_multiple_choice+eval"
elif [ "$task_type_A" = "con" ]; then
    eval_task_mode_A="none"
else
    echo "Invalid task_type_A. Please set to 'qa' or 'con'."
    exit 1
fi

eval_args_A=(
    --seed "${seed}"
    --batch_size 20
    --base_model_name "${base_model_name}"
    --dataset_name "${dataset_name_A}"
    --task_mode "${eval_task_mode_A}"
    --model_name "${eval_model_name}"
    --save_results
)
${eval_command} "${eval_args_A[@]}"


# Run eval after the model is trained on dataset B
if [[ ! " ${inst[*]} " =~ " $dataset_name_B " ]]; then
    if [ "$task_type_B" = "qa" ]; then
        eval_task_mode_B="qa_multiple_choice+eval"
    elif [ "$task_type_B" = "con" ]; then
        eval_task_mode_B="none"
    else
        echo "Invalid task_type_B. Please set to 'qa' or 'con'."
        exit 1
    fi
    
    eval_args_B=(
        --seed "${seed}"
        --batch_size 20
        --base_model_name "${base_model_name}"
        --dataset_name "${dataset_name_B}"
        --task_mode "${eval_task_mode_B}"
        --model_name "${eval_model_name}"
        --save_results
    )
    ${eval_command} "${eval_args_B[@]}"
fi

# Delete the model and keep only the eval file
#find "${eval_model_name}" -type f ! -name "eval_metrics.json" -delete
find "${eval_model_name}" -type f ! \( -name "eval_metrics.json" -o -name "*_predictions.jsonl" \) -delete


echo -e "$(date +"%Y-%m-%d %H:%M:%S")\nEvaluating:\n${eval_model_name}\n" >> "eval_global_slurm_log.txt"
cat "${eval_model_name}/eval_metrics.json" >> "eval_global_slurm_log.txt"
echo -e "\n#####\n" >> "eval_global_slurm_log.txt"
