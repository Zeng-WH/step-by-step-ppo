export HF_HOME="/ssddata/weihao00/model_zoo"
# export CUDA_HOME="/ssddata/local/cuda-latest"
# export PATH="${CUDA_HOME}/bin:${PATH}"
# export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}"
rm -rf ~/.cache/torch/kernels/*
# CUDA_VISIBLE_DEVICES=5,6,7,3  accelerate launch --config_file=configs/deepspeed_zero2.yaml --num_machines 1  --num_processes 4 ppo_train_prm_full.py \
#     --model_name meta-math/MetaMath-Mistral-7B \
#     --conv_template "vicuna_v1.1" \
#     --reward_model_name peiyi9979/math-shepherd-mistral-7b-prm \
#     --adafactor=False \
#     --tokenizer_name meta-math/MetaMath-Mistral-7B \
#     --save_freq=50 \
#     --output_max_length=2048 \
#     --batch_size=64 \
#     --mini_batch_size 1 \
#     --gradient_accumulation_steps=8 \
#     --batched_gen=True \
#     --ppo_epochs=4 \
#     --seed=42 \
#     --reward_baseline 0.4 \
#     --learning_rate=5e-7 \
#     --early_stopping=True \
#     --output_dir /ssddata/weihao00/save_dir/full_ppo/ppo_tbsz256_prm_bbsz32_pep4_lr5e7_24share_pen_0.4_l2048 \
#     --json_path /ssddata/weihao00/trl-lora-rlhf-master/data/dpo_ultra_wizard_10times_180k.json \
#     --data_split train \
#     --logging_steps 1 \
#     --init_kl_coef 0.04 \
#     --log_with tensorboard \
#     --num_epochs 2 \
#     --logging_dir /ssddata/weihao00/save_dir/full_ppo/ppo_tbsz256_prm_bbsz32_pep4_lr5e7_24share_pen_0.4_l2048 \
#     --lr_scheduler_type "linear"

CUDA_VISIBLE_DEVICES=2,4 accelerate launch --config_file=configs/deepspeed_zero2.yaml --num_machines 1  --num_processes 2 --main_process_port 29501 ppo_train_prm_full_wandb.py \
    --model_name meta-math/MetaMath-Mistral-7B \
    --conv_template "vicuna_v1.1" \
    --reward_model_name peiyi9979/math-shepherd-mistral-7b-prm \
    --adafactor=False \
    --tokenizer_name meta-math/MetaMath-Mistral-7B \
    --save_freq=25 \
    --output_max_length=2048 \
    --batch_size=64 \
    --mini_batch_size 1 \
    --gradient_accumulation_steps=32 \
    --batched_gen=True \
    --ppo_epochs=4 \
    --seed=42 \
    --reward_baseline 0.0 \
    --learning_rate=2e-7 \
    --early_stopping=True \
    --output_dir /ssddata/weihao00/save_dir/full_ppo/ppo_tbsz256_prm_bbsz128_pep4_lr2e7_24share_pen_0.4_l2048_reweight_new2_wandb \
    --json_path data/math_metamathqa_395K.json \
    --data_split train \
    --logging_steps 1 \
    --init_kl_coef 0.04 \
    --log_with wandb \
    --num_epochs 2 \
    --share_layers 10  \
    --lr_scheduler_type "linear"  