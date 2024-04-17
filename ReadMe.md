# Step-by-Step PPO Codebase


## Environment


```
conda create -n ppo_train python==3.10.0
conda activate ppo_train
conda install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=12.1 -c pytorch -c nvidia
pip install transformers==4.36.2
pip install -r requirements.txt
pip install trl
pip install flash-attn==2.3.6 --no-build-isolation
```


## Run Code

其中影响比较大的超参数包括：batch_size，mini_batch_size, reward_baseline以及init_kl_coef

output_dir表示ckpt的存储路径


```
accelerate launch --config_file=configs/deepspeed_zero2.yaml --num_machines 1  --num_processes 8 ppo_train_prm_full_wandb.py \
    --model_name meta-math/MetaMath-Mistral-7B \
    --conv_template "vicuna_v1.1" \
    --reward_model_name peiyi9979/math-shepherd-mistral-7b-prm \
    --adafactor=False \
    --tokenizer_name meta-math/MetaMath-Mistral-7B \
    --save_freq=2 \
    --output_max_length=2048 \
    --batch_size=32 \
    --mini_batch_size 1 \
    --gradient_accumulation_steps=16 \
    --batched_gen=True \
    --ppo_epochs=4 \
    --seed=42 \
    --reward_baseline 0.0 \
    --learning_rate=2e-7 \
    --early_stopping=True \
    --output_dir output_dir/rollout256_backbsz128_sample_data/ \
    --json_path data/sample_data.json \
    --data_split train \
    --logging_steps 1 \
    --init_kl_coef 0.04 \
    --log_with wandb \
    --num_epochs 2 \
    --share_layers 0  \
    --lr_scheduler_type "linear" 

```


## Infer Code

以下代码直接vllm遍历所有output_model的ckpt进行推理

```
python gsm8k_test_wandb.py --model_dir output_dir/rollout256_backbsz128_sample_data/ --model_id test --start 0 --end 1400 --batch_size 80 --tensor_parallel_size 8
```

