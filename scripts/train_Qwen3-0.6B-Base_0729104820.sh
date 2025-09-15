##### Auto-generated 2025-07-29 10:48:20 #####
# Model     : Qwen3-0.6B-Base
# LoRA mode : true
# Template  : qwen3

##### Environment #####
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export HF_HUB_OFFLINE=0
export HF_DATASETS_OFFLINE=0
export HF_DATASETS_TRUST_REMOTE_CODE=1
export TRUST_REMOTE_CODE=True
export HF_ALLOW_CODE_EVAL=1

##### Training #####
###### B  max=2000  lr=2e-4 ######
mkdir -p "/home/ubuntu/Shadow/results/0729/result-Qwen3-0.6B-Base-0729/B-2k-lora-rank8-lr0.0002-Shadow_2k"
cd "/home/ubuntu/Shadow"
llamafactory-cli train \
  --model_name_or_path "Qwen/Qwen3-0.6B-Base" \
  --stage sft \
  --do_train true \
  --finetuning_type lora --lora_rank 8 \
  --deepspeed examples/deepspeed/ds_z3_config.json \
  --dataset "Shadow_2k" \
  --template "qwen3" \
  --cutoff_len 4096 \
  --max_samples 2000 \
  --output_dir "/home/ubuntu/Shadow/results/0729/result-Qwen3-0.6B-Base-0729/B-2k-lora-rank8-lr0.0002-Shadow_2k" \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 16 \
  --learning_rate 2e-4 \
  --num_train_epochs 1 \
  --logging_steps 1 \
  --save_steps 1000 \
  --plot_loss true \
  --lr_scheduler_type cosine \
  --warmup_ratio 0.1 \
  --bf16 true \
  --val_size 0.01 \
  --per_device_eval_batch_size 1 \
  --eval_strategy steps \
  --eval_steps 10000 \
  --trust_remote_code True \
  --flash_attn fa2 \
  --overwrite_cache false

###### I  max=2000  lr=2e-4 ######
mkdir -p "/home/ubuntu/Shadow/results/0729/result-Qwen3-0.6B-Base-0729/I-2k-lora-rank8-lr0.0002-Shadow_2k"
cd "/home/ubuntu/Shadow"
llamafactory-cli train \
  --model_name_or_path "Qwen/Qwen3-0.6B" \
  --stage sft \
  --do_train true \
  --finetuning_type lora --lora_rank 8 \
  --deepspeed examples/deepspeed/ds_z3_config.json \
  --dataset "Shadow_2k" \
  --template "qwen3" \
  --cutoff_len 4096 \
  --max_samples 2000 \
  --output_dir "/home/ubuntu/Shadow/results/0729/result-Qwen3-0.6B-Base-0729/I-2k-lora-rank8-lr0.0002-Shadow_2k" \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 16 \
  --learning_rate 2e-4 \
  --num_train_epochs 1 \
  --logging_steps 1 \
  --save_steps 1000 \
  --plot_loss true \
  --lr_scheduler_type cosine \
  --warmup_ratio 0.1 \
  --bf16 true \
  --val_size 0.01 \
  --per_device_eval_batch_size 1 \
  --eval_strategy steps \
  --eval_steps 10000 \
  --trust_remote_code True \
  --flash_attn fa2 \
  --overwrite_cache false

##### LoRA delta-merge #####
python3 /home/ubuntu/Shadow/src/shadow/merge_lora.py \
  --adapter_path "/home/ubuntu/Shadow/results/0729/result-Qwen3-0.6B-Base-0729/B-2.0k-lora-rank8-ratio0.5-lr-Shadow_2k" \
  --target_base "Qwen/Qwen3-0.6B" \
  --merge_tag "B2I" \
  --template "qwen3"

python3 /home/ubuntu/Shadow/src/shadow/merge_lora.py \
  --adapter_path "/home/ubuntu/Shadow/results/0729/result-Qwen3-0.6B-Base-0729/I-2.0k-lora-rank8-ratio0.5-lr-Shadow_2k" \
  --target_base "Qwen/Qwen3-0.6B" \
  --merge_tag "I2I" \
  --template "qwen3"

##### Evaluation list #####
# ('Qwen3-0.6B-Base_merged_B2I_lora8_lr0.0002_Shadow_2k','/home/ubuntu/Shadow/results/0729/result-Qwen3-0.6B-Base-0729/merged-B2I'),
# ('Qwen3-0.6B-Base_merged_I2I_lora8_lr0.0002_Shadow_2k','/home/ubuntu/Shadow/results/0729/result-Qwen3-0.6B-Base-0729/merged-I2I'),

