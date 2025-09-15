##### Auto-generated 2025-08-09 09:47:02 #####
# Model     : Qwen3-4B-Thinking-2507
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
###### B  max=1000  lr=0.0002 ######
mkdir -p "/home/ubuntu/Shadow/results/0809/result-Qwen3-4B-Thinking-2507-0809/B-1k-lora-rank128-lr0.0002-s1k_gptoss20b_high"
cd "/home/ubuntu/Shadow"
llamafactory-cli train \
  --model_name_or_path "Qwen/Qwen3-4B-Thinking-2507" \
  --stage sft \
  --do_train true \
  --finetuning_type lora --lora_rank 128 \
  --deepspeed examples/deepspeed/ds_z3_config.json \
  --dataset "s1k_gptoss20b_high" \
  --template "qwen3" \
  --cutoff_len 32768 \
  --max_samples 1000 \
  --output_dir "/home/ubuntu/Shadow/results/0809/result-Qwen3-4B-Thinking-2507-0809/B-1k-lora-rank128-lr0.0002-s1k_gptoss20b_high" \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 16 \
  --learning_rate 0.0002 \
  --num_train_epochs 1 \
  --logging_steps 1 \
  --save_steps 1000 \
  --save_only_model True \
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

mkdir -p "/home/ubuntu/Shadow/results/0809/result-Qwen3-4B-Thinking-2507-0809/B-1k-lora-rank128-lr0.0002-s1k_gptoss20b_high/merged-B2B"
python3 /home/ubuntu/Shadow/src/shadow/merge_lora.py \
 --adapter_path "/home/ubuntu/Shadow/results/0809/result-Qwen3-4B-Thinking-2507-0809/B-1k-lora-rank128-lr0.0002-s1k_gptoss20b_high" \
 --target_base "Qwen/Qwen3-4B-Thinking-2507" \
 --merge_tag "B2B" \
 --template "qwen3"

