##### Auto-generated 2025-05-23 12:05:27 #####
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
###### B  max=2000  lr=1e-5 ######
mkdir -p "./results/0523/result-Qwen3-0.6B-Base-0523/B-2k-lora-rank128-lr0.00001-Shadow_2k"
cd "."
llamafactory-cli train \
  --model_name_or_path "Qwen/Qwen3-0.6B-Base" \
  --stage sft \
  --do_train true \
  --finetuning_type lora --lora_rank 128 \
  --deepspeed examples/deepspeed/ds_z3_config.json \
  --dataset "Shadow_2k" \
  --template "qwen3" \
  --cutoff_len 4096 \
  --max_samples 2000 \
  --output_dir "./results/0523/result-Qwen3-0.6B-Base-0523/B-2k-lora-rank128-lr0.00001-Shadow_2k" \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 16 \
  --learning_rate 1e-5 \
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

###### I  max=2000  lr=1e-5 ######
mkdir -p "./results/0523/result-Qwen3-0.6B-Base-0523/I-2k-lora-rank128-lr0.00001-Shadow_2k"
cd "."
llamafactory-cli train \
  --model_name_or_path "Qwen/Qwen3-0.6B" \
  --stage sft \
  --do_train true \
  --finetuning_type lora --lora_rank 128 \
  --deepspeed examples/deepspeed/ds_z3_config.json \
  --dataset "Shadow_2k" \
  --template "qwen3" \
  --cutoff_len 4096 \
  --max_samples 2000 \
  --output_dir "./results/0523/result-Qwen3-0.6B-Base-0523/I-2k-lora-rank128-lr0.00001-Shadow_2k" \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 16 \
  --learning_rate 1e-5 \
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
python3 ./src/shadow/merge_lora.py \
  --adapter_path "./results/0523/result-Qwen3-0.6B-Base-0523/B-2.0k-lora-rank128-ratio0.5-lr-Shadow_2k" \
  --target_base "Qwen/Qwen3-0.6B" \
  --merge_tag "B2I" \
  --template "qwen3"

python3 ./src/shadow/merge_lora.py \
  --adapter_path "./results/0523/result-Qwen3-0.6B-Base-0523/I-2.0k-lora-rank128-ratio0.5-lr-Shadow_2k" \
  --target_base "Qwen/Qwen3-0.6B" \
  --merge_tag "I2I" \
  --template "qwen3"

##### Evaluation list #####
# ('Qwen3-0.6B-Base_merged_B2I_lora128_lr0.00001_Shadow_2k','./results/0523/result-Qwen3-0.6B-Base-0523/merged-B2I'),
# ('Qwen3-0.6B-Base_merged_I2I_lora128_lr0.00001_Shadow_2k','./results/0523/result-Qwen3-0.6B-Base-0523/merged-I2I'),

please copy this eval_config to opencompass/examples/eval_shadow_202505.py and then run

cd ./opencompass
python3 ./run.py ./examples/eval_shadow_202505.py
