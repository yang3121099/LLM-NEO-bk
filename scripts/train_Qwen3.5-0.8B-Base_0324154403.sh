#!/usr/bin/env bash
set -euo pipefail

##### Auto-generated 2026-03-24 15:44:03 #####
# Model     : Qwen3.5-0.8B-Base
# LoRA mode : true
# Template  : qwen3

##### Environment #####
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export HF_HUB_OFFLINE=0
export HF_DATASETS_OFFLINE=0
export HF_DATASETS_TRUST_REMOTE_CODE=1
export TRUST_REMOTE_CODE=True
export HF_ALLOW_CODE_EVAL=1

###############################################################################
##### Step 1: Training (Base + Instruct with LoRA)                        #####
###############################################################################

###### B  max=2000  lr=0.0002 ######
mkdir -p "/home/user/LLM-NEO-bk/results/0324/result-Qwen3.5-0.8B-Base-0324/B-2k-lora-rank128-lr0.0002-Shadow_2k"
cd "/home/user/LLM-NEO-bk"
llamafactory-cli train \
  --model_name_or_path "Qwen/Qwen3.5-0.8B-Base" \
  --stage sft \
  --do_train true \
  --finetuning_type lora --lora_rank 128 \
  --dataset "Shadow_2k" \
  --template "qwen3" \
  --cutoff_len 4096 \
  --max_samples 2000 \
  --output_dir "/home/user/LLM-NEO-bk/results/0324/result-Qwen3.5-0.8B-Base-0324/B-2k-lora-rank128-lr0.0002-Shadow_2k" \
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
  --overwrite_cache false \
  --use_fast_tokenizer True

###### I  max=2000  lr=0.0002 ######
mkdir -p "/home/user/LLM-NEO-bk/results/0324/result-Qwen3.5-0.8B-Base-0324/I-2k-lora-rank128-lr0.0002-Shadow_2k"
cd "/home/user/LLM-NEO-bk"
llamafactory-cli train \
  --model_name_or_path "Qwen/Qwen3.5-0.8B" \
  --stage sft \
  --do_train true \
  --finetuning_type lora --lora_rank 128 \
  --dataset "Shadow_2k" \
  --template "qwen3" \
  --cutoff_len 4096 \
  --max_samples 2000 \
  --output_dir "/home/user/LLM-NEO-bk/results/0324/result-Qwen3.5-0.8B-Base-0324/I-2k-lora-rank128-lr0.0002-Shadow_2k" \
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
  --overwrite_cache false \
  --use_fast_tokenizer True

###############################################################################
##### Step 2: LoRA Delta Merge                                            #####
###############################################################################

### Merge: B2I (adapter=B, target=I) ###
mkdir -p "/home/user/LLM-NEO-bk/results/0324/result-Qwen3.5-0.8B-Base-0324/B-2k-lora-rank128-lr0.0002-Shadow_2k/merged-B2I"
python3 /home/user/LLM-NEO-bk/src/shadow/merge_lora.py \
  --adapter_path "/home/user/LLM-NEO-bk/results/0324/result-Qwen3.5-0.8B-Base-0324/B-2k-lora-rank128-lr0.0002-Shadow_2k" \
  --target_base "Qwen/Qwen3.5-0.8B" \
  --merge_tag "B2I" \
  --template "qwen3"

### Merge: I2I (adapter=I, target=I) ###
mkdir -p "/home/user/LLM-NEO-bk/results/0324/result-Qwen3.5-0.8B-Base-0324/I-2k-lora-rank128-lr0.0002-Shadow_2k/merged-I2I"
python3 /home/user/LLM-NEO-bk/src/shadow/merge_lora.py \
  --adapter_path "/home/user/LLM-NEO-bk/results/0324/result-Qwen3.5-0.8B-Base-0324/I-2k-lora-rank128-lr0.0002-Shadow_2k" \
  --target_base "Qwen/Qwen3.5-0.8B" \
  --merge_tag "I2I" \
  --template "qwen3"

### Merge: I2B (adapter=I, target=B) ###
# mkdir -p "/home/user/LLM-NEO-bk/results/0324/result-Qwen3.5-0.8B-Base-0324/I-2k-lora-rank128-lr0.0002-Shadow_2k/merged-I2B"
# python3 /home/user/LLM-NEO-bk/src/shadow/merge_lora.py \
#   --adapter_path "/home/user/LLM-NEO-bk/results/0324/result-Qwen3.5-0.8B-Base-0324/I-2k-lora-rank128-lr0.0002-Shadow_2k" \
#   --target_base "Qwen/Qwen3.5-0.8B-Base" \
#   --merge_tag "I2B" \
#   --template "qwen3"

### Merge: B2B (adapter=B, target=B) ###
# mkdir -p "/home/user/LLM-NEO-bk/results/0324/result-Qwen3.5-0.8B-Base-0324/B-2k-lora-rank128-lr0.0002-Shadow_2k/merged-B2B"
# python3 /home/user/LLM-NEO-bk/src/shadow/merge_lora.py \
#   --adapter_path "/home/user/LLM-NEO-bk/results/0324/result-Qwen3.5-0.8B-Base-0324/B-2k-lora-rank128-lr0.0002-Shadow_2k" \
#   --target_base "Qwen/Qwen3.5-0.8B-Base" \
#   --merge_tag "B2B" \
#   --template "qwen3"

###############################################################################
##### Evaluation list (model paths for OpenCompass)                       #####
###############################################################################

# ('result-Qwen3.5-0.8B-Base-0324/B-2k-lora-rank128-lr0.0002-Shadow_2k/merged-B2I','/home/user/LLM-NEO-bk/results/0324/result-Qwen3.5-0.8B-Base-0324/B-2k-lora-rank128-lr0.0002-Shadow_2k/merged-B2I'),
# ('result-Qwen3.5-0.8B-Base-0324/I-2k-lora-rank128-lr0.0002-Shadow_2k/merged-I2I','/home/user/LLM-NEO-bk/results/0324/result-Qwen3.5-0.8B-Base-0324/I-2k-lora-rank128-lr0.0002-Shadow_2k/merged-I2I'),
## ('result-Qwen3.5-0.8B-Base-0324/I-2k-lora-rank128-lr0.0002-Shadow_2k/merged-I2B','/home/user/LLM-NEO-bk/results/0324/result-Qwen3.5-0.8B-Base-0324/I-2k-lora-rank128-lr0.0002-Shadow_2k/merged-I2B'),
## ('result-Qwen3.5-0.8B-Base-0324/B-2k-lora-rank128-lr0.0002-Shadow_2k/merged-B2B','/home/user/LLM-NEO-bk/results/0324/result-Qwen3.5-0.8B-Base-0324/B-2k-lora-rank128-lr0.0002-Shadow_2k/merged-B2B'),

