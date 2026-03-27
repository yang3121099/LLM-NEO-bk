#!/usr/bin/env bash
set -euo pipefail

##### Auto-generated 2026-03-27 05:54:29 #####
# Experiment: Weight Similarity × Delta Transferability
# Dataset: Shadow_2k (2000 samples)
# LoRA rank: 128, LR: 2e-4

##### Paths (resolved at runtime) #####
WORKSPACE_DIR="$(cd "$(dirname "$0")/.." && pwd)"
RESULTS_DIR="$WORKSPACE_DIR/results"

##### Environment #####
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export HF_HUB_OFFLINE=0
export HF_DATASETS_OFFLINE=0
export HF_DATASETS_TRUST_REMOTE_CODE=1
export TRUST_REMOTE_CODE=True
export HF_ALLOW_CODE_EVAL=1

###############################################################################
##### Step 1: Train LoRA on each model using Shadow_2k                   #####
###############################################################################

###### Train LoRA on: Base3.1 (meta-llama/Llama-3.1-8B) ######
mkdir -p "$RESULTS_DIR/0327/result-weight-similarity-0327/Base3.1-2k-lora-rank128-lr0.0002-shadow2k"
cd "$WORKSPACE_DIR"
llamafactory-cli train \
  --model_name_or_path "meta-llama/Llama-3.1-8B" \
  --stage sft \
  --do_train true \
  --finetuning_type lora --lora_rank 128 \
  --dataset "Shadow_2k" \
  --template "llama3" \
  --cutoff_len 16384 \
  --max_samples 2000 \
  --output_dir "$RESULTS_DIR/0327/result-weight-similarity-0327/Base3.1-2k-lora-rank128-lr0.0002-shadow2k" \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 256 \
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

###### Train LoRA on: Instruct3.1 (meta-llama/Llama-3.1-8B-Instruct) ######
mkdir -p "$RESULTS_DIR/0327/result-weight-similarity-0327/Instruct3.1-2k-lora-rank128-lr0.0002-shadow2k"
cd "$WORKSPACE_DIR"
llamafactory-cli train \
  --model_name_or_path "meta-llama/Llama-3.1-8B-Instruct" \
  --stage sft \
  --do_train true \
  --finetuning_type lora --lora_rank 128 \
  --dataset "Shadow_2k" \
  --template "llama3" \
  --cutoff_len 16384 \
  --max_samples 2000 \
  --output_dir "$RESULTS_DIR/0327/result-weight-similarity-0327/Instruct3.1-2k-lora-rank128-lr0.0002-shadow2k" \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 256 \
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

###### Train LoRA on: Tulu3-SFT (allenai/Llama-3.1-Tulu-3-8B-SFT) ######
mkdir -p "$RESULTS_DIR/0327/result-weight-similarity-0327/Tulu3-SFT-2k-lora-rank128-lr0.0002-shadow2k"
cd "$WORKSPACE_DIR"
llamafactory-cli train \
  --model_name_or_path "allenai/Llama-3.1-Tulu-3-8B-SFT" \
  --stage sft \
  --do_train true \
  --finetuning_type lora --lora_rank 128 \
  --dataset "Shadow_2k" \
  --template "llama3" \
  --cutoff_len 16384 \
  --max_samples 2000 \
  --output_dir "$RESULTS_DIR/0327/result-weight-similarity-0327/Tulu3-SFT-2k-lora-rank128-lr0.0002-shadow2k" \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 256 \
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

###### Train LoRA on: Tulu3-DPO (allenai/Llama-3.1-Tulu-3-8B-DPO) ######
mkdir -p "$RESULTS_DIR/0327/result-weight-similarity-0327/Tulu3-DPO-2k-lora-rank128-lr0.0002-shadow2k"
cd "$WORKSPACE_DIR"
llamafactory-cli train \
  --model_name_or_path "allenai/Llama-3.1-Tulu-3-8B-DPO" \
  --stage sft \
  --do_train true \
  --finetuning_type lora --lora_rank 128 \
  --dataset "Shadow_2k" \
  --template "llama3" \
  --cutoff_len 16384 \
  --max_samples 2000 \
  --output_dir "$RESULTS_DIR/0327/result-weight-similarity-0327/Tulu3-DPO-2k-lora-rank128-lr0.0002-shadow2k" \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 256 \
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

###### Train LoRA on: Tulu3-RLVR (allenai/Llama-3.1-Tulu-3-8B) ######
mkdir -p "$RESULTS_DIR/0327/result-weight-similarity-0327/Tulu3-RLVR-2k-lora-rank128-lr0.0002-shadow2k"
cd "$WORKSPACE_DIR"
llamafactory-cli train \
  --model_name_or_path "allenai/Llama-3.1-Tulu-3-8B" \
  --stage sft \
  --do_train true \
  --finetuning_type lora --lora_rank 128 \
  --dataset "Shadow_2k" \
  --template "llama3" \
  --cutoff_len 16384 \
  --max_samples 2000 \
  --output_dir "$RESULTS_DIR/0327/result-weight-similarity-0327/Tulu3-RLVR-2k-lora-rank128-lr0.0002-shadow2k" \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 256 \
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

###### Train LoRA on: Tulu3.1 (allenai/Llama-3.1-Tulu-3.1-8B) ######
mkdir -p "$RESULTS_DIR/0327/result-weight-similarity-0327/Tulu3.1-2k-lora-rank128-lr0.0002-shadow2k"
cd "$WORKSPACE_DIR"
llamafactory-cli train \
  --model_name_or_path "allenai/Llama-3.1-Tulu-3.1-8B" \
  --stage sft \
  --do_train true \
  --finetuning_type lora --lora_rank 128 \
  --dataset "Shadow_2k" \
  --template "llama3" \
  --cutoff_len 16384 \
  --max_samples 2000 \
  --output_dir "$RESULTS_DIR/0327/result-weight-similarity-0327/Tulu3.1-2k-lora-rank128-lr0.0002-shadow2k" \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 256 \
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

###### Train LoRA on: Base3 (meta-llama/Llama-3-8B) ######
mkdir -p "$RESULTS_DIR/0327/result-weight-similarity-0327/Base3-2k-lora-rank128-lr0.0002-shadow2k"
cd "$WORKSPACE_DIR"
llamafactory-cli train \
  --model_name_or_path "meta-llama/Llama-3-8B" \
  --stage sft \
  --do_train true \
  --finetuning_type lora --lora_rank 128 \
  --dataset "Shadow_2k" \
  --template "llama3" \
  --cutoff_len 16384 \
  --max_samples 2000 \
  --output_dir "$RESULTS_DIR/0327/result-weight-similarity-0327/Base3-2k-lora-rank128-lr0.0002-shadow2k" \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 256 \
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

###### Train LoRA on: Instruct3 (meta-llama/Llama-3-8B-Instruct) ######
mkdir -p "$RESULTS_DIR/0327/result-weight-similarity-0327/Instruct3-2k-lora-rank128-lr0.0002-shadow2k"
cd "$WORKSPACE_DIR"
llamafactory-cli train \
  --model_name_or_path "meta-llama/Llama-3-8B-Instruct" \
  --stage sft \
  --do_train true \
  --finetuning_type lora --lora_rank 128 \
  --dataset "Shadow_2k" \
  --template "llama3" \
  --cutoff_len 16384 \
  --max_samples 2000 \
  --output_dir "$RESULTS_DIR/0327/result-weight-similarity-0327/Instruct3-2k-lora-rank128-lr0.0002-shadow2k" \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 256 \
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

###### Train LoRA on: R1-Distill (deepseek-ai/DeepSeek-R1-Distill-Llama-8B) ######
mkdir -p "$RESULTS_DIR/0327/result-weight-similarity-0327/R1-Distill-2k-lora-rank128-lr0.0002-shadow2k"
cd "$WORKSPACE_DIR"
llamafactory-cli train \
  --model_name_or_path "deepseek-ai/DeepSeek-R1-Distill-Llama-8B" \
  --stage sft \
  --do_train true \
  --finetuning_type lora --lora_rank 128 \
  --dataset "Shadow_2k" \
  --template "llama3" \
  --cutoff_len 16384 \
  --max_samples 2000 \
  --output_dir "$RESULTS_DIR/0327/result-weight-similarity-0327/R1-Distill-2k-lora-rank128-lr0.0002-shadow2k" \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 256 \
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
##### Step 2: Cross-model LoRA delta merge (Shadow-FT transfer matrix)    #####
###############################################################################

# For each (source, target) pair, merge the LoRA adapter trained on SOURCE
# onto the TARGET model. This tests delta transferability across models.
# Pairs are organized by experimental interest:
#   - Pipeline transfers (Base3.1 ↔ SFT ↔ DPO ↔ RLVR ↔ Instruct3.1)
#   - Cross-family transfers (Base3.1-trained → R1-Distill, etc.)

### Merge: Base3.12Instruct3.1 (adapter=Base3.1, target=Instruct3.1) ###
mkdir -p "$RESULTS_DIR/0327/result-weight-similarity-0327/Base3.1-2k-lora-rank128-lr0.0002-shadow2k/merged-Base3.12Instruct3.1"
python3 "$WORKSPACE_DIR/src/shadow/merge_lora.py" \
  --adapter_path "$RESULTS_DIR/0327/result-weight-similarity-0327/Base3.1-2k-lora-rank128-lr0.0002-shadow2k" \
  --target_base "meta-llama/Llama-3.1-8B-Instruct" \
  --merge_tag "Base3.12Instruct3.1" \
  --template "llama3"

### Merge: Tulu3-SFT2Instruct3.1 (adapter=Tulu3-SFT, target=Instruct3.1) ###
mkdir -p "$RESULTS_DIR/0327/result-weight-similarity-0327/Tulu3-SFT-2k-lora-rank128-lr0.0002-shadow2k/merged-Tulu3-SFT2Instruct3.1"
python3 "$WORKSPACE_DIR/src/shadow/merge_lora.py" \
  --adapter_path "$RESULTS_DIR/0327/result-weight-similarity-0327/Tulu3-SFT-2k-lora-rank128-lr0.0002-shadow2k" \
  --target_base "meta-llama/Llama-3.1-8B-Instruct" \
  --merge_tag "Tulu3-SFT2Instruct3.1" \
  --template "llama3"

### Merge: Tulu3-DPO2Instruct3.1 (adapter=Tulu3-DPO, target=Instruct3.1) ###
mkdir -p "$RESULTS_DIR/0327/result-weight-similarity-0327/Tulu3-DPO-2k-lora-rank128-lr0.0002-shadow2k/merged-Tulu3-DPO2Instruct3.1"
python3 "$WORKSPACE_DIR/src/shadow/merge_lora.py" \
  --adapter_path "$RESULTS_DIR/0327/result-weight-similarity-0327/Tulu3-DPO-2k-lora-rank128-lr0.0002-shadow2k" \
  --target_base "meta-llama/Llama-3.1-8B-Instruct" \
  --merge_tag "Tulu3-DPO2Instruct3.1" \
  --template "llama3"

### Merge: Tulu3-RLVR2Instruct3.1 (adapter=Tulu3-RLVR, target=Instruct3.1) ###
mkdir -p "$RESULTS_DIR/0327/result-weight-similarity-0327/Tulu3-RLVR-2k-lora-rank128-lr0.0002-shadow2k/merged-Tulu3-RLVR2Instruct3.1"
python3 "$WORKSPACE_DIR/src/shadow/merge_lora.py" \
  --adapter_path "$RESULTS_DIR/0327/result-weight-similarity-0327/Tulu3-RLVR-2k-lora-rank128-lr0.0002-shadow2k" \
  --target_base "meta-llama/Llama-3.1-8B-Instruct" \
  --merge_tag "Tulu3-RLVR2Instruct3.1" \
  --template "llama3"

### Merge: Instruct3.12Instruct3.1 (adapter=Instruct3.1, target=Instruct3.1) ###
mkdir -p "$RESULTS_DIR/0327/result-weight-similarity-0327/Instruct3.1-2k-lora-rank128-lr0.0002-shadow2k/merged-Instruct3.12Instruct3.1"
python3 "$WORKSPACE_DIR/src/shadow/merge_lora.py" \
  --adapter_path "$RESULTS_DIR/0327/result-weight-similarity-0327/Instruct3.1-2k-lora-rank128-lr0.0002-shadow2k" \
  --target_base "meta-llama/Llama-3.1-8B-Instruct" \
  --merge_tag "Instruct3.12Instruct3.1" \
  --template "llama3"

### Merge: Tulu3-SFT2Tulu3-SFT (adapter=Tulu3-SFT, target=Tulu3-SFT) ###
mkdir -p "$RESULTS_DIR/0327/result-weight-similarity-0327/Tulu3-SFT-2k-lora-rank128-lr0.0002-shadow2k/merged-Tulu3-SFT2Tulu3-SFT"
python3 "$WORKSPACE_DIR/src/shadow/merge_lora.py" \
  --adapter_path "$RESULTS_DIR/0327/result-weight-similarity-0327/Tulu3-SFT-2k-lora-rank128-lr0.0002-shadow2k" \
  --target_base "allenai/Llama-3.1-Tulu-3-8B-SFT" \
  --merge_tag "Tulu3-SFT2Tulu3-SFT" \
  --template "llama3"

### Merge: Tulu3-DPO2Tulu3-DPO (adapter=Tulu3-DPO, target=Tulu3-DPO) ###
mkdir -p "$RESULTS_DIR/0327/result-weight-similarity-0327/Tulu3-DPO-2k-lora-rank128-lr0.0002-shadow2k/merged-Tulu3-DPO2Tulu3-DPO"
python3 "$WORKSPACE_DIR/src/shadow/merge_lora.py" \
  --adapter_path "$RESULTS_DIR/0327/result-weight-similarity-0327/Tulu3-DPO-2k-lora-rank128-lr0.0002-shadow2k" \
  --target_base "allenai/Llama-3.1-Tulu-3-8B-DPO" \
  --merge_tag "Tulu3-DPO2Tulu3-DPO" \
  --template "llama3"

### Merge: Tulu3-RLVR2Tulu3-RLVR (adapter=Tulu3-RLVR, target=Tulu3-RLVR) ###
mkdir -p "$RESULTS_DIR/0327/result-weight-similarity-0327/Tulu3-RLVR-2k-lora-rank128-lr0.0002-shadow2k/merged-Tulu3-RLVR2Tulu3-RLVR"
python3 "$WORKSPACE_DIR/src/shadow/merge_lora.py" \
  --adapter_path "$RESULTS_DIR/0327/result-weight-similarity-0327/Tulu3-RLVR-2k-lora-rank128-lr0.0002-shadow2k" \
  --target_base "allenai/Llama-3.1-Tulu-3-8B" \
  --merge_tag "Tulu3-RLVR2Tulu3-RLVR" \
  --template "llama3"

### Merge: Base3.12Tulu3-SFT (adapter=Base3.1, target=Tulu3-SFT) ###
mkdir -p "$RESULTS_DIR/0327/result-weight-similarity-0327/Base3.1-2k-lora-rank128-lr0.0002-shadow2k/merged-Base3.12Tulu3-SFT"
python3 "$WORKSPACE_DIR/src/shadow/merge_lora.py" \
  --adapter_path "$RESULTS_DIR/0327/result-weight-similarity-0327/Base3.1-2k-lora-rank128-lr0.0002-shadow2k" \
  --target_base "allenai/Llama-3.1-Tulu-3-8B-SFT" \
  --merge_tag "Base3.12Tulu3-SFT" \
  --template "llama3"

### Merge: Base3.12Tulu3-DPO (adapter=Base3.1, target=Tulu3-DPO) ###
mkdir -p "$RESULTS_DIR/0327/result-weight-similarity-0327/Base3.1-2k-lora-rank128-lr0.0002-shadow2k/merged-Base3.12Tulu3-DPO"
python3 "$WORKSPACE_DIR/src/shadow/merge_lora.py" \
  --adapter_path "$RESULTS_DIR/0327/result-weight-similarity-0327/Base3.1-2k-lora-rank128-lr0.0002-shadow2k" \
  --target_base "allenai/Llama-3.1-Tulu-3-8B-DPO" \
  --merge_tag "Base3.12Tulu3-DPO" \
  --template "llama3"

### Merge: Base3.12Tulu3-RLVR (adapter=Base3.1, target=Tulu3-RLVR) ###
mkdir -p "$RESULTS_DIR/0327/result-weight-similarity-0327/Base3.1-2k-lora-rank128-lr0.0002-shadow2k/merged-Base3.12Tulu3-RLVR"
python3 "$WORKSPACE_DIR/src/shadow/merge_lora.py" \
  --adapter_path "$RESULTS_DIR/0327/result-weight-similarity-0327/Base3.1-2k-lora-rank128-lr0.0002-shadow2k" \
  --target_base "allenai/Llama-3.1-Tulu-3-8B" \
  --merge_tag "Base3.12Tulu3-RLVR" \
  --template "llama3"

### Merge: Base3.12Tulu3.1 (adapter=Base3.1, target=Tulu3.1) ###
mkdir -p "$RESULTS_DIR/0327/result-weight-similarity-0327/Base3.1-2k-lora-rank128-lr0.0002-shadow2k/merged-Base3.12Tulu3.1"
python3 "$WORKSPACE_DIR/src/shadow/merge_lora.py" \
  --adapter_path "$RESULTS_DIR/0327/result-weight-similarity-0327/Base3.1-2k-lora-rank128-lr0.0002-shadow2k" \
  --target_base "allenai/Llama-3.1-Tulu-3.1-8B" \
  --merge_tag "Base3.12Tulu3.1" \
  --template "llama3"

### Merge: Base32Instruct3 (adapter=Base3, target=Instruct3) ###
mkdir -p "$RESULTS_DIR/0327/result-weight-similarity-0327/Base3-2k-lora-rank128-lr0.0002-shadow2k/merged-Base32Instruct3"
python3 "$WORKSPACE_DIR/src/shadow/merge_lora.py" \
  --adapter_path "$RESULTS_DIR/0327/result-weight-similarity-0327/Base3-2k-lora-rank128-lr0.0002-shadow2k" \
  --target_base "meta-llama/Llama-3-8B-Instruct" \
  --merge_tag "Base32Instruct3" \
  --template "llama3"

### Merge: Base32R1-Distill (adapter=Base3, target=R1-Distill) ###
mkdir -p "$RESULTS_DIR/0327/result-weight-similarity-0327/Base3-2k-lora-rank128-lr0.0002-shadow2k/merged-Base32R1-Distill"
python3 "$WORKSPACE_DIR/src/shadow/merge_lora.py" \
  --adapter_path "$RESULTS_DIR/0327/result-weight-similarity-0327/Base3-2k-lora-rank128-lr0.0002-shadow2k" \
  --target_base "deepseek-ai/DeepSeek-R1-Distill-Llama-8B" \
  --merge_tag "Base32R1-Distill" \
  --template "llama3"

### Merge: Base3.12R1-Distill (adapter=Base3.1, target=R1-Distill) ###
mkdir -p "$RESULTS_DIR/0327/result-weight-similarity-0327/Base3.1-2k-lora-rank128-lr0.0002-shadow2k/merged-Base3.12R1-Distill"
python3 "$WORKSPACE_DIR/src/shadow/merge_lora.py" \
  --adapter_path "$RESULTS_DIR/0327/result-weight-similarity-0327/Base3.1-2k-lora-rank128-lr0.0002-shadow2k" \
  --target_base "deepseek-ai/DeepSeek-R1-Distill-Llama-8B" \
  --merge_tag "Base3.12R1-Distill" \
  --template "llama3"

### Merge: Instruct32Instruct3 (adapter=Instruct3, target=Instruct3) ###
mkdir -p "$RESULTS_DIR/0327/result-weight-similarity-0327/Instruct3-2k-lora-rank128-lr0.0002-shadow2k/merged-Instruct32Instruct3"
python3 "$WORKSPACE_DIR/src/shadow/merge_lora.py" \
  --adapter_path "$RESULTS_DIR/0327/result-weight-similarity-0327/Instruct3-2k-lora-rank128-lr0.0002-shadow2k" \
  --target_base "meta-llama/Llama-3-8B-Instruct" \
  --merge_tag "Instruct32Instruct3" \
  --template "llama3"

### Merge: R1-Distill2R1-Distill (adapter=R1-Distill, target=R1-Distill) ###
mkdir -p "$RESULTS_DIR/0327/result-weight-similarity-0327/R1-Distill-2k-lora-rank128-lr0.0002-shadow2k/merged-R1-Distill2R1-Distill"
python3 "$WORKSPACE_DIR/src/shadow/merge_lora.py" \
  --adapter_path "$RESULTS_DIR/0327/result-weight-similarity-0327/R1-Distill-2k-lora-rank128-lr0.0002-shadow2k" \
  --target_base "deepseek-ai/DeepSeek-R1-Distill-Llama-8B" \
  --merge_tag "R1-Distill2R1-Distill" \
  --template "llama3"

### Merge: Tulu3.12Tulu3.1 (adapter=Tulu3.1, target=Tulu3.1) ###
mkdir -p "$RESULTS_DIR/0327/result-weight-similarity-0327/Tulu3.1-2k-lora-rank128-lr0.0002-shadow2k/merged-Tulu3.12Tulu3.1"
python3 "$WORKSPACE_DIR/src/shadow/merge_lora.py" \
  --adapter_path "$RESULTS_DIR/0327/result-weight-similarity-0327/Tulu3.1-2k-lora-rank128-lr0.0002-shadow2k" \
  --target_base "allenai/Llama-3.1-Tulu-3.1-8B" \
  --merge_tag "Tulu3.12Tulu3.1" \
  --template "llama3"

### Merge: Tulu3-SFT2Tulu3-DPO (adapter=Tulu3-SFT, target=Tulu3-DPO) ###
# mkdir -p "$RESULTS_DIR/0327/result-weight-similarity-0327/Tulu3-SFT-2k-lora-rank128-lr0.0002-shadow2k/merged-Tulu3-SFT2Tulu3-DPO"
# python3 "$WORKSPACE_DIR/src/shadow/merge_lora.py" \
#   --adapter_path "$RESULTS_DIR/0327/result-weight-similarity-0327/Tulu3-SFT-2k-lora-rank128-lr0.0002-shadow2k" \
#   --target_base "allenai/Llama-3.1-Tulu-3-8B-DPO" \
#   --merge_tag "Tulu3-SFT2Tulu3-DPO" \
#   --template "llama3"

### Merge: Tulu3-DPO2Tulu3-SFT (adapter=Tulu3-DPO, target=Tulu3-SFT) ###
# mkdir -p "$RESULTS_DIR/0327/result-weight-similarity-0327/Tulu3-DPO-2k-lora-rank128-lr0.0002-shadow2k/merged-Tulu3-DPO2Tulu3-SFT"
# python3 "$WORKSPACE_DIR/src/shadow/merge_lora.py" \
#   --adapter_path "$RESULTS_DIR/0327/result-weight-similarity-0327/Tulu3-DPO-2k-lora-rank128-lr0.0002-shadow2k" \
#   --target_base "allenai/Llama-3.1-Tulu-3-8B-SFT" \
#   --merge_tag "Tulu3-DPO2Tulu3-SFT" \
#   --template "llama3"

### Merge: Tulu3-RLVR2Tulu3-DPO (adapter=Tulu3-RLVR, target=Tulu3-DPO) ###
# mkdir -p "$RESULTS_DIR/0327/result-weight-similarity-0327/Tulu3-RLVR-2k-lora-rank128-lr0.0002-shadow2k/merged-Tulu3-RLVR2Tulu3-DPO"
# python3 "$WORKSPACE_DIR/src/shadow/merge_lora.py" \
#   --adapter_path "$RESULTS_DIR/0327/result-weight-similarity-0327/Tulu3-RLVR-2k-lora-rank128-lr0.0002-shadow2k" \
#   --target_base "allenai/Llama-3.1-Tulu-3-8B-DPO" \
#   --merge_tag "Tulu3-RLVR2Tulu3-DPO" \
#   --template "llama3"

### Merge: Instruct3.12Base3.1 (adapter=Instruct3.1, target=Base3.1) ###
# mkdir -p "$RESULTS_DIR/0327/result-weight-similarity-0327/Instruct3.1-2k-lora-rank128-lr0.0002-shadow2k/merged-Instruct3.12Base3.1"
# python3 "$WORKSPACE_DIR/src/shadow/merge_lora.py" \
#   --adapter_path "$RESULTS_DIR/0327/result-weight-similarity-0327/Instruct3.1-2k-lora-rank128-lr0.0002-shadow2k" \
#   --target_base "meta-llama/Llama-3.1-8B" \
#   --merge_tag "Instruct3.12Base3.1" \
#   --template "llama3"

### Merge: Instruct32Base3 (adapter=Instruct3, target=Base3) ###
# mkdir -p "$RESULTS_DIR/0327/result-weight-similarity-0327/Instruct3-2k-lora-rank128-lr0.0002-shadow2k/merged-Instruct32Base3"
# python3 "$WORKSPACE_DIR/src/shadow/merge_lora.py" \
#   --adapter_path "$RESULTS_DIR/0327/result-weight-similarity-0327/Instruct3-2k-lora-rank128-lr0.0002-shadow2k" \
#   --target_base "meta-llama/Llama-3-8B" \
#   --merge_tag "Instruct32Base3" \
#   --template "llama3"

### Merge: R1-Distill2Instruct3.1 (adapter=R1-Distill, target=Instruct3.1) ###
# mkdir -p "$RESULTS_DIR/0327/result-weight-similarity-0327/R1-Distill-2k-lora-rank128-lr0.0002-shadow2k/merged-R1-Distill2Instruct3.1"
# python3 "$WORKSPACE_DIR/src/shadow/merge_lora.py" \
#   --adapter_path "$RESULTS_DIR/0327/result-weight-similarity-0327/R1-Distill-2k-lora-rank128-lr0.0002-shadow2k" \
#   --target_base "meta-llama/Llama-3.1-8B-Instruct" \
#   --merge_tag "R1-Distill2Instruct3.1" \
#   --template "llama3"

### Merge: R1-Distill2Base3.1 (adapter=R1-Distill, target=Base3.1) ###
# mkdir -p "$RESULTS_DIR/0327/result-weight-similarity-0327/R1-Distill-2k-lora-rank128-lr0.0002-shadow2k/merged-R1-Distill2Base3.1"
# python3 "$WORKSPACE_DIR/src/shadow/merge_lora.py" \
#   --adapter_path "$RESULTS_DIR/0327/result-weight-similarity-0327/R1-Distill-2k-lora-rank128-lr0.0002-shadow2k" \
#   --target_base "meta-llama/Llama-3.1-8B" \
#   --merge_tag "R1-Distill2Base3.1" \
#   --template "llama3"

### Merge: Tulu3.12Instruct3.1 (adapter=Tulu3.1, target=Instruct3.1) ###
# mkdir -p "$RESULTS_DIR/0327/result-weight-similarity-0327/Tulu3.1-2k-lora-rank128-lr0.0002-shadow2k/merged-Tulu3.12Instruct3.1"
# python3 "$WORKSPACE_DIR/src/shadow/merge_lora.py" \
#   --adapter_path "$RESULTS_DIR/0327/result-weight-similarity-0327/Tulu3.1-2k-lora-rank128-lr0.0002-shadow2k" \
#   --target_base "meta-llama/Llama-3.1-8B-Instruct" \
#   --merge_tag "Tulu3.12Instruct3.1" \
#   --template "llama3"

### Merge: Base3.12Instruct3 (adapter=Base3.1, target=Instruct3) ###
# mkdir -p "$RESULTS_DIR/0327/result-weight-similarity-0327/Base3.1-2k-lora-rank128-lr0.0002-shadow2k/merged-Base3.12Instruct3"
# python3 "$WORKSPACE_DIR/src/shadow/merge_lora.py" \
#   --adapter_path "$RESULTS_DIR/0327/result-weight-similarity-0327/Base3.1-2k-lora-rank128-lr0.0002-shadow2k" \
#   --target_base "meta-llama/Llama-3-8B-Instruct" \
#   --merge_tag "Base3.12Instruct3" \
#   --template "llama3"

### Merge: Base32Instruct3.1 (adapter=Base3, target=Instruct3.1) ###
# mkdir -p "$RESULTS_DIR/0327/result-weight-similarity-0327/Base3-2k-lora-rank128-lr0.0002-shadow2k/merged-Base32Instruct3.1"
# python3 "$WORKSPACE_DIR/src/shadow/merge_lora.py" \
#   --adapter_path "$RESULTS_DIR/0327/result-weight-similarity-0327/Base3-2k-lora-rank128-lr0.0002-shadow2k" \
#   --target_base "meta-llama/Llama-3.1-8B-Instruct" \
#   --merge_tag "Base32Instruct3.1" \
#   --template "llama3"

###############################################################################
##### Step 3: Compute pairwise weight similarity matrix                   #####
###############################################################################

# Run weight similarity analysis across all models
# This generates the σ matrix for comparison with eval performance
cd "$WORKSPACE_DIR"
python3 weight_similarity_matrix.py --models \
  Base3.1:meta-llama/Llama-3.1-8B \
  Instruct3.1:meta-llama/Llama-3.1-8B-Instruct \
  Tulu3-SFT:allenai/Llama-3.1-Tulu-3-8B-SFT \
  Tulu3-DPO:allenai/Llama-3.1-Tulu-3-8B-DPO \
  Tulu3-RLVR:allenai/Llama-3.1-Tulu-3-8B \
  Tulu3.1:allenai/Llama-3.1-Tulu-3.1-8B \
  Base3:meta-llama/Llama-3-8B \
  Instruct3:meta-llama/Llama-3-8B-Instruct \
  R1-Distill:deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
  --output_dir "$RESULTS_DIR/0327/result-weight-similarity-0327/weight_similarity"

###############################################################################
##### Step 4: Evaluation — model paths for OpenCompass                    #####
###############################################################################

# Copy the entries below into eval_weight_similarity.py
# or use the auto-generated config directly.

# === Original models (no training) ===
# ('Instruct3.1-orig', 'meta-llama/Llama-3.1-8B-Instruct'),
# ('Tulu3-SFT-orig', 'allenai/Llama-3.1-Tulu-3-8B-SFT'),
# ('Tulu3-DPO-orig', 'allenai/Llama-3.1-Tulu-3-8B-DPO'),
# ('Tulu3-RLVR-orig', 'allenai/Llama-3.1-Tulu-3-8B'),
# ('Tulu3.1-orig', 'allenai/Llama-3.1-Tulu-3.1-8B'),
# ('Instruct3-orig', 'meta-llama/Llama-3-8B-Instruct'),
# ('R1-Distill-orig', 'deepseek-ai/DeepSeek-R1-Distill-Llama-8B'),

# === Merged models ===
('Base3.12Instruct3.1','$RESULTS_DIR/0327/result-weight-similarity-0327/Base3.1-2k-lora-rank128-lr0.0002-shadow2k/merged-Base3.12Instruct3.1'),
('Tulu3-SFT2Instruct3.1','$RESULTS_DIR/0327/result-weight-similarity-0327/Tulu3-SFT-2k-lora-rank128-lr0.0002-shadow2k/merged-Tulu3-SFT2Instruct3.1'),
('Tulu3-DPO2Instruct3.1','$RESULTS_DIR/0327/result-weight-similarity-0327/Tulu3-DPO-2k-lora-rank128-lr0.0002-shadow2k/merged-Tulu3-DPO2Instruct3.1'),
('Tulu3-RLVR2Instruct3.1','$RESULTS_DIR/0327/result-weight-similarity-0327/Tulu3-RLVR-2k-lora-rank128-lr0.0002-shadow2k/merged-Tulu3-RLVR2Instruct3.1'),
('Instruct3.12Instruct3.1','$RESULTS_DIR/0327/result-weight-similarity-0327/Instruct3.1-2k-lora-rank128-lr0.0002-shadow2k/merged-Instruct3.12Instruct3.1'),
('Tulu3-SFT2Tulu3-SFT','$RESULTS_DIR/0327/result-weight-similarity-0327/Tulu3-SFT-2k-lora-rank128-lr0.0002-shadow2k/merged-Tulu3-SFT2Tulu3-SFT'),
('Tulu3-DPO2Tulu3-DPO','$RESULTS_DIR/0327/result-weight-similarity-0327/Tulu3-DPO-2k-lora-rank128-lr0.0002-shadow2k/merged-Tulu3-DPO2Tulu3-DPO'),
('Tulu3-RLVR2Tulu3-RLVR','$RESULTS_DIR/0327/result-weight-similarity-0327/Tulu3-RLVR-2k-lora-rank128-lr0.0002-shadow2k/merged-Tulu3-RLVR2Tulu3-RLVR'),
('Base3.12Tulu3-SFT','$RESULTS_DIR/0327/result-weight-similarity-0327/Base3.1-2k-lora-rank128-lr0.0002-shadow2k/merged-Base3.12Tulu3-SFT'),
('Base3.12Tulu3-DPO','$RESULTS_DIR/0327/result-weight-similarity-0327/Base3.1-2k-lora-rank128-lr0.0002-shadow2k/merged-Base3.12Tulu3-DPO'),
('Base3.12Tulu3-RLVR','$RESULTS_DIR/0327/result-weight-similarity-0327/Base3.1-2k-lora-rank128-lr0.0002-shadow2k/merged-Base3.12Tulu3-RLVR'),
('Base3.12Tulu3.1','$RESULTS_DIR/0327/result-weight-similarity-0327/Base3.1-2k-lora-rank128-lr0.0002-shadow2k/merged-Base3.12Tulu3.1'),
('Base32Instruct3','$RESULTS_DIR/0327/result-weight-similarity-0327/Base3-2k-lora-rank128-lr0.0002-shadow2k/merged-Base32Instruct3'),
('Base32R1-Distill','$RESULTS_DIR/0327/result-weight-similarity-0327/Base3-2k-lora-rank128-lr0.0002-shadow2k/merged-Base32R1-Distill'),
('Base3.12R1-Distill','$RESULTS_DIR/0327/result-weight-similarity-0327/Base3.1-2k-lora-rank128-lr0.0002-shadow2k/merged-Base3.12R1-Distill'),
('Instruct32Instruct3','$RESULTS_DIR/0327/result-weight-similarity-0327/Instruct3-2k-lora-rank128-lr0.0002-shadow2k/merged-Instruct32Instruct3'),
('R1-Distill2R1-Distill','$RESULTS_DIR/0327/result-weight-similarity-0327/R1-Distill-2k-lora-rank128-lr0.0002-shadow2k/merged-R1-Distill2R1-Distill'),
('Tulu3.12Tulu3.1','$RESULTS_DIR/0327/result-weight-similarity-0327/Tulu3.1-2k-lora-rank128-lr0.0002-shadow2k/merged-Tulu3.12Tulu3.1'),
# ('Tulu3-SFT2Tulu3-DPO','$RESULTS_DIR/0327/result-weight-similarity-0327/Tulu3-SFT-2k-lora-rank128-lr0.0002-shadow2k/merged-Tulu3-SFT2Tulu3-DPO'),
# ('Tulu3-DPO2Tulu3-SFT','$RESULTS_DIR/0327/result-weight-similarity-0327/Tulu3-DPO-2k-lora-rank128-lr0.0002-shadow2k/merged-Tulu3-DPO2Tulu3-SFT'),
# ('Tulu3-RLVR2Tulu3-DPO','$RESULTS_DIR/0327/result-weight-similarity-0327/Tulu3-RLVR-2k-lora-rank128-lr0.0002-shadow2k/merged-Tulu3-RLVR2Tulu3-DPO'),
# ('Instruct3.12Base3.1','$RESULTS_DIR/0327/result-weight-similarity-0327/Instruct3.1-2k-lora-rank128-lr0.0002-shadow2k/merged-Instruct3.12Base3.1'),
# ('Instruct32Base3','$RESULTS_DIR/0327/result-weight-similarity-0327/Instruct3-2k-lora-rank128-lr0.0002-shadow2k/merged-Instruct32Base3'),
# ('R1-Distill2Instruct3.1','$RESULTS_DIR/0327/result-weight-similarity-0327/R1-Distill-2k-lora-rank128-lr0.0002-shadow2k/merged-R1-Distill2Instruct3.1'),
# ('R1-Distill2Base3.1','$RESULTS_DIR/0327/result-weight-similarity-0327/R1-Distill-2k-lora-rank128-lr0.0002-shadow2k/merged-R1-Distill2Base3.1'),
# ('Tulu3.12Instruct3.1','$RESULTS_DIR/0327/result-weight-similarity-0327/Tulu3.1-2k-lora-rank128-lr0.0002-shadow2k/merged-Tulu3.12Instruct3.1'),
# ('Base3.12Instruct3','$RESULTS_DIR/0327/result-weight-similarity-0327/Base3.1-2k-lora-rank128-lr0.0002-shadow2k/merged-Base3.12Instruct3'),
# ('Base32Instruct3.1','$RESULTS_DIR/0327/result-weight-similarity-0327/Base3-2k-lora-rank128-lr0.0002-shadow2k/merged-Base32Instruct3.1'),

