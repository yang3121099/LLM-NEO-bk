#!/usr/bin/env bash
set -euo pipefail

##### Auto-generated 2026-03-27 06:02:04 #####
# Cross-Delta Transfer Experiment
# Sources: Base SFT DPO RLVR
# Targets: Base SFT DPO RLVR Instruct Tulu3.1 Llama3-Inst R1-Distill
# Dataset: Shadow_2k (2000 samples), LoRA rank 128

WORKSPACE_DIR="$(cd "$(dirname "$0")/.." && pwd)"
RESULTS_DIR="$WORKSPACE_DIR/results"

export VLLM_WORKER_MULTIPROC_METHOD=spawn
export HF_HUB_OFFLINE=0
export HF_DATASETS_OFFLINE=0
export HF_DATASETS_TRUST_REMOTE_CODE=1
export TRUST_REMOTE_CODE=True
export HF_ALLOW_CODE_EVAL=1

###############################################################################
##### Step 1: Train LoRA on 4 source models                              #####
###############################################################################

###### Train LoRA: src-Base (meta-llama/Llama-3.1-8B) ######
mkdir -p "$RESULTS_DIR/0327/cross-delta-0327/src-Base-lora-r128-shadow2k"
cd "$WORKSPACE_DIR"
llamafactory-cli train \
  --model_name_or_path "meta-llama/Llama-3.1-8B" \
  --stage sft --do_train true \
  --finetuning_type lora --lora_rank 128 \
  --dataset "Shadow_2k" --template "llama3" \
  --cutoff_len 16384 --max_samples 2000 \
  --output_dir "$RESULTS_DIR/0327/cross-delta-0327/src-Base-lora-r128-shadow2k" \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 256 \
  --learning_rate 0.0002 --num_train_epochs 1 \
  --logging_steps 1 --save_steps 1000 --save_only_model True \
  --plot_loss true --lr_scheduler_type cosine --warmup_ratio 0.1 \
  --bf16 true --val_size 0.01 \
  --per_device_eval_batch_size 1 --eval_strategy steps --eval_steps 10000 \
  --trust_remote_code True --flash_attn fa2 \
  --overwrite_cache false --use_fast_tokenizer True

###### Train LoRA: src-SFT (allenai/Llama-3.1-Tulu-3-8B-SFT) ######
mkdir -p "$RESULTS_DIR/0327/cross-delta-0327/src-SFT-lora-r128-shadow2k"
cd "$WORKSPACE_DIR"
llamafactory-cli train \
  --model_name_or_path "allenai/Llama-3.1-Tulu-3-8B-SFT" \
  --stage sft --do_train true \
  --finetuning_type lora --lora_rank 128 \
  --dataset "Shadow_2k" --template "llama3" \
  --cutoff_len 16384 --max_samples 2000 \
  --output_dir "$RESULTS_DIR/0327/cross-delta-0327/src-SFT-lora-r128-shadow2k" \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 256 \
  --learning_rate 0.0002 --num_train_epochs 1 \
  --logging_steps 1 --save_steps 1000 --save_only_model True \
  --plot_loss true --lr_scheduler_type cosine --warmup_ratio 0.1 \
  --bf16 true --val_size 0.01 \
  --per_device_eval_batch_size 1 --eval_strategy steps --eval_steps 10000 \
  --trust_remote_code True --flash_attn fa2 \
  --overwrite_cache false --use_fast_tokenizer True

###### Train LoRA: src-DPO (allenai/Llama-3.1-Tulu-3-8B-DPO) ######
mkdir -p "$RESULTS_DIR/0327/cross-delta-0327/src-DPO-lora-r128-shadow2k"
cd "$WORKSPACE_DIR"
llamafactory-cli train \
  --model_name_or_path "allenai/Llama-3.1-Tulu-3-8B-DPO" \
  --stage sft --do_train true \
  --finetuning_type lora --lora_rank 128 \
  --dataset "Shadow_2k" --template "llama3" \
  --cutoff_len 16384 --max_samples 2000 \
  --output_dir "$RESULTS_DIR/0327/cross-delta-0327/src-DPO-lora-r128-shadow2k" \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 256 \
  --learning_rate 0.0002 --num_train_epochs 1 \
  --logging_steps 1 --save_steps 1000 --save_only_model True \
  --plot_loss true --lr_scheduler_type cosine --warmup_ratio 0.1 \
  --bf16 true --val_size 0.01 \
  --per_device_eval_batch_size 1 --eval_strategy steps --eval_steps 10000 \
  --trust_remote_code True --flash_attn fa2 \
  --overwrite_cache false --use_fast_tokenizer True

###### Train LoRA: src-RLVR (allenai/Llama-3.1-Tulu-3-8B) ######
mkdir -p "$RESULTS_DIR/0327/cross-delta-0327/src-RLVR-lora-r128-shadow2k"
cd "$WORKSPACE_DIR"
llamafactory-cli train \
  --model_name_or_path "allenai/Llama-3.1-Tulu-3-8B" \
  --stage sft --do_train true \
  --finetuning_type lora --lora_rank 128 \
  --dataset "Shadow_2k" --template "llama3" \
  --cutoff_len 16384 --max_samples 2000 \
  --output_dir "$RESULTS_DIR/0327/cross-delta-0327/src-RLVR-lora-r128-shadow2k" \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 256 \
  --learning_rate 0.0002 --num_train_epochs 1 \
  --logging_steps 1 --save_steps 1000 --save_only_model True \
  --plot_loss true --lr_scheduler_type cosine --warmup_ratio 0.1 \
  --bf16 true --val_size 0.01 \
  --per_device_eval_batch_size 1 --eval_strategy steps --eval_steps 10000 \
  --trust_remote_code True --flash_attn fa2 \
  --overwrite_cache false --use_fast_tokenizer True

###############################################################################
##### Step 2: Direct-FT baselines on extra targets                       #####
###############################################################################

###### Train LoRA: tgt-Instruct (meta-llama/Llama-3.1-8B-Instruct) ######
mkdir -p "$RESULTS_DIR/0327/cross-delta-0327/tgt-Instruct-lora-r128-shadow2k"
cd "$WORKSPACE_DIR"
llamafactory-cli train \
  --model_name_or_path "meta-llama/Llama-3.1-8B-Instruct" \
  --stage sft --do_train true \
  --finetuning_type lora --lora_rank 128 \
  --dataset "Shadow_2k" --template "llama3" \
  --cutoff_len 16384 --max_samples 2000 \
  --output_dir "$RESULTS_DIR/0327/cross-delta-0327/tgt-Instruct-lora-r128-shadow2k" \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 256 \
  --learning_rate 0.0002 --num_train_epochs 1 \
  --logging_steps 1 --save_steps 1000 --save_only_model True \
  --plot_loss true --lr_scheduler_type cosine --warmup_ratio 0.1 \
  --bf16 true --val_size 0.01 \
  --per_device_eval_batch_size 1 --eval_strategy steps --eval_steps 10000 \
  --trust_remote_code True --flash_attn fa2 \
  --overwrite_cache false --use_fast_tokenizer True

###### Train LoRA: tgt-Tulu3.1 (allenai/Llama-3.1-Tulu-3.1-8B) ######
mkdir -p "$RESULTS_DIR/0327/cross-delta-0327/tgt-Tulu3.1-lora-r128-shadow2k"
cd "$WORKSPACE_DIR"
llamafactory-cli train \
  --model_name_or_path "allenai/Llama-3.1-Tulu-3.1-8B" \
  --stage sft --do_train true \
  --finetuning_type lora --lora_rank 128 \
  --dataset "Shadow_2k" --template "llama3" \
  --cutoff_len 16384 --max_samples 2000 \
  --output_dir "$RESULTS_DIR/0327/cross-delta-0327/tgt-Tulu3.1-lora-r128-shadow2k" \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 256 \
  --learning_rate 0.0002 --num_train_epochs 1 \
  --logging_steps 1 --save_steps 1000 --save_only_model True \
  --plot_loss true --lr_scheduler_type cosine --warmup_ratio 0.1 \
  --bf16 true --val_size 0.01 \
  --per_device_eval_batch_size 1 --eval_strategy steps --eval_steps 10000 \
  --trust_remote_code True --flash_attn fa2 \
  --overwrite_cache false --use_fast_tokenizer True

###### Train LoRA: tgt-Llama3-Inst (meta-llama/Meta-Llama-3-8B-Instruct) ######
mkdir -p "$RESULTS_DIR/0327/cross-delta-0327/tgt-Llama3-Inst-lora-r128-shadow2k"
cd "$WORKSPACE_DIR"
llamafactory-cli train \
  --model_name_or_path "meta-llama/Meta-Llama-3-8B-Instruct" \
  --stage sft --do_train true \
  --finetuning_type lora --lora_rank 128 \
  --dataset "Shadow_2k" --template "llama3" \
  --cutoff_len 16384 --max_samples 2000 \
  --output_dir "$RESULTS_DIR/0327/cross-delta-0327/tgt-Llama3-Inst-lora-r128-shadow2k" \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 256 \
  --learning_rate 0.0002 --num_train_epochs 1 \
  --logging_steps 1 --save_steps 1000 --save_only_model True \
  --plot_loss true --lr_scheduler_type cosine --warmup_ratio 0.1 \
  --bf16 true --val_size 0.01 \
  --per_device_eval_batch_size 1 --eval_strategy steps --eval_steps 10000 \
  --trust_remote_code True --flash_attn fa2 \
  --overwrite_cache false --use_fast_tokenizer True

###### Train LoRA: tgt-R1-Distill (deepseek-ai/DeepSeek-R1-Distill-Llama-8B) ######
mkdir -p "$RESULTS_DIR/0327/cross-delta-0327/tgt-R1-Distill-lora-r128-shadow2k"
cd "$WORKSPACE_DIR"
llamafactory-cli train \
  --model_name_or_path "deepseek-ai/DeepSeek-R1-Distill-Llama-8B" \
  --stage sft --do_train true \
  --finetuning_type lora --lora_rank 128 \
  --dataset "Shadow_2k" --template "llama3" \
  --cutoff_len 16384 --max_samples 2000 \
  --output_dir "$RESULTS_DIR/0327/cross-delta-0327/tgt-R1-Distill-lora-r128-shadow2k" \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 256 \
  --learning_rate 0.0002 --num_train_epochs 1 \
  --logging_steps 1 --save_steps 1000 --save_only_model True \
  --plot_loss true --lr_scheduler_type cosine --warmup_ratio 0.1 \
  --bf16 true --val_size 0.01 \
  --per_device_eval_batch_size 1 --eval_strategy steps --eval_steps 10000 \
  --trust_remote_code True --flash_attn fa2 \
  --overwrite_cache false --use_fast_tokenizer True

###############################################################################
##### Step 3: Cross-model delta merge (4×8 = 32 combinations)           #####
###############################################################################

# merge_lora.py applies the LoRA delta from source training onto target model.
# LoRA adapters only touch attention/MLP projections — embed_tokens and
# lm_head are NOT modified. Shape mismatches are auto-skipped.

### Merge: Base2Base ###
mkdir -p "$RESULTS_DIR/0327/cross-delta-0327/src-Base-lora-r128-shadow2k/merged-Base2Base"
python3 "$WORKSPACE_DIR/src/shadow/merge_lora.py" \
  --adapter_path "$RESULTS_DIR/0327/cross-delta-0327/src-Base-lora-r128-shadow2k" \
  --target_base "meta-llama/Llama-3.1-8B" \
  --merge_tag "Base2Base" \
  --template "llama3"

### Merge: Base2SFT ###
mkdir -p "$RESULTS_DIR/0327/cross-delta-0327/src-Base-lora-r128-shadow2k/merged-Base2SFT"
python3 "$WORKSPACE_DIR/src/shadow/merge_lora.py" \
  --adapter_path "$RESULTS_DIR/0327/cross-delta-0327/src-Base-lora-r128-shadow2k" \
  --target_base "allenai/Llama-3.1-Tulu-3-8B-SFT" \
  --merge_tag "Base2SFT" \
  --template "llama3"

### Merge: Base2DPO ###
mkdir -p "$RESULTS_DIR/0327/cross-delta-0327/src-Base-lora-r128-shadow2k/merged-Base2DPO"
python3 "$WORKSPACE_DIR/src/shadow/merge_lora.py" \
  --adapter_path "$RESULTS_DIR/0327/cross-delta-0327/src-Base-lora-r128-shadow2k" \
  --target_base "allenai/Llama-3.1-Tulu-3-8B-DPO" \
  --merge_tag "Base2DPO" \
  --template "llama3"

### Merge: Base2RLVR ###
mkdir -p "$RESULTS_DIR/0327/cross-delta-0327/src-Base-lora-r128-shadow2k/merged-Base2RLVR"
python3 "$WORKSPACE_DIR/src/shadow/merge_lora.py" \
  --adapter_path "$RESULTS_DIR/0327/cross-delta-0327/src-Base-lora-r128-shadow2k" \
  --target_base "allenai/Llama-3.1-Tulu-3-8B" \
  --merge_tag "Base2RLVR" \
  --template "llama3"

### Merge: Base2Instruct ###
mkdir -p "$RESULTS_DIR/0327/cross-delta-0327/src-Base-lora-r128-shadow2k/merged-Base2Instruct"
python3 "$WORKSPACE_DIR/src/shadow/merge_lora.py" \
  --adapter_path "$RESULTS_DIR/0327/cross-delta-0327/src-Base-lora-r128-shadow2k" \
  --target_base "meta-llama/Llama-3.1-8B-Instruct" \
  --merge_tag "Base2Instruct" \
  --template "llama3"

### Merge: Base2Tulu3.1 ###
mkdir -p "$RESULTS_DIR/0327/cross-delta-0327/src-Base-lora-r128-shadow2k/merged-Base2Tulu3.1"
python3 "$WORKSPACE_DIR/src/shadow/merge_lora.py" \
  --adapter_path "$RESULTS_DIR/0327/cross-delta-0327/src-Base-lora-r128-shadow2k" \
  --target_base "allenai/Llama-3.1-Tulu-3.1-8B" \
  --merge_tag "Base2Tulu3.1" \
  --template "llama3"

### Merge: Base2Llama3-Inst ###
mkdir -p "$RESULTS_DIR/0327/cross-delta-0327/src-Base-lora-r128-shadow2k/merged-Base2Llama3-Inst"
python3 "$WORKSPACE_DIR/src/shadow/merge_lora.py" \
  --adapter_path "$RESULTS_DIR/0327/cross-delta-0327/src-Base-lora-r128-shadow2k" \
  --target_base "meta-llama/Meta-Llama-3-8B-Instruct" \
  --merge_tag "Base2Llama3-Inst" \
  --template "llama3"

### Merge: Base2R1-Distill ###
mkdir -p "$RESULTS_DIR/0327/cross-delta-0327/src-Base-lora-r128-shadow2k/merged-Base2R1-Distill"
python3 "$WORKSPACE_DIR/src/shadow/merge_lora.py" \
  --adapter_path "$RESULTS_DIR/0327/cross-delta-0327/src-Base-lora-r128-shadow2k" \
  --target_base "deepseek-ai/DeepSeek-R1-Distill-Llama-8B" \
  --merge_tag "Base2R1-Distill" \
  --template "llama3"

### Merge: SFT2Base ###
mkdir -p "$RESULTS_DIR/0327/cross-delta-0327/src-SFT-lora-r128-shadow2k/merged-SFT2Base"
python3 "$WORKSPACE_DIR/src/shadow/merge_lora.py" \
  --adapter_path "$RESULTS_DIR/0327/cross-delta-0327/src-SFT-lora-r128-shadow2k" \
  --target_base "meta-llama/Llama-3.1-8B" \
  --merge_tag "SFT2Base" \
  --template "llama3"

### Merge: SFT2SFT ###
mkdir -p "$RESULTS_DIR/0327/cross-delta-0327/src-SFT-lora-r128-shadow2k/merged-SFT2SFT"
python3 "$WORKSPACE_DIR/src/shadow/merge_lora.py" \
  --adapter_path "$RESULTS_DIR/0327/cross-delta-0327/src-SFT-lora-r128-shadow2k" \
  --target_base "allenai/Llama-3.1-Tulu-3-8B-SFT" \
  --merge_tag "SFT2SFT" \
  --template "llama3"

### Merge: SFT2DPO ###
mkdir -p "$RESULTS_DIR/0327/cross-delta-0327/src-SFT-lora-r128-shadow2k/merged-SFT2DPO"
python3 "$WORKSPACE_DIR/src/shadow/merge_lora.py" \
  --adapter_path "$RESULTS_DIR/0327/cross-delta-0327/src-SFT-lora-r128-shadow2k" \
  --target_base "allenai/Llama-3.1-Tulu-3-8B-DPO" \
  --merge_tag "SFT2DPO" \
  --template "llama3"

### Merge: SFT2RLVR ###
mkdir -p "$RESULTS_DIR/0327/cross-delta-0327/src-SFT-lora-r128-shadow2k/merged-SFT2RLVR"
python3 "$WORKSPACE_DIR/src/shadow/merge_lora.py" \
  --adapter_path "$RESULTS_DIR/0327/cross-delta-0327/src-SFT-lora-r128-shadow2k" \
  --target_base "allenai/Llama-3.1-Tulu-3-8B" \
  --merge_tag "SFT2RLVR" \
  --template "llama3"

### Merge: SFT2Instruct ###
mkdir -p "$RESULTS_DIR/0327/cross-delta-0327/src-SFT-lora-r128-shadow2k/merged-SFT2Instruct"
python3 "$WORKSPACE_DIR/src/shadow/merge_lora.py" \
  --adapter_path "$RESULTS_DIR/0327/cross-delta-0327/src-SFT-lora-r128-shadow2k" \
  --target_base "meta-llama/Llama-3.1-8B-Instruct" \
  --merge_tag "SFT2Instruct" \
  --template "llama3"

### Merge: SFT2Tulu3.1 ###
mkdir -p "$RESULTS_DIR/0327/cross-delta-0327/src-SFT-lora-r128-shadow2k/merged-SFT2Tulu3.1"
python3 "$WORKSPACE_DIR/src/shadow/merge_lora.py" \
  --adapter_path "$RESULTS_DIR/0327/cross-delta-0327/src-SFT-lora-r128-shadow2k" \
  --target_base "allenai/Llama-3.1-Tulu-3.1-8B" \
  --merge_tag "SFT2Tulu3.1" \
  --template "llama3"

### Merge: SFT2Llama3-Inst ###
mkdir -p "$RESULTS_DIR/0327/cross-delta-0327/src-SFT-lora-r128-shadow2k/merged-SFT2Llama3-Inst"
python3 "$WORKSPACE_DIR/src/shadow/merge_lora.py" \
  --adapter_path "$RESULTS_DIR/0327/cross-delta-0327/src-SFT-lora-r128-shadow2k" \
  --target_base "meta-llama/Meta-Llama-3-8B-Instruct" \
  --merge_tag "SFT2Llama3-Inst" \
  --template "llama3"

### Merge: SFT2R1-Distill ###
mkdir -p "$RESULTS_DIR/0327/cross-delta-0327/src-SFT-lora-r128-shadow2k/merged-SFT2R1-Distill"
python3 "$WORKSPACE_DIR/src/shadow/merge_lora.py" \
  --adapter_path "$RESULTS_DIR/0327/cross-delta-0327/src-SFT-lora-r128-shadow2k" \
  --target_base "deepseek-ai/DeepSeek-R1-Distill-Llama-8B" \
  --merge_tag "SFT2R1-Distill" \
  --template "llama3"

### Merge: DPO2Base ###
mkdir -p "$RESULTS_DIR/0327/cross-delta-0327/src-DPO-lora-r128-shadow2k/merged-DPO2Base"
python3 "$WORKSPACE_DIR/src/shadow/merge_lora.py" \
  --adapter_path "$RESULTS_DIR/0327/cross-delta-0327/src-DPO-lora-r128-shadow2k" \
  --target_base "meta-llama/Llama-3.1-8B" \
  --merge_tag "DPO2Base" \
  --template "llama3"

### Merge: DPO2SFT ###
mkdir -p "$RESULTS_DIR/0327/cross-delta-0327/src-DPO-lora-r128-shadow2k/merged-DPO2SFT"
python3 "$WORKSPACE_DIR/src/shadow/merge_lora.py" \
  --adapter_path "$RESULTS_DIR/0327/cross-delta-0327/src-DPO-lora-r128-shadow2k" \
  --target_base "allenai/Llama-3.1-Tulu-3-8B-SFT" \
  --merge_tag "DPO2SFT" \
  --template "llama3"

### Merge: DPO2DPO ###
mkdir -p "$RESULTS_DIR/0327/cross-delta-0327/src-DPO-lora-r128-shadow2k/merged-DPO2DPO"
python3 "$WORKSPACE_DIR/src/shadow/merge_lora.py" \
  --adapter_path "$RESULTS_DIR/0327/cross-delta-0327/src-DPO-lora-r128-shadow2k" \
  --target_base "allenai/Llama-3.1-Tulu-3-8B-DPO" \
  --merge_tag "DPO2DPO" \
  --template "llama3"

### Merge: DPO2RLVR ###
mkdir -p "$RESULTS_DIR/0327/cross-delta-0327/src-DPO-lora-r128-shadow2k/merged-DPO2RLVR"
python3 "$WORKSPACE_DIR/src/shadow/merge_lora.py" \
  --adapter_path "$RESULTS_DIR/0327/cross-delta-0327/src-DPO-lora-r128-shadow2k" \
  --target_base "allenai/Llama-3.1-Tulu-3-8B" \
  --merge_tag "DPO2RLVR" \
  --template "llama3"

### Merge: DPO2Instruct ###
mkdir -p "$RESULTS_DIR/0327/cross-delta-0327/src-DPO-lora-r128-shadow2k/merged-DPO2Instruct"
python3 "$WORKSPACE_DIR/src/shadow/merge_lora.py" \
  --adapter_path "$RESULTS_DIR/0327/cross-delta-0327/src-DPO-lora-r128-shadow2k" \
  --target_base "meta-llama/Llama-3.1-8B-Instruct" \
  --merge_tag "DPO2Instruct" \
  --template "llama3"

### Merge: DPO2Tulu3.1 ###
mkdir -p "$RESULTS_DIR/0327/cross-delta-0327/src-DPO-lora-r128-shadow2k/merged-DPO2Tulu3.1"
python3 "$WORKSPACE_DIR/src/shadow/merge_lora.py" \
  --adapter_path "$RESULTS_DIR/0327/cross-delta-0327/src-DPO-lora-r128-shadow2k" \
  --target_base "allenai/Llama-3.1-Tulu-3.1-8B" \
  --merge_tag "DPO2Tulu3.1" \
  --template "llama3"

### Merge: DPO2Llama3-Inst ###
mkdir -p "$RESULTS_DIR/0327/cross-delta-0327/src-DPO-lora-r128-shadow2k/merged-DPO2Llama3-Inst"
python3 "$WORKSPACE_DIR/src/shadow/merge_lora.py" \
  --adapter_path "$RESULTS_DIR/0327/cross-delta-0327/src-DPO-lora-r128-shadow2k" \
  --target_base "meta-llama/Meta-Llama-3-8B-Instruct" \
  --merge_tag "DPO2Llama3-Inst" \
  --template "llama3"

### Merge: DPO2R1-Distill ###
mkdir -p "$RESULTS_DIR/0327/cross-delta-0327/src-DPO-lora-r128-shadow2k/merged-DPO2R1-Distill"
python3 "$WORKSPACE_DIR/src/shadow/merge_lora.py" \
  --adapter_path "$RESULTS_DIR/0327/cross-delta-0327/src-DPO-lora-r128-shadow2k" \
  --target_base "deepseek-ai/DeepSeek-R1-Distill-Llama-8B" \
  --merge_tag "DPO2R1-Distill" \
  --template "llama3"

### Merge: RLVR2Base ###
mkdir -p "$RESULTS_DIR/0327/cross-delta-0327/src-RLVR-lora-r128-shadow2k/merged-RLVR2Base"
python3 "$WORKSPACE_DIR/src/shadow/merge_lora.py" \
  --adapter_path "$RESULTS_DIR/0327/cross-delta-0327/src-RLVR-lora-r128-shadow2k" \
  --target_base "meta-llama/Llama-3.1-8B" \
  --merge_tag "RLVR2Base" \
  --template "llama3"

### Merge: RLVR2SFT ###
mkdir -p "$RESULTS_DIR/0327/cross-delta-0327/src-RLVR-lora-r128-shadow2k/merged-RLVR2SFT"
python3 "$WORKSPACE_DIR/src/shadow/merge_lora.py" \
  --adapter_path "$RESULTS_DIR/0327/cross-delta-0327/src-RLVR-lora-r128-shadow2k" \
  --target_base "allenai/Llama-3.1-Tulu-3-8B-SFT" \
  --merge_tag "RLVR2SFT" \
  --template "llama3"

### Merge: RLVR2DPO ###
mkdir -p "$RESULTS_DIR/0327/cross-delta-0327/src-RLVR-lora-r128-shadow2k/merged-RLVR2DPO"
python3 "$WORKSPACE_DIR/src/shadow/merge_lora.py" \
  --adapter_path "$RESULTS_DIR/0327/cross-delta-0327/src-RLVR-lora-r128-shadow2k" \
  --target_base "allenai/Llama-3.1-Tulu-3-8B-DPO" \
  --merge_tag "RLVR2DPO" \
  --template "llama3"

### Merge: RLVR2RLVR ###
mkdir -p "$RESULTS_DIR/0327/cross-delta-0327/src-RLVR-lora-r128-shadow2k/merged-RLVR2RLVR"
python3 "$WORKSPACE_DIR/src/shadow/merge_lora.py" \
  --adapter_path "$RESULTS_DIR/0327/cross-delta-0327/src-RLVR-lora-r128-shadow2k" \
  --target_base "allenai/Llama-3.1-Tulu-3-8B" \
  --merge_tag "RLVR2RLVR" \
  --template "llama3"

### Merge: RLVR2Instruct ###
mkdir -p "$RESULTS_DIR/0327/cross-delta-0327/src-RLVR-lora-r128-shadow2k/merged-RLVR2Instruct"
python3 "$WORKSPACE_DIR/src/shadow/merge_lora.py" \
  --adapter_path "$RESULTS_DIR/0327/cross-delta-0327/src-RLVR-lora-r128-shadow2k" \
  --target_base "meta-llama/Llama-3.1-8B-Instruct" \
  --merge_tag "RLVR2Instruct" \
  --template "llama3"

### Merge: RLVR2Tulu3.1 ###
mkdir -p "$RESULTS_DIR/0327/cross-delta-0327/src-RLVR-lora-r128-shadow2k/merged-RLVR2Tulu3.1"
python3 "$WORKSPACE_DIR/src/shadow/merge_lora.py" \
  --adapter_path "$RESULTS_DIR/0327/cross-delta-0327/src-RLVR-lora-r128-shadow2k" \
  --target_base "allenai/Llama-3.1-Tulu-3.1-8B" \
  --merge_tag "RLVR2Tulu3.1" \
  --template "llama3"

### Merge: RLVR2Llama3-Inst ###
mkdir -p "$RESULTS_DIR/0327/cross-delta-0327/src-RLVR-lora-r128-shadow2k/merged-RLVR2Llama3-Inst"
python3 "$WORKSPACE_DIR/src/shadow/merge_lora.py" \
  --adapter_path "$RESULTS_DIR/0327/cross-delta-0327/src-RLVR-lora-r128-shadow2k" \
  --target_base "meta-llama/Meta-Llama-3-8B-Instruct" \
  --merge_tag "RLVR2Llama3-Inst" \
  --template "llama3"

### Merge: RLVR2R1-Distill ###
mkdir -p "$RESULTS_DIR/0327/cross-delta-0327/src-RLVR-lora-r128-shadow2k/merged-RLVR2R1-Distill"
python3 "$WORKSPACE_DIR/src/shadow/merge_lora.py" \
  --adapter_path "$RESULTS_DIR/0327/cross-delta-0327/src-RLVR-lora-r128-shadow2k" \
  --target_base "deepseek-ai/DeepSeek-R1-Distill-Llama-8B" \
  --merge_tag "RLVR2R1-Distill" \
  --template "llama3"

### Baseline: DirectFT-Base ###
mkdir -p "$RESULTS_DIR/0327/cross-delta-0327/src-Base-lora-r128-shadow2k/merged-DirectFT-Base"
python3 "$WORKSPACE_DIR/src/shadow/merge_lora.py" \
  --adapter_path "$RESULTS_DIR/0327/cross-delta-0327/src-Base-lora-r128-shadow2k" \
  --target_base "meta-llama/Llama-3.1-8B" \
  --merge_tag "DirectFT-Base" \
  --template "llama3"

### Baseline: DirectFT-SFT ###
mkdir -p "$RESULTS_DIR/0327/cross-delta-0327/src-SFT-lora-r128-shadow2k/merged-DirectFT-SFT"
python3 "$WORKSPACE_DIR/src/shadow/merge_lora.py" \
  --adapter_path "$RESULTS_DIR/0327/cross-delta-0327/src-SFT-lora-r128-shadow2k" \
  --target_base "allenai/Llama-3.1-Tulu-3-8B-SFT" \
  --merge_tag "DirectFT-SFT" \
  --template "llama3"

### Baseline: DirectFT-DPO ###
mkdir -p "$RESULTS_DIR/0327/cross-delta-0327/src-DPO-lora-r128-shadow2k/merged-DirectFT-DPO"
python3 "$WORKSPACE_DIR/src/shadow/merge_lora.py" \
  --adapter_path "$RESULTS_DIR/0327/cross-delta-0327/src-DPO-lora-r128-shadow2k" \
  --target_base "allenai/Llama-3.1-Tulu-3-8B-DPO" \
  --merge_tag "DirectFT-DPO" \
  --template "llama3"

### Baseline: DirectFT-RLVR ###
mkdir -p "$RESULTS_DIR/0327/cross-delta-0327/src-RLVR-lora-r128-shadow2k/merged-DirectFT-RLVR"
python3 "$WORKSPACE_DIR/src/shadow/merge_lora.py" \
  --adapter_path "$RESULTS_DIR/0327/cross-delta-0327/src-RLVR-lora-r128-shadow2k" \
  --target_base "allenai/Llama-3.1-Tulu-3-8B" \
  --merge_tag "DirectFT-RLVR" \
  --template "llama3"

### Baseline: DirectFT-Instruct ###
mkdir -p "$RESULTS_DIR/0327/cross-delta-0327/tgt-Instruct-lora-r128-shadow2k/merged-DirectFT-Instruct"
python3 "$WORKSPACE_DIR/src/shadow/merge_lora.py" \
  --adapter_path "$RESULTS_DIR/0327/cross-delta-0327/tgt-Instruct-lora-r128-shadow2k" \
  --target_base "meta-llama/Llama-3.1-8B-Instruct" \
  --merge_tag "DirectFT-Instruct" \
  --template "llama3"

### Baseline: DirectFT-Tulu3.1 ###
mkdir -p "$RESULTS_DIR/0327/cross-delta-0327/tgt-Tulu3.1-lora-r128-shadow2k/merged-DirectFT-Tulu3.1"
python3 "$WORKSPACE_DIR/src/shadow/merge_lora.py" \
  --adapter_path "$RESULTS_DIR/0327/cross-delta-0327/tgt-Tulu3.1-lora-r128-shadow2k" \
  --target_base "allenai/Llama-3.1-Tulu-3.1-8B" \
  --merge_tag "DirectFT-Tulu3.1" \
  --template "llama3"

### Baseline: DirectFT-Llama3-Inst ###
mkdir -p "$RESULTS_DIR/0327/cross-delta-0327/tgt-Llama3-Inst-lora-r128-shadow2k/merged-DirectFT-Llama3-Inst"
python3 "$WORKSPACE_DIR/src/shadow/merge_lora.py" \
  --adapter_path "$RESULTS_DIR/0327/cross-delta-0327/tgt-Llama3-Inst-lora-r128-shadow2k" \
  --target_base "meta-llama/Meta-Llama-3-8B-Instruct" \
  --merge_tag "DirectFT-Llama3-Inst" \
  --template "llama3"

### Baseline: DirectFT-R1-Distill ###
mkdir -p "$RESULTS_DIR/0327/cross-delta-0327/tgt-R1-Distill-lora-r128-shadow2k/merged-DirectFT-R1-Distill"
python3 "$WORKSPACE_DIR/src/shadow/merge_lora.py" \
  --adapter_path "$RESULTS_DIR/0327/cross-delta-0327/tgt-R1-Distill-lora-r128-shadow2k" \
  --target_base "deepseek-ai/DeepSeek-R1-Distill-Llama-8B" \
  --merge_tag "DirectFT-R1-Distill" \
  --template "llama3"

###############################################################################
##### Step 4: Weight similarity matrix (σ)                              #####
###############################################################################

cd "$WORKSPACE_DIR"
python3 weight_similarity_matrix.py --models \
  Base:meta-llama/Llama-3.1-8B \
  SFT:allenai/Llama-3.1-Tulu-3-8B-SFT \
  DPO:allenai/Llama-3.1-Tulu-3-8B-DPO \
  RLVR:allenai/Llama-3.1-Tulu-3-8B \
  Instruct:meta-llama/Llama-3.1-8B-Instruct \
  Tulu3.1:allenai/Llama-3.1-Tulu-3.1-8B \
  Llama3-Inst:meta-llama/Meta-Llama-3-8B-Instruct \
  R1-Distill:deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
  --output_dir "$RESULTS_DIR/0327/cross-delta-0327/weight_similarity"

