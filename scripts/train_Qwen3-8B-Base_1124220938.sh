##### Auto-generated 2025-11-24 22:09:38 #####
# Model     : Qwen3-8B-Base
# LoRA mode : true
# KD mode   : false
# Template  : qwen3

##### Environment #####
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export HF_HUB_OFFLINE=0
export HF_DATASETS_OFFLINE=0
export HF_DATASETS_TRUST_REMOTE_CODE=1
export TRUST_REMOTE_CODE=True
export HF_ALLOW_CODE_EVAL=1

##### Training #####
##### Standard Training #####
###### B  max=100000  lr=0.0002 ######
mkdir -p "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core"
cd "/dockerdata/LLM-NEO-bk"
llamafactory-cli train \
  --model_name_or_path "Qwen/Qwen3-8B-Base" \
  --stage sft \
  --do_train true \
  --finetuning_type lora --lora_rank 128 \
  --deepspeed examples/deepspeed/ds_z3_config.json \
  --dataset "BAAI-Infinity-7M-core" \
  --template "qwen3" \
  --cutoff_len 4096 \
  --max_samples 100000 \
  --output_dir "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core" \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 16 \
  --learning_rate 0.0002 \
  --num_train_epochs 1 \
  --logging_steps 1 \
  --save_steps 10 \
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

###### I  max=100000  lr=0.0002 ######
mkdir -p "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core"
cd "/dockerdata/LLM-NEO-bk"
llamafactory-cli train \
  --model_name_or_path "Qwen/Qwen3-8B" \
  --stage sft \
  --do_train true \
  --finetuning_type lora --lora_rank 128 \
  --deepspeed examples/deepspeed/ds_z3_config.json \
  --dataset "BAAI-Infinity-7M-core" \
  --template "qwen3" \
  --cutoff_len 4096 \
  --max_samples 100000 \
  --output_dir "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core" \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 16 \
  --learning_rate 0.0002 \
  --num_train_epochs 1 \
  --logging_steps 1 \
  --save_steps 10 \
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

##### LoRA delta-merge #####
### Standard LoRA Merges ###
if [ -d "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-0" ]; then
  mkdir -p "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-0/merged-B2I-s0"
  python3 /dockerdata/LLM-NEO-bk/src/shadow/merge_lora.py \
    --adapter_path "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-0" \
    --target_base "Qwen/Qwen3-8B" \
    --merge_tag "B2I-s0" \
    --template "qwen3"
fi

if [ -d "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-10" ]; then
  mkdir -p "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-10/merged-B2I-s10"
  python3 /dockerdata/LLM-NEO-bk/src/shadow/merge_lora.py \
    --adapter_path "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-10" \
    --target_base "Qwen/Qwen3-8B" \
    --merge_tag "B2I-s10" \
    --template "qwen3"
fi

if [ -d "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-20" ]; then
  mkdir -p "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-20/merged-B2I-s20"
  python3 /dockerdata/LLM-NEO-bk/src/shadow/merge_lora.py \
    --adapter_path "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-20" \
    --target_base "Qwen/Qwen3-8B" \
    --merge_tag "B2I-s20" \
    --template "qwen3"
fi

if [ -d "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-30" ]; then
  mkdir -p "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-30/merged-B2I-s30"
  python3 /dockerdata/LLM-NEO-bk/src/shadow/merge_lora.py \
    --adapter_path "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-30" \
    --target_base "Qwen/Qwen3-8B" \
    --merge_tag "B2I-s30" \
    --template "qwen3"
fi

if [ -d "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-40" ]; then
  mkdir -p "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-40/merged-B2I-s40"
  python3 /dockerdata/LLM-NEO-bk/src/shadow/merge_lora.py \
    --adapter_path "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-40" \
    --target_base "Qwen/Qwen3-8B" \
    --merge_tag "B2I-s40" \
    --template "qwen3"
fi

if [ -d "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-50" ]; then
  mkdir -p "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-50/merged-B2I-s50"
  python3 /dockerdata/LLM-NEO-bk/src/shadow/merge_lora.py \
    --adapter_path "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-50" \
    --target_base "Qwen/Qwen3-8B" \
    --merge_tag "B2I-s50" \
    --template "qwen3"
fi

if [ -d "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-60" ]; then
  mkdir -p "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-60/merged-B2I-s60"
  python3 /dockerdata/LLM-NEO-bk/src/shadow/merge_lora.py \
    --adapter_path "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-60" \
    --target_base "Qwen/Qwen3-8B" \
    --merge_tag "B2I-s60" \
    --template "qwen3"
fi

if [ -d "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-70" ]; then
  mkdir -p "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-70/merged-B2I-s70"
  python3 /dockerdata/LLM-NEO-bk/src/shadow/merge_lora.py \
    --adapter_path "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-70" \
    --target_base "Qwen/Qwen3-8B" \
    --merge_tag "B2I-s70" \
    --template "qwen3"
fi

if [ -d "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-80" ]; then
  mkdir -p "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-80/merged-B2I-s80"
  python3 /dockerdata/LLM-NEO-bk/src/shadow/merge_lora.py \
    --adapter_path "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-80" \
    --target_base "Qwen/Qwen3-8B" \
    --merge_tag "B2I-s80" \
    --template "qwen3"
fi

if [ -d "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-90" ]; then
  mkdir -p "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-90/merged-B2I-s90"
  python3 /dockerdata/LLM-NEO-bk/src/shadow/merge_lora.py \
    --adapter_path "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-90" \
    --target_base "Qwen/Qwen3-8B" \
    --merge_tag "B2I-s90" \
    --template "qwen3"
fi

if [ -d "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-100" ]; then
  mkdir -p "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-100/merged-B2I-s100"
  python3 /dockerdata/LLM-NEO-bk/src/shadow/merge_lora.py \
    --adapter_path "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-100" \
    --target_base "Qwen/Qwen3-8B" \
    --merge_tag "B2I-s100" \
    --template "qwen3"
fi

if [ -d "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-110" ]; then
  mkdir -p "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-110/merged-B2I-s110"
  python3 /dockerdata/LLM-NEO-bk/src/shadow/merge_lora.py \
    --adapter_path "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-110" \
    --target_base "Qwen/Qwen3-8B" \
    --merge_tag "B2I-s110" \
    --template "qwen3"
fi

if [ -d "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-120" ]; then
  mkdir -p "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-120/merged-B2I-s120"
  python3 /dockerdata/LLM-NEO-bk/src/shadow/merge_lora.py \
    --adapter_path "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-120" \
    --target_base "Qwen/Qwen3-8B" \
    --merge_tag "B2I-s120" \
    --template "qwen3"
fi

if [ -d "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-130" ]; then
  mkdir -p "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-130/merged-B2I-s130"
  python3 /dockerdata/LLM-NEO-bk/src/shadow/merge_lora.py \
    --adapter_path "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-130" \
    --target_base "Qwen/Qwen3-8B" \
    --merge_tag "B2I-s130" \
    --template "qwen3"
fi

if [ -d "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-140" ]; then
  mkdir -p "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-140/merged-B2I-s140"
  python3 /dockerdata/LLM-NEO-bk/src/shadow/merge_lora.py \
    --adapter_path "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-140" \
    --target_base "Qwen/Qwen3-8B" \
    --merge_tag "B2I-s140" \
    --template "qwen3"
fi

if [ -d "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-150" ]; then
  mkdir -p "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-150/merged-B2I-s150"
  python3 /dockerdata/LLM-NEO-bk/src/shadow/merge_lora.py \
    --adapter_path "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-150" \
    --target_base "Qwen/Qwen3-8B" \
    --merge_tag "B2I-s150" \
    --template "qwen3"
fi

if [ -d "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-160" ]; then
  mkdir -p "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-160/merged-B2I-s160"
  python3 /dockerdata/LLM-NEO-bk/src/shadow/merge_lora.py \
    --adapter_path "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-160" \
    --target_base "Qwen/Qwen3-8B" \
    --merge_tag "B2I-s160" \
    --template "qwen3"
fi

if [ -d "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-170" ]; then
  mkdir -p "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-170/merged-B2I-s170"
  python3 /dockerdata/LLM-NEO-bk/src/shadow/merge_lora.py \
    --adapter_path "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-170" \
    --target_base "Qwen/Qwen3-8B" \
    --merge_tag "B2I-s170" \
    --template "qwen3"
fi

if [ -d "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-180" ]; then
  mkdir -p "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-180/merged-B2I-s180"
  python3 /dockerdata/LLM-NEO-bk/src/shadow/merge_lora.py \
    --adapter_path "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-180" \
    --target_base "Qwen/Qwen3-8B" \
    --merge_tag "B2I-s180" \
    --template "qwen3"
fi

if [ -d "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-190" ]; then
  mkdir -p "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-190/merged-B2I-s190"
  python3 /dockerdata/LLM-NEO-bk/src/shadow/merge_lora.py \
    --adapter_path "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-190" \
    --target_base "Qwen/Qwen3-8B" \
    --merge_tag "B2I-s190" \
    --template "qwen3"
fi

if [ -d "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-200" ]; then
  mkdir -p "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-200/merged-B2I-s200"
  python3 /dockerdata/LLM-NEO-bk/src/shadow/merge_lora.py \
    --adapter_path "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-200" \
    --target_base "Qwen/Qwen3-8B" \
    --merge_tag "B2I-s200" \
    --template "qwen3"
fi

if [ -d "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-210" ]; then
  mkdir -p "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-210/merged-B2I-s210"
  python3 /dockerdata/LLM-NEO-bk/src/shadow/merge_lora.py \
    --adapter_path "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-210" \
    --target_base "Qwen/Qwen3-8B" \
    --merge_tag "B2I-s210" \
    --template "qwen3"
fi

if [ -d "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-220" ]; then
  mkdir -p "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-220/merged-B2I-s220"
  python3 /dockerdata/LLM-NEO-bk/src/shadow/merge_lora.py \
    --adapter_path "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-220" \
    --target_base "Qwen/Qwen3-8B" \
    --merge_tag "B2I-s220" \
    --template "qwen3"
fi

if [ -d "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-230" ]; then
  mkdir -p "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-230/merged-B2I-s230"
  python3 /dockerdata/LLM-NEO-bk/src/shadow/merge_lora.py \
    --adapter_path "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-230" \
    --target_base "Qwen/Qwen3-8B" \
    --merge_tag "B2I-s230" \
    --template "qwen3"
fi

if [ -d "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-240" ]; then
  mkdir -p "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-240/merged-B2I-s240"
  python3 /dockerdata/LLM-NEO-bk/src/shadow/merge_lora.py \
    --adapter_path "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-240" \
    --target_base "Qwen/Qwen3-8B" \
    --merge_tag "B2I-s240" \
    --template "qwen3"
fi

if [ -d "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-250" ]; then
  mkdir -p "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-250/merged-B2I-s250"
  python3 /dockerdata/LLM-NEO-bk/src/shadow/merge_lora.py \
    --adapter_path "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-250" \
    --target_base "Qwen/Qwen3-8B" \
    --merge_tag "B2I-s250" \
    --template "qwen3"
fi

if [ -d "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-260" ]; then
  mkdir -p "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-260/merged-B2I-s260"
  python3 /dockerdata/LLM-NEO-bk/src/shadow/merge_lora.py \
    --adapter_path "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-260" \
    --target_base "Qwen/Qwen3-8B" \
    --merge_tag "B2I-s260" \
    --template "qwen3"
fi

if [ -d "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-270" ]; then
  mkdir -p "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-270/merged-B2I-s270"
  python3 /dockerdata/LLM-NEO-bk/src/shadow/merge_lora.py \
    --adapter_path "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-270" \
    --target_base "Qwen/Qwen3-8B" \
    --merge_tag "B2I-s270" \
    --template "qwen3"
fi

if [ -d "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-280" ]; then
  mkdir -p "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-280/merged-B2I-s280"
  python3 /dockerdata/LLM-NEO-bk/src/shadow/merge_lora.py \
    --adapter_path "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-280" \
    --target_base "Qwen/Qwen3-8B" \
    --merge_tag "B2I-s280" \
    --template "qwen3"
fi

if [ -d "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-290" ]; then
  mkdir -p "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-290/merged-B2I-s290"
  python3 /dockerdata/LLM-NEO-bk/src/shadow/merge_lora.py \
    --adapter_path "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-290" \
    --target_base "Qwen/Qwen3-8B" \
    --merge_tag "B2I-s290" \
    --template "qwen3"
fi

if [ -d "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-300" ]; then
  mkdir -p "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-300/merged-B2I-s300"
  python3 /dockerdata/LLM-NEO-bk/src/shadow/merge_lora.py \
    --adapter_path "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-300" \
    --target_base "Qwen/Qwen3-8B" \
    --merge_tag "B2I-s300" \
    --template "qwen3"
fi

if [ -d "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-310" ]; then
  mkdir -p "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-310/merged-B2I-s310"
  python3 /dockerdata/LLM-NEO-bk/src/shadow/merge_lora.py \
    --adapter_path "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-310" \
    --target_base "Qwen/Qwen3-8B" \
    --merge_tag "B2I-s310" \
    --template "qwen3"
fi

if [ -d "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-320" ]; then
  mkdir -p "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-320/merged-B2I-s320"
  python3 /dockerdata/LLM-NEO-bk/src/shadow/merge_lora.py \
    --adapter_path "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-320" \
    --target_base "Qwen/Qwen3-8B" \
    --merge_tag "B2I-s320" \
    --template "qwen3"
fi

if [ -d "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-330" ]; then
  mkdir -p "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-330/merged-B2I-s330"
  python3 /dockerdata/LLM-NEO-bk/src/shadow/merge_lora.py \
    --adapter_path "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-330" \
    --target_base "Qwen/Qwen3-8B" \
    --merge_tag "B2I-s330" \
    --template "qwen3"
fi

if [ -d "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-340" ]; then
  mkdir -p "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-340/merged-B2I-s340"
  python3 /dockerdata/LLM-NEO-bk/src/shadow/merge_lora.py \
    --adapter_path "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-340" \
    --target_base "Qwen/Qwen3-8B" \
    --merge_tag "B2I-s340" \
    --template "qwen3"
fi

if [ -d "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-350" ]; then
  mkdir -p "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-350/merged-B2I-s350"
  python3 /dockerdata/LLM-NEO-bk/src/shadow/merge_lora.py \
    --adapter_path "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-350" \
    --target_base "Qwen/Qwen3-8B" \
    --merge_tag "B2I-s350" \
    --template "qwen3"
fi

if [ -d "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-360" ]; then
  mkdir -p "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-360/merged-B2I-s360"
  python3 /dockerdata/LLM-NEO-bk/src/shadow/merge_lora.py \
    --adapter_path "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-360" \
    --target_base "Qwen/Qwen3-8B" \
    --merge_tag "B2I-s360" \
    --template "qwen3"
fi

if [ -d "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-370" ]; then
  mkdir -p "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-370/merged-B2I-s370"
  python3 /dockerdata/LLM-NEO-bk/src/shadow/merge_lora.py \
    --adapter_path "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-370" \
    --target_base "Qwen/Qwen3-8B" \
    --merge_tag "B2I-s370" \
    --template "qwen3"
fi

if [ -d "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-380" ]; then
  mkdir -p "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-380/merged-B2I-s380"
  python3 /dockerdata/LLM-NEO-bk/src/shadow/merge_lora.py \
    --adapter_path "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-380" \
    --target_base "Qwen/Qwen3-8B" \
    --merge_tag "B2I-s380" \
    --template "qwen3"
fi

if [ -d "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-390" ]; then
  mkdir -p "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-390/merged-B2I-s390"
  python3 /dockerdata/LLM-NEO-bk/src/shadow/merge_lora.py \
    --adapter_path "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-390" \
    --target_base "Qwen/Qwen3-8B" \
    --merge_tag "B2I-s390" \
    --template "qwen3"
fi

if [ -d "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-400" ]; then
  mkdir -p "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-400/merged-B2I-s400"
  python3 /dockerdata/LLM-NEO-bk/src/shadow/merge_lora.py \
    --adapter_path "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-400" \
    --target_base "Qwen/Qwen3-8B" \
    --merge_tag "B2I-s400" \
    --template "qwen3"
fi

if [ -d "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-0" ]; then
  mkdir -p "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-0/merged-I2I-s0"
  python3 /dockerdata/LLM-NEO-bk/src/shadow/merge_lora.py \
    --adapter_path "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-0" \
    --target_base "Qwen/Qwen3-8B" \
    --merge_tag "I2I-s0" \
    --template "qwen3"
fi

if [ -d "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-10" ]; then
  mkdir -p "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-10/merged-I2I-s10"
  python3 /dockerdata/LLM-NEO-bk/src/shadow/merge_lora.py \
    --adapter_path "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-10" \
    --target_base "Qwen/Qwen3-8B" \
    --merge_tag "I2I-s10" \
    --template "qwen3"
fi

if [ -d "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-20" ]; then
  mkdir -p "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-20/merged-I2I-s20"
  python3 /dockerdata/LLM-NEO-bk/src/shadow/merge_lora.py \
    --adapter_path "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-20" \
    --target_base "Qwen/Qwen3-8B" \
    --merge_tag "I2I-s20" \
    --template "qwen3"
fi

if [ -d "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-30" ]; then
  mkdir -p "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-30/merged-I2I-s30"
  python3 /dockerdata/LLM-NEO-bk/src/shadow/merge_lora.py \
    --adapter_path "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-30" \
    --target_base "Qwen/Qwen3-8B" \
    --merge_tag "I2I-s30" \
    --template "qwen3"
fi

if [ -d "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-40" ]; then
  mkdir -p "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-40/merged-I2I-s40"
  python3 /dockerdata/LLM-NEO-bk/src/shadow/merge_lora.py \
    --adapter_path "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-40" \
    --target_base "Qwen/Qwen3-8B" \
    --merge_tag "I2I-s40" \
    --template "qwen3"
fi

if [ -d "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-50" ]; then
  mkdir -p "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-50/merged-I2I-s50"
  python3 /dockerdata/LLM-NEO-bk/src/shadow/merge_lora.py \
    --adapter_path "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-50" \
    --target_base "Qwen/Qwen3-8B" \
    --merge_tag "I2I-s50" \
    --template "qwen3"
fi

if [ -d "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-60" ]; then
  mkdir -p "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-60/merged-I2I-s60"
  python3 /dockerdata/LLM-NEO-bk/src/shadow/merge_lora.py \
    --adapter_path "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-60" \
    --target_base "Qwen/Qwen3-8B" \
    --merge_tag "I2I-s60" \
    --template "qwen3"
fi

if [ -d "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-70" ]; then
  mkdir -p "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-70/merged-I2I-s70"
  python3 /dockerdata/LLM-NEO-bk/src/shadow/merge_lora.py \
    --adapter_path "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-70" \
    --target_base "Qwen/Qwen3-8B" \
    --merge_tag "I2I-s70" \
    --template "qwen3"
fi

if [ -d "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-80" ]; then
  mkdir -p "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-80/merged-I2I-s80"
  python3 /dockerdata/LLM-NEO-bk/src/shadow/merge_lora.py \
    --adapter_path "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-80" \
    --target_base "Qwen/Qwen3-8B" \
    --merge_tag "I2I-s80" \
    --template "qwen3"
fi

if [ -d "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-90" ]; then
  mkdir -p "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-90/merged-I2I-s90"
  python3 /dockerdata/LLM-NEO-bk/src/shadow/merge_lora.py \
    --adapter_path "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-90" \
    --target_base "Qwen/Qwen3-8B" \
    --merge_tag "I2I-s90" \
    --template "qwen3"
fi

if [ -d "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-100" ]; then
  mkdir -p "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-100/merged-I2I-s100"
  python3 /dockerdata/LLM-NEO-bk/src/shadow/merge_lora.py \
    --adapter_path "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-100" \
    --target_base "Qwen/Qwen3-8B" \
    --merge_tag "I2I-s100" \
    --template "qwen3"
fi

if [ -d "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-110" ]; then
  mkdir -p "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-110/merged-I2I-s110"
  python3 /dockerdata/LLM-NEO-bk/src/shadow/merge_lora.py \
    --adapter_path "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-110" \
    --target_base "Qwen/Qwen3-8B" \
    --merge_tag "I2I-s110" \
    --template "qwen3"
fi

if [ -d "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-120" ]; then
  mkdir -p "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-120/merged-I2I-s120"
  python3 /dockerdata/LLM-NEO-bk/src/shadow/merge_lora.py \
    --adapter_path "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-120" \
    --target_base "Qwen/Qwen3-8B" \
    --merge_tag "I2I-s120" \
    --template "qwen3"
fi

if [ -d "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-130" ]; then
  mkdir -p "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-130/merged-I2I-s130"
  python3 /dockerdata/LLM-NEO-bk/src/shadow/merge_lora.py \
    --adapter_path "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-130" \
    --target_base "Qwen/Qwen3-8B" \
    --merge_tag "I2I-s130" \
    --template "qwen3"
fi

if [ -d "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-140" ]; then
  mkdir -p "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-140/merged-I2I-s140"
  python3 /dockerdata/LLM-NEO-bk/src/shadow/merge_lora.py \
    --adapter_path "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-140" \
    --target_base "Qwen/Qwen3-8B" \
    --merge_tag "I2I-s140" \
    --template "qwen3"
fi

if [ -d "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-150" ]; then
  mkdir -p "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-150/merged-I2I-s150"
  python3 /dockerdata/LLM-NEO-bk/src/shadow/merge_lora.py \
    --adapter_path "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-150" \
    --target_base "Qwen/Qwen3-8B" \
    --merge_tag "I2I-s150" \
    --template "qwen3"
fi

if [ -d "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-160" ]; then
  mkdir -p "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-160/merged-I2I-s160"
  python3 /dockerdata/LLM-NEO-bk/src/shadow/merge_lora.py \
    --adapter_path "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-160" \
    --target_base "Qwen/Qwen3-8B" \
    --merge_tag "I2I-s160" \
    --template "qwen3"
fi

if [ -d "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-170" ]; then
  mkdir -p "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-170/merged-I2I-s170"
  python3 /dockerdata/LLM-NEO-bk/src/shadow/merge_lora.py \
    --adapter_path "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-170" \
    --target_base "Qwen/Qwen3-8B" \
    --merge_tag "I2I-s170" \
    --template "qwen3"
fi

if [ -d "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-180" ]; then
  mkdir -p "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-180/merged-I2I-s180"
  python3 /dockerdata/LLM-NEO-bk/src/shadow/merge_lora.py \
    --adapter_path "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-180" \
    --target_base "Qwen/Qwen3-8B" \
    --merge_tag "I2I-s180" \
    --template "qwen3"
fi

if [ -d "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-190" ]; then
  mkdir -p "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-190/merged-I2I-s190"
  python3 /dockerdata/LLM-NEO-bk/src/shadow/merge_lora.py \
    --adapter_path "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-190" \
    --target_base "Qwen/Qwen3-8B" \
    --merge_tag "I2I-s190" \
    --template "qwen3"
fi

if [ -d "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-200" ]; then
  mkdir -p "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-200/merged-I2I-s200"
  python3 /dockerdata/LLM-NEO-bk/src/shadow/merge_lora.py \
    --adapter_path "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-200" \
    --target_base "Qwen/Qwen3-8B" \
    --merge_tag "I2I-s200" \
    --template "qwen3"
fi

if [ -d "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-210" ]; then
  mkdir -p "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-210/merged-I2I-s210"
  python3 /dockerdata/LLM-NEO-bk/src/shadow/merge_lora.py \
    --adapter_path "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-210" \
    --target_base "Qwen/Qwen3-8B" \
    --merge_tag "I2I-s210" \
    --template "qwen3"
fi

if [ -d "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-220" ]; then
  mkdir -p "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-220/merged-I2I-s220"
  python3 /dockerdata/LLM-NEO-bk/src/shadow/merge_lora.py \
    --adapter_path "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-220" \
    --target_base "Qwen/Qwen3-8B" \
    --merge_tag "I2I-s220" \
    --template "qwen3"
fi

if [ -d "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-230" ]; then
  mkdir -p "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-230/merged-I2I-s230"
  python3 /dockerdata/LLM-NEO-bk/src/shadow/merge_lora.py \
    --adapter_path "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-230" \
    --target_base "Qwen/Qwen3-8B" \
    --merge_tag "I2I-s230" \
    --template "qwen3"
fi

if [ -d "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-240" ]; then
  mkdir -p "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-240/merged-I2I-s240"
  python3 /dockerdata/LLM-NEO-bk/src/shadow/merge_lora.py \
    --adapter_path "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-240" \
    --target_base "Qwen/Qwen3-8B" \
    --merge_tag "I2I-s240" \
    --template "qwen3"
fi

if [ -d "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-250" ]; then
  mkdir -p "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-250/merged-I2I-s250"
  python3 /dockerdata/LLM-NEO-bk/src/shadow/merge_lora.py \
    --adapter_path "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-250" \
    --target_base "Qwen/Qwen3-8B" \
    --merge_tag "I2I-s250" \
    --template "qwen3"
fi

if [ -d "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-260" ]; then
  mkdir -p "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-260/merged-I2I-s260"
  python3 /dockerdata/LLM-NEO-bk/src/shadow/merge_lora.py \
    --adapter_path "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-260" \
    --target_base "Qwen/Qwen3-8B" \
    --merge_tag "I2I-s260" \
    --template "qwen3"
fi

if [ -d "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-270" ]; then
  mkdir -p "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-270/merged-I2I-s270"
  python3 /dockerdata/LLM-NEO-bk/src/shadow/merge_lora.py \
    --adapter_path "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-270" \
    --target_base "Qwen/Qwen3-8B" \
    --merge_tag "I2I-s270" \
    --template "qwen3"
fi

if [ -d "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-280" ]; then
  mkdir -p "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-280/merged-I2I-s280"
  python3 /dockerdata/LLM-NEO-bk/src/shadow/merge_lora.py \
    --adapter_path "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-280" \
    --target_base "Qwen/Qwen3-8B" \
    --merge_tag "I2I-s280" \
    --template "qwen3"
fi

if [ -d "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-290" ]; then
  mkdir -p "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-290/merged-I2I-s290"
  python3 /dockerdata/LLM-NEO-bk/src/shadow/merge_lora.py \
    --adapter_path "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-290" \
    --target_base "Qwen/Qwen3-8B" \
    --merge_tag "I2I-s290" \
    --template "qwen3"
fi

if [ -d "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-300" ]; then
  mkdir -p "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-300/merged-I2I-s300"
  python3 /dockerdata/LLM-NEO-bk/src/shadow/merge_lora.py \
    --adapter_path "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-300" \
    --target_base "Qwen/Qwen3-8B" \
    --merge_tag "I2I-s300" \
    --template "qwen3"
fi

if [ -d "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-310" ]; then
  mkdir -p "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-310/merged-I2I-s310"
  python3 /dockerdata/LLM-NEO-bk/src/shadow/merge_lora.py \
    --adapter_path "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-310" \
    --target_base "Qwen/Qwen3-8B" \
    --merge_tag "I2I-s310" \
    --template "qwen3"
fi

if [ -d "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-320" ]; then
  mkdir -p "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-320/merged-I2I-s320"
  python3 /dockerdata/LLM-NEO-bk/src/shadow/merge_lora.py \
    --adapter_path "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-320" \
    --target_base "Qwen/Qwen3-8B" \
    --merge_tag "I2I-s320" \
    --template "qwen3"
fi

if [ -d "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-330" ]; then
  mkdir -p "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-330/merged-I2I-s330"
  python3 /dockerdata/LLM-NEO-bk/src/shadow/merge_lora.py \
    --adapter_path "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-330" \
    --target_base "Qwen/Qwen3-8B" \
    --merge_tag "I2I-s330" \
    --template "qwen3"
fi

if [ -d "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-340" ]; then
  mkdir -p "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-340/merged-I2I-s340"
  python3 /dockerdata/LLM-NEO-bk/src/shadow/merge_lora.py \
    --adapter_path "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-340" \
    --target_base "Qwen/Qwen3-8B" \
    --merge_tag "I2I-s340" \
    --template "qwen3"
fi

if [ -d "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-350" ]; then
  mkdir -p "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-350/merged-I2I-s350"
  python3 /dockerdata/LLM-NEO-bk/src/shadow/merge_lora.py \
    --adapter_path "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-350" \
    --target_base "Qwen/Qwen3-8B" \
    --merge_tag "I2I-s350" \
    --template "qwen3"
fi

if [ -d "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-360" ]; then
  mkdir -p "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-360/merged-I2I-s360"
  python3 /dockerdata/LLM-NEO-bk/src/shadow/merge_lora.py \
    --adapter_path "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-360" \
    --target_base "Qwen/Qwen3-8B" \
    --merge_tag "I2I-s360" \
    --template "qwen3"
fi

if [ -d "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-370" ]; then
  mkdir -p "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-370/merged-I2I-s370"
  python3 /dockerdata/LLM-NEO-bk/src/shadow/merge_lora.py \
    --adapter_path "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-370" \
    --target_base "Qwen/Qwen3-8B" \
    --merge_tag "I2I-s370" \
    --template "qwen3"
fi

if [ -d "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-380" ]; then
  mkdir -p "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-380/merged-I2I-s380"
  python3 /dockerdata/LLM-NEO-bk/src/shadow/merge_lora.py \
    --adapter_path "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-380" \
    --target_base "Qwen/Qwen3-8B" \
    --merge_tag "I2I-s380" \
    --template "qwen3"
fi

if [ -d "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-390" ]; then
  mkdir -p "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-390/merged-I2I-s390"
  python3 /dockerdata/LLM-NEO-bk/src/shadow/merge_lora.py \
    --adapter_path "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-390" \
    --target_base "Qwen/Qwen3-8B" \
    --merge_tag "I2I-s390" \
    --template "qwen3"
fi

if [ -d "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-400" ]; then
  mkdir -p "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-400/merged-I2I-s400"
  python3 /dockerdata/LLM-NEO-bk/src/shadow/merge_lora.py \
    --adapter_path "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-400" \
    --target_base "Qwen/Qwen3-8B" \
    --merge_tag "I2I-s400" \
    --template "qwen3"
fi

if [ -d "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-0" ]; then
  mkdir -p "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-0/merged-I2B-s0"
  #python3 /dockerdata/LLM-NEO-bk/src/shadow/merge_lora.py \
  #  --adapter_path "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-0" \
  #  --target_base "Qwen/Qwen3-8B-Base" \
  #  --merge_tag "I2B-s0" \
  #  --template "qwen3"
fi

if [ -d "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-10" ]; then
  mkdir -p "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-10/merged-I2B-s10"
  #python3 /dockerdata/LLM-NEO-bk/src/shadow/merge_lora.py \
  #  --adapter_path "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-10" \
  #  --target_base "Qwen/Qwen3-8B-Base" \
  #  --merge_tag "I2B-s10" \
  #  --template "qwen3"
fi

if [ -d "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-20" ]; then
  mkdir -p "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-20/merged-I2B-s20"
  #python3 /dockerdata/LLM-NEO-bk/src/shadow/merge_lora.py \
  #  --adapter_path "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-20" \
  #  --target_base "Qwen/Qwen3-8B-Base" \
  #  --merge_tag "I2B-s20" \
  #  --template "qwen3"
fi

if [ -d "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-30" ]; then
  mkdir -p "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-30/merged-I2B-s30"
  #python3 /dockerdata/LLM-NEO-bk/src/shadow/merge_lora.py \
  #  --adapter_path "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-30" \
  #  --target_base "Qwen/Qwen3-8B-Base" \
  #  --merge_tag "I2B-s30" \
  #  --template "qwen3"
fi

if [ -d "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-40" ]; then
  mkdir -p "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-40/merged-I2B-s40"
  #python3 /dockerdata/LLM-NEO-bk/src/shadow/merge_lora.py \
  #  --adapter_path "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-40" \
  #  --target_base "Qwen/Qwen3-8B-Base" \
  #  --merge_tag "I2B-s40" \
  #  --template "qwen3"
fi

if [ -d "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-50" ]; then
  mkdir -p "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-50/merged-I2B-s50"
  #python3 /dockerdata/LLM-NEO-bk/src/shadow/merge_lora.py \
  #  --adapter_path "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-50" \
  #  --target_base "Qwen/Qwen3-8B-Base" \
  #  --merge_tag "I2B-s50" \
  #  --template "qwen3"
fi

if [ -d "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-60" ]; then
  mkdir -p "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-60/merged-I2B-s60"
  #python3 /dockerdata/LLM-NEO-bk/src/shadow/merge_lora.py \
  #  --adapter_path "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-60" \
  #  --target_base "Qwen/Qwen3-8B-Base" \
  #  --merge_tag "I2B-s60" \
  #  --template "qwen3"
fi

if [ -d "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-70" ]; then
  mkdir -p "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-70/merged-I2B-s70"
  #python3 /dockerdata/LLM-NEO-bk/src/shadow/merge_lora.py \
  #  --adapter_path "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-70" \
  #  --target_base "Qwen/Qwen3-8B-Base" \
  #  --merge_tag "I2B-s70" \
  #  --template "qwen3"
fi

if [ -d "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-80" ]; then
  mkdir -p "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-80/merged-I2B-s80"
  #python3 /dockerdata/LLM-NEO-bk/src/shadow/merge_lora.py \
  #  --adapter_path "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-80" \
  #  --target_base "Qwen/Qwen3-8B-Base" \
  #  --merge_tag "I2B-s80" \
  #  --template "qwen3"
fi

if [ -d "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-90" ]; then
  mkdir -p "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-90/merged-I2B-s90"
  #python3 /dockerdata/LLM-NEO-bk/src/shadow/merge_lora.py \
  #  --adapter_path "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-90" \
  #  --target_base "Qwen/Qwen3-8B-Base" \
  #  --merge_tag "I2B-s90" \
  #  --template "qwen3"
fi

if [ -d "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-100" ]; then
  mkdir -p "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-100/merged-I2B-s100"
  #python3 /dockerdata/LLM-NEO-bk/src/shadow/merge_lora.py \
  #  --adapter_path "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-100" \
  #  --target_base "Qwen/Qwen3-8B-Base" \
  #  --merge_tag "I2B-s100" \
  #  --template "qwen3"
fi

if [ -d "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-110" ]; then
  mkdir -p "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-110/merged-I2B-s110"
  #python3 /dockerdata/LLM-NEO-bk/src/shadow/merge_lora.py \
  #  --adapter_path "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-110" \
  #  --target_base "Qwen/Qwen3-8B-Base" \
  #  --merge_tag "I2B-s110" \
  #  --template "qwen3"
fi

if [ -d "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-120" ]; then
  mkdir -p "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-120/merged-I2B-s120"
  #python3 /dockerdata/LLM-NEO-bk/src/shadow/merge_lora.py \
  #  --adapter_path "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-120" \
  #  --target_base "Qwen/Qwen3-8B-Base" \
  #  --merge_tag "I2B-s120" \
  #  --template "qwen3"
fi

if [ -d "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-130" ]; then
  mkdir -p "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-130/merged-I2B-s130"
  #python3 /dockerdata/LLM-NEO-bk/src/shadow/merge_lora.py \
  #  --adapter_path "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-130" \
  #  --target_base "Qwen/Qwen3-8B-Base" \
  #  --merge_tag "I2B-s130" \
  #  --template "qwen3"
fi

if [ -d "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-140" ]; then
  mkdir -p "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-140/merged-I2B-s140"
  #python3 /dockerdata/LLM-NEO-bk/src/shadow/merge_lora.py \
  #  --adapter_path "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-140" \
  #  --target_base "Qwen/Qwen3-8B-Base" \
  #  --merge_tag "I2B-s140" \
  #  --template "qwen3"
fi

if [ -d "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-150" ]; then
  mkdir -p "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-150/merged-I2B-s150"
  #python3 /dockerdata/LLM-NEO-bk/src/shadow/merge_lora.py \
  #  --adapter_path "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-150" \
  #  --target_base "Qwen/Qwen3-8B-Base" \
  #  --merge_tag "I2B-s150" \
  #  --template "qwen3"
fi

if [ -d "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-160" ]; then
  mkdir -p "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-160/merged-I2B-s160"
  #python3 /dockerdata/LLM-NEO-bk/src/shadow/merge_lora.py \
  #  --adapter_path "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-160" \
  #  --target_base "Qwen/Qwen3-8B-Base" \
  #  --merge_tag "I2B-s160" \
  #  --template "qwen3"
fi

if [ -d "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-170" ]; then
  mkdir -p "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-170/merged-I2B-s170"
  #python3 /dockerdata/LLM-NEO-bk/src/shadow/merge_lora.py \
  #  --adapter_path "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-170" \
  #  --target_base "Qwen/Qwen3-8B-Base" \
  #  --merge_tag "I2B-s170" \
  #  --template "qwen3"
fi

if [ -d "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-180" ]; then
  mkdir -p "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-180/merged-I2B-s180"
  #python3 /dockerdata/LLM-NEO-bk/src/shadow/merge_lora.py \
  #  --adapter_path "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-180" \
  #  --target_base "Qwen/Qwen3-8B-Base" \
  #  --merge_tag "I2B-s180" \
  #  --template "qwen3"
fi

if [ -d "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-190" ]; then
  mkdir -p "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-190/merged-I2B-s190"
  #python3 /dockerdata/LLM-NEO-bk/src/shadow/merge_lora.py \
  #  --adapter_path "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-190" \
  #  --target_base "Qwen/Qwen3-8B-Base" \
  #  --merge_tag "I2B-s190" \
  #  --template "qwen3"
fi

if [ -d "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-200" ]; then
  mkdir -p "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-200/merged-I2B-s200"
  #python3 /dockerdata/LLM-NEO-bk/src/shadow/merge_lora.py \
  #  --adapter_path "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-200" \
  #  --target_base "Qwen/Qwen3-8B-Base" \
  #  --merge_tag "I2B-s200" \
  #  --template "qwen3"
fi

if [ -d "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-210" ]; then
  mkdir -p "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-210/merged-I2B-s210"
  #python3 /dockerdata/LLM-NEO-bk/src/shadow/merge_lora.py \
  #  --adapter_path "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-210" \
  #  --target_base "Qwen/Qwen3-8B-Base" \
  #  --merge_tag "I2B-s210" \
  #  --template "qwen3"
fi

if [ -d "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-220" ]; then
  mkdir -p "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-220/merged-I2B-s220"
  #python3 /dockerdata/LLM-NEO-bk/src/shadow/merge_lora.py \
  #  --adapter_path "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-220" \
  #  --target_base "Qwen/Qwen3-8B-Base" \
  #  --merge_tag "I2B-s220" \
  #  --template "qwen3"
fi

if [ -d "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-230" ]; then
  mkdir -p "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-230/merged-I2B-s230"
  #python3 /dockerdata/LLM-NEO-bk/src/shadow/merge_lora.py \
  #  --adapter_path "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-230" \
  #  --target_base "Qwen/Qwen3-8B-Base" \
  #  --merge_tag "I2B-s230" \
  #  --template "qwen3"
fi

if [ -d "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-240" ]; then
  mkdir -p "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-240/merged-I2B-s240"
  #python3 /dockerdata/LLM-NEO-bk/src/shadow/merge_lora.py \
  #  --adapter_path "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-240" \
  #  --target_base "Qwen/Qwen3-8B-Base" \
  #  --merge_tag "I2B-s240" \
  #  --template "qwen3"
fi

if [ -d "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-250" ]; then
  mkdir -p "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-250/merged-I2B-s250"
  #python3 /dockerdata/LLM-NEO-bk/src/shadow/merge_lora.py \
  #  --adapter_path "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-250" \
  #  --target_base "Qwen/Qwen3-8B-Base" \
  #  --merge_tag "I2B-s250" \
  #  --template "qwen3"
fi

if [ -d "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-260" ]; then
  mkdir -p "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-260/merged-I2B-s260"
  #python3 /dockerdata/LLM-NEO-bk/src/shadow/merge_lora.py \
  #  --adapter_path "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-260" \
  #  --target_base "Qwen/Qwen3-8B-Base" \
  #  --merge_tag "I2B-s260" \
  #  --template "qwen3"
fi

if [ -d "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-270" ]; then
  mkdir -p "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-270/merged-I2B-s270"
  #python3 /dockerdata/LLM-NEO-bk/src/shadow/merge_lora.py \
  #  --adapter_path "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-270" \
  #  --target_base "Qwen/Qwen3-8B-Base" \
  #  --merge_tag "I2B-s270" \
  #  --template "qwen3"
fi

if [ -d "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-280" ]; then
  mkdir -p "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-280/merged-I2B-s280"
  #python3 /dockerdata/LLM-NEO-bk/src/shadow/merge_lora.py \
  #  --adapter_path "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-280" \
  #  --target_base "Qwen/Qwen3-8B-Base" \
  #  --merge_tag "I2B-s280" \
  #  --template "qwen3"
fi

if [ -d "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-290" ]; then
  mkdir -p "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-290/merged-I2B-s290"
  #python3 /dockerdata/LLM-NEO-bk/src/shadow/merge_lora.py \
  #  --adapter_path "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-290" \
  #  --target_base "Qwen/Qwen3-8B-Base" \
  #  --merge_tag "I2B-s290" \
  #  --template "qwen3"
fi

if [ -d "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-300" ]; then
  mkdir -p "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-300/merged-I2B-s300"
  #python3 /dockerdata/LLM-NEO-bk/src/shadow/merge_lora.py \
  #  --adapter_path "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-300" \
  #  --target_base "Qwen/Qwen3-8B-Base" \
  #  --merge_tag "I2B-s300" \
  #  --template "qwen3"
fi

if [ -d "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-310" ]; then
  mkdir -p "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-310/merged-I2B-s310"
  #python3 /dockerdata/LLM-NEO-bk/src/shadow/merge_lora.py \
  #  --adapter_path "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-310" \
  #  --target_base "Qwen/Qwen3-8B-Base" \
  #  --merge_tag "I2B-s310" \
  #  --template "qwen3"
fi

if [ -d "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-320" ]; then
  mkdir -p "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-320/merged-I2B-s320"
  #python3 /dockerdata/LLM-NEO-bk/src/shadow/merge_lora.py \
  #  --adapter_path "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-320" \
  #  --target_base "Qwen/Qwen3-8B-Base" \
  #  --merge_tag "I2B-s320" \
  #  --template "qwen3"
fi

if [ -d "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-330" ]; then
  mkdir -p "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-330/merged-I2B-s330"
  #python3 /dockerdata/LLM-NEO-bk/src/shadow/merge_lora.py \
  #  --adapter_path "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-330" \
  #  --target_base "Qwen/Qwen3-8B-Base" \
  #  --merge_tag "I2B-s330" \
  #  --template "qwen3"
fi

if [ -d "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-340" ]; then
  mkdir -p "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-340/merged-I2B-s340"
  #python3 /dockerdata/LLM-NEO-bk/src/shadow/merge_lora.py \
  #  --adapter_path "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-340" \
  #  --target_base "Qwen/Qwen3-8B-Base" \
  #  --merge_tag "I2B-s340" \
  #  --template "qwen3"
fi

if [ -d "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-350" ]; then
  mkdir -p "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-350/merged-I2B-s350"
  #python3 /dockerdata/LLM-NEO-bk/src/shadow/merge_lora.py \
  #  --adapter_path "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-350" \
  #  --target_base "Qwen/Qwen3-8B-Base" \
  #  --merge_tag "I2B-s350" \
  #  --template "qwen3"
fi

if [ -d "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-360" ]; then
  mkdir -p "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-360/merged-I2B-s360"
  #python3 /dockerdata/LLM-NEO-bk/src/shadow/merge_lora.py \
  #  --adapter_path "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-360" \
  #  --target_base "Qwen/Qwen3-8B-Base" \
  #  --merge_tag "I2B-s360" \
  #  --template "qwen3"
fi

if [ -d "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-370" ]; then
  mkdir -p "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-370/merged-I2B-s370"
  #python3 /dockerdata/LLM-NEO-bk/src/shadow/merge_lora.py \
  #  --adapter_path "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-370" \
  #  --target_base "Qwen/Qwen3-8B-Base" \
  #  --merge_tag "I2B-s370" \
  #  --template "qwen3"
fi

if [ -d "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-380" ]; then
  mkdir -p "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-380/merged-I2B-s380"
  #python3 /dockerdata/LLM-NEO-bk/src/shadow/merge_lora.py \
  #  --adapter_path "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-380" \
  #  --target_base "Qwen/Qwen3-8B-Base" \
  #  --merge_tag "I2B-s380" \
  #  --template "qwen3"
fi

if [ -d "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-390" ]; then
  mkdir -p "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-390/merged-I2B-s390"
  #python3 /dockerdata/LLM-NEO-bk/src/shadow/merge_lora.py \
  #  --adapter_path "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-390" \
  #  --target_base "Qwen/Qwen3-8B-Base" \
  #  --merge_tag "I2B-s390" \
  #  --template "qwen3"
fi

if [ -d "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-400" ]; then
  mkdir -p "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-400/merged-I2B-s400"
  #python3 /dockerdata/LLM-NEO-bk/src/shadow/merge_lora.py \
  #  --adapter_path "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-400" \
  #  --target_base "Qwen/Qwen3-8B-Base" \
  #  --merge_tag "I2B-s400" \
  #  --template "qwen3"
fi

if [ -d "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-0" ]; then
  mkdir -p "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-0/merged-B2B-s0"
  #python3 /dockerdata/LLM-NEO-bk/src/shadow/merge_lora.py \
  #  --adapter_path "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-0" \
  #  --target_base "Qwen/Qwen3-8B-Base" \
  #  --merge_tag "B2B-s0" \
  #  --template "qwen3"
fi

if [ -d "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-10" ]; then
  mkdir -p "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-10/merged-B2B-s10"
  #python3 /dockerdata/LLM-NEO-bk/src/shadow/merge_lora.py \
  #  --adapter_path "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-10" \
  #  --target_base "Qwen/Qwen3-8B-Base" \
  #  --merge_tag "B2B-s10" \
  #  --template "qwen3"
fi

if [ -d "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-20" ]; then
  mkdir -p "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-20/merged-B2B-s20"
  #python3 /dockerdata/LLM-NEO-bk/src/shadow/merge_lora.py \
  #  --adapter_path "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-20" \
  #  --target_base "Qwen/Qwen3-8B-Base" \
  #  --merge_tag "B2B-s20" \
  #  --template "qwen3"
fi

if [ -d "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-30" ]; then
  mkdir -p "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-30/merged-B2B-s30"
  #python3 /dockerdata/LLM-NEO-bk/src/shadow/merge_lora.py \
  #  --adapter_path "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-30" \
  #  --target_base "Qwen/Qwen3-8B-Base" \
  #  --merge_tag "B2B-s30" \
  #  --template "qwen3"
fi

if [ -d "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-40" ]; then
  mkdir -p "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-40/merged-B2B-s40"
  #python3 /dockerdata/LLM-NEO-bk/src/shadow/merge_lora.py \
  #  --adapter_path "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-40" \
  #  --target_base "Qwen/Qwen3-8B-Base" \
  #  --merge_tag "B2B-s40" \
  #  --template "qwen3"
fi

if [ -d "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-50" ]; then
  mkdir -p "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-50/merged-B2B-s50"
  #python3 /dockerdata/LLM-NEO-bk/src/shadow/merge_lora.py \
  #  --adapter_path "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-50" \
  #  --target_base "Qwen/Qwen3-8B-Base" \
  #  --merge_tag "B2B-s50" \
  #  --template "qwen3"
fi

if [ -d "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-60" ]; then
  mkdir -p "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-60/merged-B2B-s60"
  #python3 /dockerdata/LLM-NEO-bk/src/shadow/merge_lora.py \
  #  --adapter_path "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-60" \
  #  --target_base "Qwen/Qwen3-8B-Base" \
  #  --merge_tag "B2B-s60" \
  #  --template "qwen3"
fi

if [ -d "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-70" ]; then
  mkdir -p "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-70/merged-B2B-s70"
  #python3 /dockerdata/LLM-NEO-bk/src/shadow/merge_lora.py \
  #  --adapter_path "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-70" \
  #  --target_base "Qwen/Qwen3-8B-Base" \
  #  --merge_tag "B2B-s70" \
  #  --template "qwen3"
fi

if [ -d "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-80" ]; then
  mkdir -p "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-80/merged-B2B-s80"
  #python3 /dockerdata/LLM-NEO-bk/src/shadow/merge_lora.py \
  #  --adapter_path "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-80" \
  #  --target_base "Qwen/Qwen3-8B-Base" \
  #  --merge_tag "B2B-s80" \
  #  --template "qwen3"
fi

if [ -d "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-90" ]; then
  mkdir -p "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-90/merged-B2B-s90"
  #python3 /dockerdata/LLM-NEO-bk/src/shadow/merge_lora.py \
  #  --adapter_path "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-90" \
  #  --target_base "Qwen/Qwen3-8B-Base" \
  #  --merge_tag "B2B-s90" \
  #  --template "qwen3"
fi

if [ -d "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-100" ]; then
  mkdir -p "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-100/merged-B2B-s100"
  #python3 /dockerdata/LLM-NEO-bk/src/shadow/merge_lora.py \
  #  --adapter_path "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-100" \
  #  --target_base "Qwen/Qwen3-8B-Base" \
  #  --merge_tag "B2B-s100" \
  #  --template "qwen3"
fi

if [ -d "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-110" ]; then
  mkdir -p "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-110/merged-B2B-s110"
  #python3 /dockerdata/LLM-NEO-bk/src/shadow/merge_lora.py \
  #  --adapter_path "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-110" \
  #  --target_base "Qwen/Qwen3-8B-Base" \
  #  --merge_tag "B2B-s110" \
  #  --template "qwen3"
fi

if [ -d "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-120" ]; then
  mkdir -p "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-120/merged-B2B-s120"
  #python3 /dockerdata/LLM-NEO-bk/src/shadow/merge_lora.py \
  #  --adapter_path "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-120" \
  #  --target_base "Qwen/Qwen3-8B-Base" \
  #  --merge_tag "B2B-s120" \
  #  --template "qwen3"
fi

if [ -d "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-130" ]; then
  mkdir -p "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-130/merged-B2B-s130"
  #python3 /dockerdata/LLM-NEO-bk/src/shadow/merge_lora.py \
  #  --adapter_path "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-130" \
  #  --target_base "Qwen/Qwen3-8B-Base" \
  #  --merge_tag "B2B-s130" \
  #  --template "qwen3"
fi

if [ -d "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-140" ]; then
  mkdir -p "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-140/merged-B2B-s140"
  #python3 /dockerdata/LLM-NEO-bk/src/shadow/merge_lora.py \
  #  --adapter_path "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-140" \
  #  --target_base "Qwen/Qwen3-8B-Base" \
  #  --merge_tag "B2B-s140" \
  #  --template "qwen3"
fi

if [ -d "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-150" ]; then
  mkdir -p "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-150/merged-B2B-s150"
  #python3 /dockerdata/LLM-NEO-bk/src/shadow/merge_lora.py \
  #  --adapter_path "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-150" \
  #  --target_base "Qwen/Qwen3-8B-Base" \
  #  --merge_tag "B2B-s150" \
  #  --template "qwen3"
fi

if [ -d "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-160" ]; then
  mkdir -p "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-160/merged-B2B-s160"
  #python3 /dockerdata/LLM-NEO-bk/src/shadow/merge_lora.py \
  #  --adapter_path "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-160" \
  #  --target_base "Qwen/Qwen3-8B-Base" \
  #  --merge_tag "B2B-s160" \
  #  --template "qwen3"
fi

if [ -d "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-170" ]; then
  mkdir -p "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-170/merged-B2B-s170"
  #python3 /dockerdata/LLM-NEO-bk/src/shadow/merge_lora.py \
  #  --adapter_path "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-170" \
  #  --target_base "Qwen/Qwen3-8B-Base" \
  #  --merge_tag "B2B-s170" \
  #  --template "qwen3"
fi

if [ -d "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-180" ]; then
  mkdir -p "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-180/merged-B2B-s180"
  #python3 /dockerdata/LLM-NEO-bk/src/shadow/merge_lora.py \
  #  --adapter_path "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-180" \
  #  --target_base "Qwen/Qwen3-8B-Base" \
  #  --merge_tag "B2B-s180" \
  #  --template "qwen3"
fi

if [ -d "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-190" ]; then
  mkdir -p "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-190/merged-B2B-s190"
  #python3 /dockerdata/LLM-NEO-bk/src/shadow/merge_lora.py \
  #  --adapter_path "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-190" \
  #  --target_base "Qwen/Qwen3-8B-Base" \
  #  --merge_tag "B2B-s190" \
  #  --template "qwen3"
fi

if [ -d "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-200" ]; then
  mkdir -p "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-200/merged-B2B-s200"
  #python3 /dockerdata/LLM-NEO-bk/src/shadow/merge_lora.py \
  #  --adapter_path "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-200" \
  #  --target_base "Qwen/Qwen3-8B-Base" \
  #  --merge_tag "B2B-s200" \
  #  --template "qwen3"
fi

if [ -d "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-210" ]; then
  mkdir -p "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-210/merged-B2B-s210"
  #python3 /dockerdata/LLM-NEO-bk/src/shadow/merge_lora.py \
  #  --adapter_path "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-210" \
  #  --target_base "Qwen/Qwen3-8B-Base" \
  #  --merge_tag "B2B-s210" \
  #  --template "qwen3"
fi

if [ -d "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-220" ]; then
  mkdir -p "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-220/merged-B2B-s220"
  #python3 /dockerdata/LLM-NEO-bk/src/shadow/merge_lora.py \
  #  --adapter_path "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-220" \
  #  --target_base "Qwen/Qwen3-8B-Base" \
  #  --merge_tag "B2B-s220" \
  #  --template "qwen3"
fi

if [ -d "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-230" ]; then
  mkdir -p "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-230/merged-B2B-s230"
  #python3 /dockerdata/LLM-NEO-bk/src/shadow/merge_lora.py \
  #  --adapter_path "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-230" \
  #  --target_base "Qwen/Qwen3-8B-Base" \
  #  --merge_tag "B2B-s230" \
  #  --template "qwen3"
fi

if [ -d "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-240" ]; then
  mkdir -p "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-240/merged-B2B-s240"
  #python3 /dockerdata/LLM-NEO-bk/src/shadow/merge_lora.py \
  #  --adapter_path "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-240" \
  #  --target_base "Qwen/Qwen3-8B-Base" \
  #  --merge_tag "B2B-s240" \
  #  --template "qwen3"
fi

if [ -d "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-250" ]; then
  mkdir -p "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-250/merged-B2B-s250"
  #python3 /dockerdata/LLM-NEO-bk/src/shadow/merge_lora.py \
  #  --adapter_path "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-250" \
  #  --target_base "Qwen/Qwen3-8B-Base" \
  #  --merge_tag "B2B-s250" \
  #  --template "qwen3"
fi

if [ -d "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-260" ]; then
  mkdir -p "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-260/merged-B2B-s260"
  #python3 /dockerdata/LLM-NEO-bk/src/shadow/merge_lora.py \
  #  --adapter_path "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-260" \
  #  --target_base "Qwen/Qwen3-8B-Base" \
  #  --merge_tag "B2B-s260" \
  #  --template "qwen3"
fi

if [ -d "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-270" ]; then
  mkdir -p "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-270/merged-B2B-s270"
  #python3 /dockerdata/LLM-NEO-bk/src/shadow/merge_lora.py \
  #  --adapter_path "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-270" \
  #  --target_base "Qwen/Qwen3-8B-Base" \
  #  --merge_tag "B2B-s270" \
  #  --template "qwen3"
fi

if [ -d "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-280" ]; then
  mkdir -p "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-280/merged-B2B-s280"
  #python3 /dockerdata/LLM-NEO-bk/src/shadow/merge_lora.py \
  #  --adapter_path "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-280" \
  #  --target_base "Qwen/Qwen3-8B-Base" \
  #  --merge_tag "B2B-s280" \
  #  --template "qwen3"
fi

if [ -d "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-290" ]; then
  mkdir -p "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-290/merged-B2B-s290"
  #python3 /dockerdata/LLM-NEO-bk/src/shadow/merge_lora.py \
  #  --adapter_path "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-290" \
  #  --target_base "Qwen/Qwen3-8B-Base" \
  #  --merge_tag "B2B-s290" \
  #  --template "qwen3"
fi

if [ -d "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-300" ]; then
  mkdir -p "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-300/merged-B2B-s300"
  #python3 /dockerdata/LLM-NEO-bk/src/shadow/merge_lora.py \
  #  --adapter_path "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-300" \
  #  --target_base "Qwen/Qwen3-8B-Base" \
  #  --merge_tag "B2B-s300" \
  #  --template "qwen3"
fi

if [ -d "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-310" ]; then
  mkdir -p "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-310/merged-B2B-s310"
  #python3 /dockerdata/LLM-NEO-bk/src/shadow/merge_lora.py \
  #  --adapter_path "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-310" \
  #  --target_base "Qwen/Qwen3-8B-Base" \
  #  --merge_tag "B2B-s310" \
  #  --template "qwen3"
fi

if [ -d "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-320" ]; then
  mkdir -p "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-320/merged-B2B-s320"
  #python3 /dockerdata/LLM-NEO-bk/src/shadow/merge_lora.py \
  #  --adapter_path "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-320" \
  #  --target_base "Qwen/Qwen3-8B-Base" \
  #  --merge_tag "B2B-s320" \
  #  --template "qwen3"
fi

if [ -d "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-330" ]; then
  mkdir -p "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-330/merged-B2B-s330"
  #python3 /dockerdata/LLM-NEO-bk/src/shadow/merge_lora.py \
  #  --adapter_path "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-330" \
  #  --target_base "Qwen/Qwen3-8B-Base" \
  #  --merge_tag "B2B-s330" \
  #  --template "qwen3"
fi

if [ -d "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-340" ]; then
  mkdir -p "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-340/merged-B2B-s340"
  #python3 /dockerdata/LLM-NEO-bk/src/shadow/merge_lora.py \
  #  --adapter_path "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-340" \
  #  --target_base "Qwen/Qwen3-8B-Base" \
  #  --merge_tag "B2B-s340" \
  #  --template "qwen3"
fi

if [ -d "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-350" ]; then
  mkdir -p "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-350/merged-B2B-s350"
  #python3 /dockerdata/LLM-NEO-bk/src/shadow/merge_lora.py \
  #  --adapter_path "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-350" \
  #  --target_base "Qwen/Qwen3-8B-Base" \
  #  --merge_tag "B2B-s350" \
  #  --template "qwen3"
fi

if [ -d "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-360" ]; then
  mkdir -p "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-360/merged-B2B-s360"
  #python3 /dockerdata/LLM-NEO-bk/src/shadow/merge_lora.py \
  #  --adapter_path "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-360" \
  #  --target_base "Qwen/Qwen3-8B-Base" \
  #  --merge_tag "B2B-s360" \
  #  --template "qwen3"
fi

if [ -d "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-370" ]; then
  mkdir -p "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-370/merged-B2B-s370"
  #python3 /dockerdata/LLM-NEO-bk/src/shadow/merge_lora.py \
  #  --adapter_path "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-370" \
  #  --target_base "Qwen/Qwen3-8B-Base" \
  #  --merge_tag "B2B-s370" \
  #  --template "qwen3"
fi

if [ -d "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-380" ]; then
  mkdir -p "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-380/merged-B2B-s380"
  #python3 /dockerdata/LLM-NEO-bk/src/shadow/merge_lora.py \
  #  --adapter_path "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-380" \
  #  --target_base "Qwen/Qwen3-8B-Base" \
  #  --merge_tag "B2B-s380" \
  #  --template "qwen3"
fi

if [ -d "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-390" ]; then
  mkdir -p "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-390/merged-B2B-s390"
  #python3 /dockerdata/LLM-NEO-bk/src/shadow/merge_lora.py \
  #  --adapter_path "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-390" \
  #  --target_base "Qwen/Qwen3-8B-Base" \
  #  --merge_tag "B2B-s390" \
  #  --template "qwen3"
fi

if [ -d "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-400" ]; then
  mkdir -p "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-400/merged-B2B-s400"
  #python3 /dockerdata/LLM-NEO-bk/src/shadow/merge_lora.py \
  #  --adapter_path "/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-400" \
  #  --target_base "Qwen/Qwen3-8B-Base" \
  #  --merge_tag "B2B-s400" \
  #  --template "qwen3"
fi

##### Evaluation list #####
# ('B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-0/merged-B2I-s0','/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-0/merged-B2I-s0'),
# ('B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-10/merged-B2I-s10','/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-10/merged-B2I-s10'),
# ('B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-20/merged-B2I-s20','/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-20/merged-B2I-s20'),
# ('B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-30/merged-B2I-s30','/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-30/merged-B2I-s30'),
# ('B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-40/merged-B2I-s40','/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-40/merged-B2I-s40'),
# ('B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-50/merged-B2I-s50','/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-50/merged-B2I-s50'),
# ('B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-60/merged-B2I-s60','/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-60/merged-B2I-s60'),
# ('B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-70/merged-B2I-s70','/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-70/merged-B2I-s70'),
# ('B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-80/merged-B2I-s80','/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-80/merged-B2I-s80'),
# ('B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-90/merged-B2I-s90','/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-90/merged-B2I-s90'),
# ('B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-100/merged-B2I-s100','/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-100/merged-B2I-s100'),
# ('B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-110/merged-B2I-s110','/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-110/merged-B2I-s110'),
# ('B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-120/merged-B2I-s120','/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-120/merged-B2I-s120'),
# ('B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-130/merged-B2I-s130','/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-130/merged-B2I-s130'),
# ('B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-140/merged-B2I-s140','/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-140/merged-B2I-s140'),
# ('B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-150/merged-B2I-s150','/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-150/merged-B2I-s150'),
# ('B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-160/merged-B2I-s160','/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-160/merged-B2I-s160'),
# ('B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-170/merged-B2I-s170','/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-170/merged-B2I-s170'),
# ('B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-180/merged-B2I-s180','/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-180/merged-B2I-s180'),
# ('B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-190/merged-B2I-s190','/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-190/merged-B2I-s190'),
# ('B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-200/merged-B2I-s200','/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-200/merged-B2I-s200'),
# ('B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-210/merged-B2I-s210','/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-210/merged-B2I-s210'),
# ('B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-220/merged-B2I-s220','/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-220/merged-B2I-s220'),
# ('B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-230/merged-B2I-s230','/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-230/merged-B2I-s230'),
# ('B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-240/merged-B2I-s240','/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-240/merged-B2I-s240'),
# ('B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-250/merged-B2I-s250','/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-250/merged-B2I-s250'),
# ('B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-260/merged-B2I-s260','/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-260/merged-B2I-s260'),
# ('B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-270/merged-B2I-s270','/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-270/merged-B2I-s270'),
# ('B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-280/merged-B2I-s280','/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-280/merged-B2I-s280'),
# ('B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-290/merged-B2I-s290','/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-290/merged-B2I-s290'),
# ('B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-300/merged-B2I-s300','/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-300/merged-B2I-s300'),
# ('B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-310/merged-B2I-s310','/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-310/merged-B2I-s310'),
# ('B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-320/merged-B2I-s320','/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-320/merged-B2I-s320'),
# ('B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-330/merged-B2I-s330','/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-330/merged-B2I-s330'),
# ('B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-340/merged-B2I-s340','/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-340/merged-B2I-s340'),
# ('B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-350/merged-B2I-s350','/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-350/merged-B2I-s350'),
# ('B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-360/merged-B2I-s360','/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-360/merged-B2I-s360'),
# ('B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-370/merged-B2I-s370','/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-370/merged-B2I-s370'),
# ('B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-380/merged-B2I-s380','/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-380/merged-B2I-s380'),
# ('B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-390/merged-B2I-s390','/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-390/merged-B2I-s390'),
# ('B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-400/merged-B2I-s400','/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-400/merged-B2I-s400'),
# ('I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-0/merged-I2I-s0','/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-0/merged-I2I-s0'),
# ('I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-10/merged-I2I-s10','/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-10/merged-I2I-s10'),
# ('I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-20/merged-I2I-s20','/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-20/merged-I2I-s20'),
# ('I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-30/merged-I2I-s30','/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-30/merged-I2I-s30'),
# ('I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-40/merged-I2I-s40','/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-40/merged-I2I-s40'),
# ('I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-50/merged-I2I-s50','/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-50/merged-I2I-s50'),
# ('I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-60/merged-I2I-s60','/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-60/merged-I2I-s60'),
# ('I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-70/merged-I2I-s70','/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-70/merged-I2I-s70'),
# ('I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-80/merged-I2I-s80','/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-80/merged-I2I-s80'),
# ('I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-90/merged-I2I-s90','/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-90/merged-I2I-s90'),
# ('I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-100/merged-I2I-s100','/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-100/merged-I2I-s100'),
# ('I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-110/merged-I2I-s110','/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-110/merged-I2I-s110'),
# ('I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-120/merged-I2I-s120','/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-120/merged-I2I-s120'),
# ('I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-130/merged-I2I-s130','/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-130/merged-I2I-s130'),
# ('I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-140/merged-I2I-s140','/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-140/merged-I2I-s140'),
# ('I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-150/merged-I2I-s150','/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-150/merged-I2I-s150'),
# ('I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-160/merged-I2I-s160','/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-160/merged-I2I-s160'),
# ('I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-170/merged-I2I-s170','/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-170/merged-I2I-s170'),
# ('I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-180/merged-I2I-s180','/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-180/merged-I2I-s180'),
# ('I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-190/merged-I2I-s190','/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-190/merged-I2I-s190'),
# ('I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-200/merged-I2I-s200','/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-200/merged-I2I-s200'),
# ('I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-210/merged-I2I-s210','/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-210/merged-I2I-s210'),
# ('I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-220/merged-I2I-s220','/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-220/merged-I2I-s220'),
# ('I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-230/merged-I2I-s230','/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-230/merged-I2I-s230'),
# ('I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-240/merged-I2I-s240','/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-240/merged-I2I-s240'),
# ('I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-250/merged-I2I-s250','/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-250/merged-I2I-s250'),
# ('I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-260/merged-I2I-s260','/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-260/merged-I2I-s260'),
# ('I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-270/merged-I2I-s270','/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-270/merged-I2I-s270'),
# ('I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-280/merged-I2I-s280','/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-280/merged-I2I-s280'),
# ('I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-290/merged-I2I-s290','/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-290/merged-I2I-s290'),
# ('I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-300/merged-I2I-s300','/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-300/merged-I2I-s300'),
# ('I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-310/merged-I2I-s310','/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-310/merged-I2I-s310'),
# ('I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-320/merged-I2I-s320','/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-320/merged-I2I-s320'),
# ('I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-330/merged-I2I-s330','/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-330/merged-I2I-s330'),
# ('I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-340/merged-I2I-s340','/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-340/merged-I2I-s340'),
# ('I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-350/merged-I2I-s350','/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-350/merged-I2I-s350'),
# ('I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-360/merged-I2I-s360','/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-360/merged-I2I-s360'),
# ('I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-370/merged-I2I-s370','/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-370/merged-I2I-s370'),
# ('I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-380/merged-I2I-s380','/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-380/merged-I2I-s380'),
# ('I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-390/merged-I2I-s390','/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-390/merged-I2I-s390'),
# ('I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-400/merged-I2I-s400','/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-400/merged-I2I-s400'),
## ('I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-0/merged-I2B-s0','/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-0/merged-I2B-s0'),
## ('I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-10/merged-I2B-s10','/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-10/merged-I2B-s10'),
## ('I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-20/merged-I2B-s20','/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-20/merged-I2B-s20'),
## ('I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-30/merged-I2B-s30','/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-30/merged-I2B-s30'),
## ('I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-40/merged-I2B-s40','/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-40/merged-I2B-s40'),
## ('I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-50/merged-I2B-s50','/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-50/merged-I2B-s50'),
## ('I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-60/merged-I2B-s60','/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-60/merged-I2B-s60'),
## ('I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-70/merged-I2B-s70','/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-70/merged-I2B-s70'),
## ('I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-80/merged-I2B-s80','/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-80/merged-I2B-s80'),
## ('I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-90/merged-I2B-s90','/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-90/merged-I2B-s90'),
## ('I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-100/merged-I2B-s100','/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-100/merged-I2B-s100'),
## ('I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-110/merged-I2B-s110','/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-110/merged-I2B-s110'),
## ('I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-120/merged-I2B-s120','/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-120/merged-I2B-s120'),
## ('I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-130/merged-I2B-s130','/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-130/merged-I2B-s130'),
## ('I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-140/merged-I2B-s140','/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-140/merged-I2B-s140'),
## ('I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-150/merged-I2B-s150','/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-150/merged-I2B-s150'),
## ('I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-160/merged-I2B-s160','/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-160/merged-I2B-s160'),
## ('I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-170/merged-I2B-s170','/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-170/merged-I2B-s170'),
## ('I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-180/merged-I2B-s180','/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-180/merged-I2B-s180'),
## ('I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-190/merged-I2B-s190','/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-190/merged-I2B-s190'),
## ('I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-200/merged-I2B-s200','/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-200/merged-I2B-s200'),
## ('I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-210/merged-I2B-s210','/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-210/merged-I2B-s210'),
## ('I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-220/merged-I2B-s220','/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-220/merged-I2B-s220'),
## ('I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-230/merged-I2B-s230','/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-230/merged-I2B-s230'),
## ('I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-240/merged-I2B-s240','/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-240/merged-I2B-s240'),
## ('I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-250/merged-I2B-s250','/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-250/merged-I2B-s250'),
## ('I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-260/merged-I2B-s260','/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-260/merged-I2B-s260'),
## ('I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-270/merged-I2B-s270','/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-270/merged-I2B-s270'),
## ('I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-280/merged-I2B-s280','/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-280/merged-I2B-s280'),
## ('I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-290/merged-I2B-s290','/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-290/merged-I2B-s290'),
## ('I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-300/merged-I2B-s300','/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-300/merged-I2B-s300'),
## ('I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-310/merged-I2B-s310','/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-310/merged-I2B-s310'),
## ('I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-320/merged-I2B-s320','/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-320/merged-I2B-s320'),
## ('I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-330/merged-I2B-s330','/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-330/merged-I2B-s330'),
## ('I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-340/merged-I2B-s340','/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-340/merged-I2B-s340'),
## ('I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-350/merged-I2B-s350','/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-350/merged-I2B-s350'),
## ('I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-360/merged-I2B-s360','/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-360/merged-I2B-s360'),
## ('I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-370/merged-I2B-s370','/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-370/merged-I2B-s370'),
## ('I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-380/merged-I2B-s380','/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-380/merged-I2B-s380'),
## ('I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-390/merged-I2B-s390','/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-390/merged-I2B-s390'),
## ('I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-400/merged-I2B-s400','/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/I-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-400/merged-I2B-s400'),
## ('B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-0/merged-B2B-s0','/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-0/merged-B2B-s0'),
## ('B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-10/merged-B2B-s10','/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-10/merged-B2B-s10'),
## ('B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-20/merged-B2B-s20','/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-20/merged-B2B-s20'),
## ('B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-30/merged-B2B-s30','/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-30/merged-B2B-s30'),
## ('B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-40/merged-B2B-s40','/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-40/merged-B2B-s40'),
## ('B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-50/merged-B2B-s50','/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-50/merged-B2B-s50'),
## ('B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-60/merged-B2B-s60','/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-60/merged-B2B-s60'),
## ('B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-70/merged-B2B-s70','/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-70/merged-B2B-s70'),
## ('B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-80/merged-B2B-s80','/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-80/merged-B2B-s80'),
## ('B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-90/merged-B2B-s90','/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-90/merged-B2B-s90'),
## ('B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-100/merged-B2B-s100','/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-100/merged-B2B-s100'),
## ('B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-110/merged-B2B-s110','/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-110/merged-B2B-s110'),
## ('B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-120/merged-B2B-s120','/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-120/merged-B2B-s120'),
## ('B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-130/merged-B2B-s130','/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-130/merged-B2B-s130'),
## ('B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-140/merged-B2B-s140','/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-140/merged-B2B-s140'),
## ('B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-150/merged-B2B-s150','/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-150/merged-B2B-s150'),
## ('B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-160/merged-B2B-s160','/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-160/merged-B2B-s160'),
## ('B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-170/merged-B2B-s170','/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-170/merged-B2B-s170'),
## ('B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-180/merged-B2B-s180','/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-180/merged-B2B-s180'),
## ('B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-190/merged-B2B-s190','/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-190/merged-B2B-s190'),
## ('B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-200/merged-B2B-s200','/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-200/merged-B2B-s200'),
## ('B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-210/merged-B2B-s210','/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-210/merged-B2B-s210'),
## ('B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-220/merged-B2B-s220','/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-220/merged-B2B-s220'),
## ('B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-230/merged-B2B-s230','/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-230/merged-B2B-s230'),
## ('B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-240/merged-B2B-s240','/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-240/merged-B2B-s240'),
## ('B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-250/merged-B2B-s250','/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-250/merged-B2B-s250'),
## ('B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-260/merged-B2B-s260','/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-260/merged-B2B-s260'),
## ('B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-270/merged-B2B-s270','/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-270/merged-B2B-s270'),
## ('B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-280/merged-B2B-s280','/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-280/merged-B2B-s280'),
## ('B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-290/merged-B2B-s290','/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-290/merged-B2B-s290'),
## ('B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-300/merged-B2B-s300','/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-300/merged-B2B-s300'),
## ('B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-310/merged-B2B-s310','/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-310/merged-B2B-s310'),
## ('B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-320/merged-B2B-s320','/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-320/merged-B2B-s320'),
## ('B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-330/merged-B2B-s330','/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-330/merged-B2B-s330'),
## ('B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-340/merged-B2B-s340','/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-340/merged-B2B-s340'),
## ('B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-350/merged-B2B-s350','/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-350/merged-B2B-s350'),
## ('B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-360/merged-B2B-s360','/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-360/merged-B2B-s360'),
## ('B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-370/merged-B2B-s370','/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-370/merged-B2B-s370'),
## ('B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-380/merged-B2B-s380','/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-380/merged-B2B-s380'),
## ('B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-390/merged-B2B-s390','/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-390/merged-B2B-s390'),
## ('B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-400/merged-B2B-s400','/dockerdata/LLM-NEO-bk/results/1124/result-Qwen3-8B-Base-1124/B-100k-lora-rank128-lr0.0002-BAAI-Infinity-7M-core/checkpoint-400/merged-B2B-s400'),

# please copy this eval_config to opencompass/examples/eval_shadow_202505.py and then run

cd ./opencompass
python3 ./run.py ./examples/eval_shadow_202505.py

