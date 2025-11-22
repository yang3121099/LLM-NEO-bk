##### Auto-generated 2025-11-22 16:01:17 #####
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
###### B  max=2000  lr=0.0002 ######
mkdir -p "/workspace/LLM-NEO-bk/results/1122/result-Qwen3-8B-Base-1122/B-2k-lora-rank64-lr0.0002-Shadow_2k_re_adapt_epoch1"
cd "/workspace/LLM-NEO-bk"
llamafactory-cli train \
  --model_name_or_path "Qwen/Qwen3-8B-Base" \
  --stage sft \
  --do_train true \
  --finetuning_type lora --lora_rank 64 --lora_dropout 0.05 --use_dora True --optim adamw_torch\
  --dataset "Shadow_2k" \
  --template "qwen3" \
  --cutoff_len 4096 \
  --max_samples 2000 \
  --output_dir "/workspace/LLM-NEO-bk/results/1122/result-Qwen3-8B-Base-1122/B-2k-lora-rank64-lr0.0002-Shadow_2k_re_adapt_epoch1" \
  --per_device_train_batch_size 20 \
  --gradient_accumulation_steps 1 \
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
mkdir -p "/workspace/LLM-NEO-bk/results/1122/result-Qwen3-8B-Base-1122/I-2k-lora-rank64-lr0.0002-Shadow_2k_re_adapt_epoch1"
cd "/workspace/LLM-NEO-bk"
llamafactory-cli train \
  --model_name_or_path "Qwen/Qwen3-8B" \
  --stage sft \
  --do_train true \
  --finetuning_type lora --lora_rank 64 --lora_dropout 0.05 --use_dora True --optim adamw_torch\
  --dataset "Shadow_2k" \
  --template "qwen3" \
  --cutoff_len 4096 \
  --max_samples 2000 \
  --output_dir "/workspace/LLM-NEO-bk/results/1122/result-Qwen3-8B-Base-1122/I-2k-lora-rank64-lr0.0002-Shadow_2k_re_adapt_epoch1" \
  --per_device_train_batch_size 20 \
  --gradient_accumulation_steps 1 \
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

##### LoRA delta-merge #####
### Standard LoRA Merges ###
mkdir -p "/workspace/LLM-NEO-bk/results/1122/result-Qwen3-8B-Base-1122/B-2k-lora-rank64-lr0.0002-Shadow_2k_re_adapt_epoch1/merged-B2I"
python3 /workspace/LLM-NEO-bk/src/shadow/merge_lora.py \
  --adapter_path "/workspace/LLM-NEO-bk/results/1122/result-Qwen3-8B-Base-1122/B-2k-lora-rank64-lr0.0002-Shadow_2k_re_adapt_epoch1" \
  --target_base "Qwen/Qwen3-8B" \
  --merge_tag "B2I" \
  --template "qwen3"

mkdir -p "/workspace/LLM-NEO-bk/results/1122/result-Qwen3-8B-Base-1122/I-2k-lora-rank64-lr0.0002-Shadow_2k_re_adapt_epoch1/merged-I2I"
python3 /workspace/LLM-NEO-bk/src/shadow/merge_lora.py \
  --adapter_path "/workspace/LLM-NEO-bk/results/1122/result-Qwen3-8B-Base-1122/I-2k-lora-rank64-lr0.0002-Shadow_2k_re_adapt_epoch1" \
  --target_base "Qwen/Qwen3-8B" \
  --merge_tag "I2I" \
  --template "qwen3"

mkdir -p "/workspace/LLM-NEO-bk/results/1122/result-Qwen3-8B-Base-1122/I-2k-lora-rank64-lr0.0002-Shadow_2k_re_adapt_epoch1/merged-I2B"
#python3 /workspace/LLM-NEO-bk/src/shadow/merge_lora.py \
#  --adapter_path "/workspace/LLM-NEO-bk/results/1122/result-Qwen3-8B-Base-1122/I-2k-lora-rank64-lr0.0002-Shadow_2k_re_adapt_epoch1" \
#  --target_base "Qwen/Qwen3-8B-Base" \
#  --merge_tag "I2B" \
#  --template "qwen3"

mkdir -p "/workspace/LLM-NEO-bk/results/1122/result-Qwen3-8B-Base-1122/B-2k-lora-rank64-lr0.0002-Shadow_2k_re_adapt_epoch1/merged-B2B"
#python3 /workspace/LLM-NEO-bk/src/shadow/merge_lora.py \
#  --adapter_path "/workspace/LLM-NEO-bk/results/1122/result-Qwen3-8B-Base-1122/B-2k-lora-rank64-lr0.0002-Shadow_2k_re_adapt_epoch1" \
#  --target_base "Qwen/Qwen3-8B-Base" \
#  --merge_tag "B2B" \
#  --template "qwen3"

##### Evaluation list #####
# ('result-Qwen3-8B-Base-1122/B-2k-lora-rank64-lr0.0002-Shadow_2k_re_adapt_epoch1/merged-B2I','/workspace/LLM-NEO-bk/results/1122/result-Qwen3-8B-Base-1122/B-2k-lora-rank64-lr0.0002-Shadow_2k_re_adapt_epoch1/merged-B2I'),
# ('result-Qwen3-8B-Base-1122/I-2k-lora-rank64-lr0.0002-Shadow_2k_re_adapt_epoch1/merged-I2I','/workspace/LLM-NEO-bk/results/1122/result-Qwen3-8B-Base-1122/I-2k-lora-rank64-lr0.0002-Shadow_2k_re_adapt_epoch1/merged-I2I'),
## ('result-Qwen3-8B-Base-1122/I-2k-lora-rank64-lr0.0002-Shadow_2k_re_adapt_epoch1/merged-I2B','/workspace/LLM-NEO-bk/results/1122/result-Qwen3-8B-Base-1122/I-2k-lora-rank64-lr0.0002-Shadow_2k_re_adapt_epoch1/merged-I2B'),
## ('result-Qwen3-8B-Base-1122/B-2k-lora-rank64-lr0.0002-Shadow_2k_re_adapt_epoch1/merged-B2B','/workspace/LLM-NEO-bk/results/1122/result-Qwen3-8B-Base-1122/B-2k-lora-rank64-lr0.0002-Shadow_2k_re_adapt_epoch1/merged-B2B'),

# please copy this eval_config to opencompass/examples/eval_shadow_202505.py and then run

# cd ./opencompass
# python3 ./run.py ./examples/eval_shadow_202505.py

