##### Auto-generated 2025-07-27 14:06:59 #####
# Model     : Qwen3-8B-Base
# LoRA mode : true
# Template  : qwen3

##### Environment #####
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export HF_HUB_OFFLINE=0
export HF_DATASETS_OFFLINE=0
export HF_DATASETS_TRUST_REMOTE_CODE=1
export TRUST_REMOTE_CODE=True
export HF_ALLOW_CODE_EVAL=1

# ===== helper functions for eval toggling and running (executed at runtime) =====
EVAL_RUN_ID=${EVAL_RUN_ID:-$(date +%Y%m%d_%H%M%S)}
OC_DIR="$WORKSPACE_DIR/opencompass"
EVAL_CFG="$OC_DIR/examples/eval_shadow_202505.py"

toggle_eval_line() {
  # Usage: toggle_eval_line <cfg> <relpath> <uncomment|comment> [comment_level]
  local cfg="$1" rel="$2" action="$3" level="${4:-1}"
  python3 - "$cfg" "$rel" "$action" "$level" <<'PY'
import sys,re
cfg, rel, action, level = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
with open(cfg, 'r', encoding='utf-8') as f:
    lines = f.readlines()
target = f"('{rel}',"
def is_tuple_line(s):
    return re.match(r"^\s*#*\s*\('", s) is not None
for i, ln in enumerate(lines):
    if target in ln and is_tuple_line(ln):
        if action == 'uncomment':
            lines[i] = re.sub(r"^\s*#*\s*", "", ln, count=1)
        else:
            prefix = "#" * level + " "
            core = re.sub(r"^\s*#*\s*", "", ln, count=1)
            lines[i] = prefix + core
        break
with open(cfg, 'w', encoding='utf-8') as f:
    f.writelines(lines)
PY
}

run_eval_for_relpath() {
  # Usage: run_eval_for_relpath <relpath> [restore_comment_level]
  local rel="$1" level="${2:-1}"
  local abs="$RESULTS_DIR/$rel"
  echo "[EVAL] enabling entry: $rel"
  toggle_eval_line "$EVAL_CFG" "$rel" uncomment 0
  ( cd "$OC_DIR" && python3 ./run.py ./examples/eval_shadow_202505.py -r "$EVAL_RUN_ID" )
  echo "[EVAL] disabling entry: $rel"
  toggle_eval_line "$EVAL_CFG" "$rel" comment "$level"
  if [[ -d "$abs" ]]; then
    echo "[CLEAN] deleting merged dir: $abs"
    rm -rf "$abs"
  fi
}
##### Training #####
###### B  max=2000  lr=0.000005 ######
mkdir -p "/home/ubuntu/Shadow/results/0727/result-Qwen3-8B-Base-0727/B-2k-lora-rank128-lr0.000005-Shadow_2k"
cd "/home/ubuntu/Shadow"
llamafactory-cli train \
  --model_name_or_path "Qwen/Qwen3-8B-Base" \
  --stage sft \
  --do_train true \
  --finetuning_type lora --lora_rank 128 \
  --dataset "Shadow_2k" \
  --template "qwen3" \
  --cutoff_len 4096 \
  --max_samples 2000 \
  --output_dir "/home/ubuntu/Shadow/results/0727/result-Qwen3-8B-Base-0727/B-2k-lora-rank128-lr0.000005-Shadow_2k" \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 16 \
  --learning_rate 0.000005 \
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
  --save_only_model True \
  --overwrite_cache false

###### I  max=2000  lr=0.000005 ######
mkdir -p "/home/ubuntu/Shadow/results/0727/result-Qwen3-8B-Base-0727/I-2k-lora-rank128-lr0.000005-Shadow_2k"
cd "/home/ubuntu/Shadow"
llamafactory-cli train \
  --model_name_or_path "Qwen/Qwen3-8B" \
  --stage sft \
  --do_train true \
  --finetuning_type lora --lora_rank 128 \
  --dataset "Shadow_2k" \
  --template "qwen3" \
  --cutoff_len 4096 \
  --max_samples 2000 \
  --output_dir "/home/ubuntu/Shadow/results/0727/result-Qwen3-8B-Base-0727/I-2k-lora-rank128-lr0.000005-Shadow_2k" \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 16 \
  --learning_rate 0.000005 \
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
  --save_only_model True \
  --overwrite_cache false

###### B  max=2000  lr=0.00001 ######
mkdir -p "/home/ubuntu/Shadow/results/0727/result-Qwen3-8B-Base-0727/B-2k-lora-rank128-lr0.00001-Shadow_2k"
cd "/home/ubuntu/Shadow"
llamafactory-cli train \
  --model_name_or_path "Qwen/Qwen3-8B-Base" \
  --stage sft \
  --do_train true \
  --finetuning_type lora --lora_rank 128 \
  --dataset "Shadow_2k" \
  --template "qwen3" \
  --cutoff_len 4096 \
  --max_samples 2000 \
  --output_dir "/home/ubuntu/Shadow/results/0727/result-Qwen3-8B-Base-0727/B-2k-lora-rank128-lr0.00001-Shadow_2k" \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 16 \
  --learning_rate 0.00001 \
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
  --save_only_model True \
  --overwrite_cache false

###### I  max=2000  lr=0.00001 ######
mkdir -p "/home/ubuntu/Shadow/results/0727/result-Qwen3-8B-Base-0727/I-2k-lora-rank128-lr0.00001-Shadow_2k"
cd "/home/ubuntu/Shadow"
llamafactory-cli train \
  --model_name_or_path "Qwen/Qwen3-8B" \
  --stage sft \
  --do_train true \
  --finetuning_type lora --lora_rank 128 \
  --dataset "Shadow_2k" \
  --template "qwen3" \
  --cutoff_len 4096 \
  --max_samples 2000 \
  --output_dir "/home/ubuntu/Shadow/results/0727/result-Qwen3-8B-Base-0727/I-2k-lora-rank128-lr0.00001-Shadow_2k" \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 16 \
  --learning_rate 0.00001 \
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
  --save_only_model True \
  --overwrite_cache false

###### B  max=2000  lr=0.00002 ######
mkdir -p "/home/ubuntu/Shadow/results/0727/result-Qwen3-8B-Base-0727/B-2k-lora-rank128-lr0.00002-Shadow_2k"
cd "/home/ubuntu/Shadow"
llamafactory-cli train \
  --model_name_or_path "Qwen/Qwen3-8B-Base" \
  --stage sft \
  --do_train true \
  --finetuning_type lora --lora_rank 128 \
  --dataset "Shadow_2k" \
  --template "qwen3" \
  --cutoff_len 4096 \
  --max_samples 2000 \
  --output_dir "/home/ubuntu/Shadow/results/0727/result-Qwen3-8B-Base-0727/B-2k-lora-rank128-lr0.00002-Shadow_2k" \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 16 \
  --learning_rate 0.00002 \
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
  --save_only_model True \
  --overwrite_cache false

###### I  max=2000  lr=0.00002 ######
mkdir -p "/home/ubuntu/Shadow/results/0727/result-Qwen3-8B-Base-0727/I-2k-lora-rank128-lr0.00002-Shadow_2k"
cd "/home/ubuntu/Shadow"
llamafactory-cli train \
  --model_name_or_path "Qwen/Qwen3-8B" \
  --stage sft \
  --do_train true \
  --finetuning_type lora --lora_rank 128 \
  --dataset "Shadow_2k" \
  --template "qwen3" \
  --cutoff_len 4096 \
  --max_samples 2000 \
  --output_dir "/home/ubuntu/Shadow/results/0727/result-Qwen3-8B-Base-0727/I-2k-lora-rank128-lr0.00002-Shadow_2k" \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 16 \
  --learning_rate 0.00002 \
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
  --save_only_model True \
  --overwrite_cache false

###### B  max=2000  lr=0.00005 ######
mkdir -p "/home/ubuntu/Shadow/results/0727/result-Qwen3-8B-Base-0727/B-2k-lora-rank128-lr0.00005-Shadow_2k"
cd "/home/ubuntu/Shadow"
llamafactory-cli train \
  --model_name_or_path "Qwen/Qwen3-8B-Base" \
  --stage sft \
  --do_train true \
  --finetuning_type lora --lora_rank 128 \
  --dataset "Shadow_2k" \
  --template "qwen3" \
  --cutoff_len 4096 \
  --max_samples 2000 \
  --output_dir "/home/ubuntu/Shadow/results/0727/result-Qwen3-8B-Base-0727/B-2k-lora-rank128-lr0.00005-Shadow_2k" \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 16 \
  --learning_rate 0.00005 \
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
  --save_only_model True \
  --overwrite_cache false

###### I  max=2000  lr=0.00005 ######
mkdir -p "/home/ubuntu/Shadow/results/0727/result-Qwen3-8B-Base-0727/I-2k-lora-rank128-lr0.00005-Shadow_2k"
cd "/home/ubuntu/Shadow"
llamafactory-cli train \
  --model_name_or_path "Qwen/Qwen3-8B" \
  --stage sft \
  --do_train true \
  --finetuning_type lora --lora_rank 128 \
  --dataset "Shadow_2k" \
  --template "qwen3" \
  --cutoff_len 4096 \
  --max_samples 2000 \
  --output_dir "/home/ubuntu/Shadow/results/0727/result-Qwen3-8B-Base-0727/I-2k-lora-rank128-lr0.00005-Shadow_2k" \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 16 \
  --learning_rate 0.00005 \
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
  --save_only_model True \
  --overwrite_cache false

###### B  max=2000  lr=0.0001 ######
mkdir -p "/home/ubuntu/Shadow/results/0727/result-Qwen3-8B-Base-0727/B-2k-lora-rank128-lr0.0001-Shadow_2k"
cd "/home/ubuntu/Shadow"
llamafactory-cli train \
  --model_name_or_path "Qwen/Qwen3-8B-Base" \
  --stage sft \
  --do_train true \
  --finetuning_type lora --lora_rank 128 \
  --dataset "Shadow_2k" \
  --template "qwen3" \
  --cutoff_len 4096 \
  --max_samples 2000 \
  --output_dir "/home/ubuntu/Shadow/results/0727/result-Qwen3-8B-Base-0727/B-2k-lora-rank128-lr0.0001-Shadow_2k" \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 16 \
  --learning_rate 0.0001 \
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
  --save_only_model True \
  --overwrite_cache false

###### I  max=2000  lr=0.0001 ######
mkdir -p "/home/ubuntu/Shadow/results/0727/result-Qwen3-8B-Base-0727/I-2k-lora-rank128-lr0.0001-Shadow_2k"
cd "/home/ubuntu/Shadow"
llamafactory-cli train \
  --model_name_or_path "Qwen/Qwen3-8B" \
  --stage sft \
  --do_train true \
  --finetuning_type lora --lora_rank 128 \
  --dataset "Shadow_2k" \
  --template "qwen3" \
  --cutoff_len 4096 \
  --max_samples 2000 \
  --output_dir "/home/ubuntu/Shadow/results/0727/result-Qwen3-8B-Base-0727/I-2k-lora-rank128-lr0.0001-Shadow_2k" \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 16 \
  --learning_rate 0.0001 \
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
  --save_only_model True \
  --overwrite_cache false

###### B  max=2000  lr=0.0002 ######
mkdir -p "/home/ubuntu/Shadow/results/0727/result-Qwen3-8B-Base-0727/B-2k-lora-rank128-lr0.0002-Shadow_2k"
cd "/home/ubuntu/Shadow"
llamafactory-cli train \
  --model_name_or_path "Qwen/Qwen3-8B-Base" \
  --stage sft \
  --do_train true \
  --finetuning_type lora --lora_rank 128 \
  --dataset "Shadow_2k" \
  --template "qwen3" \
  --cutoff_len 4096 \
  --max_samples 2000 \
  --output_dir "/home/ubuntu/Shadow/results/0727/result-Qwen3-8B-Base-0727/B-2k-lora-rank128-lr0.0002-Shadow_2k" \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 16 \
  --learning_rate 0.0002 \
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
  --save_only_model True \
  --overwrite_cache false

###### I  max=2000  lr=0.0002 ######
mkdir -p "/home/ubuntu/Shadow/results/0727/result-Qwen3-8B-Base-0727/I-2k-lora-rank128-lr0.0002-Shadow_2k"
cd "/home/ubuntu/Shadow"
llamafactory-cli train \
  --model_name_or_path "Qwen/Qwen3-8B" \
  --stage sft \
  --do_train true \
  --finetuning_type lora --lora_rank 128 \
  --dataset "Shadow_2k" \
  --template "qwen3" \
  --cutoff_len 4096 \
  --max_samples 2000 \
  --output_dir "/home/ubuntu/Shadow/results/0727/result-Qwen3-8B-Base-0727/I-2k-lora-rank128-lr0.0002-Shadow_2k" \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 16 \
  --learning_rate 0.0002 \
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
  --save_only_model True \
  --overwrite_cache false

###### B  max=2000  lr=0.0005 ######
mkdir -p "/home/ubuntu/Shadow/results/0727/result-Qwen3-8B-Base-0727/B-2k-lora-rank128-lr0.0005-Shadow_2k"
cd "/home/ubuntu/Shadow"
llamafactory-cli train \
  --model_name_or_path "Qwen/Qwen3-8B-Base" \
  --stage sft \
  --do_train true \
  --finetuning_type lora --lora_rank 128 \
  --dataset "Shadow_2k" \
  --template "qwen3" \
  --cutoff_len 4096 \
  --max_samples 2000 \
  --output_dir "/home/ubuntu/Shadow/results/0727/result-Qwen3-8B-Base-0727/B-2k-lora-rank128-lr0.0005-Shadow_2k" \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 16 \
  --learning_rate 0.0005 \
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
  --save_only_model True \
  --overwrite_cache false

###### I  max=2000  lr=0.0005 ######
mkdir -p "/home/ubuntu/Shadow/results/0727/result-Qwen3-8B-Base-0727/I-2k-lora-rank128-lr0.0005-Shadow_2k"
cd "/home/ubuntu/Shadow"
llamafactory-cli train \
  --model_name_or_path "Qwen/Qwen3-8B" \
  --stage sft \
  --do_train true \
  --finetuning_type lora --lora_rank 128 \
  --dataset "Shadow_2k" \
  --template "qwen3" \
  --cutoff_len 4096 \
  --max_samples 2000 \
  --output_dir "/home/ubuntu/Shadow/results/0727/result-Qwen3-8B-Base-0727/I-2k-lora-rank128-lr0.0005-Shadow_2k" \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 16 \
  --learning_rate 0.0005 \
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
  --save_only_model True \
  --overwrite_cache false

###### B  max=2000  lr=0.001 ######
mkdir -p "/home/ubuntu/Shadow/results/0727/result-Qwen3-8B-Base-0727/B-2k-lora-rank128-lr0.001-Shadow_2k"
cd "/home/ubuntu/Shadow"
llamafactory-cli train \
  --model_name_or_path "Qwen/Qwen3-8B-Base" \
  --stage sft \
  --do_train true \
  --finetuning_type lora --lora_rank 128 \
  --dataset "Shadow_2k" \
  --template "qwen3" \
  --cutoff_len 4096 \
  --max_samples 2000 \
  --output_dir "/home/ubuntu/Shadow/results/0727/result-Qwen3-8B-Base-0727/B-2k-lora-rank128-lr0.001-Shadow_2k" \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 16 \
  --learning_rate 0.001 \
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
  --save_only_model True \
  --overwrite_cache false

###### I  max=2000  lr=0.001 ######
mkdir -p "/home/ubuntu/Shadow/results/0727/result-Qwen3-8B-Base-0727/I-2k-lora-rank128-lr0.001-Shadow_2k"
cd "/home/ubuntu/Shadow"
llamafactory-cli train \
  --model_name_or_path "Qwen/Qwen3-8B" \
  --stage sft \
  --do_train true \
  --finetuning_type lora --lora_rank 128 \
  --dataset "Shadow_2k" \
  --template "qwen3" \
  --cutoff_len 4096 \
  --max_samples 2000 \
  --output_dir "/home/ubuntu/Shadow/results/0727/result-Qwen3-8B-Base-0727/I-2k-lora-rank128-lr0.001-Shadow_2k" \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 16 \
  --learning_rate 0.001 \
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
  --save_only_model True \
  --overwrite_cache false

###### B  max=2000  lr=0.002 ######
mkdir -p "/home/ubuntu/Shadow/results/0727/result-Qwen3-8B-Base-0727/B-2k-lora-rank128-lr0.002-Shadow_2k"
cd "/home/ubuntu/Shadow"
llamafactory-cli train \
  --model_name_or_path "Qwen/Qwen3-8B-Base" \
  --stage sft \
  --do_train true \
  --finetuning_type lora --lora_rank 128 \
  --dataset "Shadow_2k" \
  --template "qwen3" \
  --cutoff_len 4096 \
  --max_samples 2000 \
  --output_dir "/home/ubuntu/Shadow/results/0727/result-Qwen3-8B-Base-0727/B-2k-lora-rank128-lr0.002-Shadow_2k" \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 16 \
  --learning_rate 0.002 \
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
  --save_only_model True \
  --overwrite_cache false

###### I  max=2000  lr=0.002 ######
mkdir -p "/home/ubuntu/Shadow/results/0727/result-Qwen3-8B-Base-0727/I-2k-lora-rank128-lr0.002-Shadow_2k"
cd "/home/ubuntu/Shadow"
llamafactory-cli train \
  --model_name_or_path "Qwen/Qwen3-8B" \
  --stage sft \
  --do_train true \
  --finetuning_type lora --lora_rank 128 \
  --dataset "Shadow_2k" \
  --template "qwen3" \
  --cutoff_len 4096 \
  --max_samples 2000 \
  --output_dir "/home/ubuntu/Shadow/results/0727/result-Qwen3-8B-Base-0727/I-2k-lora-rank128-lr0.002-Shadow_2k" \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 16 \
  --learning_rate 0.002 \
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
  --save_only_model True \
  --overwrite_cache false

##### Case merge & eval - PART 1 (enabled): B2I / I2I #####
python3 /home/ubuntu/Shadow/src/shadow/merge_lora.py \
  --adapter_path "/home/ubuntu/Shadow/results/0727/result-Qwen3-8B-Base-0727/B-2k-lora-rank128-lr0.000005-Shadow_2k" \
  --target_base "Qwen/Qwen3-8B" \
  --merge_tag "B2I" \
  --template "qwen3"

run_eval_for_relpath "0727/result-Qwen3-8B-Base-0727/merged-B2I-lr0.000005" 1
python3 /home/ubuntu/Shadow/src/shadow/merge_lora.py \
  --adapter_path "/home/ubuntu/Shadow/results/0727/result-Qwen3-8B-Base-0727/I-2k-lora-rank128-lr0.000005-Shadow_2k" \
  --target_base "Qwen/Qwen3-8B" \
  --merge_tag "I2I" \
  --template "qwen3"

run_eval_for_relpath "0727/result-Qwen3-8B-Base-0727/merged-I2I-lr0.000005" 1
python3 /home/ubuntu/Shadow/src/shadow/merge_lora.py \
  --adapter_path "/home/ubuntu/Shadow/results/0727/result-Qwen3-8B-Base-0727/B-2k-lora-rank128-lr0.00001-Shadow_2k" \
  --target_base "Qwen/Qwen3-8B" \
  --merge_tag "B2I" \
  --template "qwen3"

run_eval_for_relpath "0727/result-Qwen3-8B-Base-0727/merged-B2I-lr0.00001" 1
python3 /home/ubuntu/Shadow/src/shadow/merge_lora.py \
  --adapter_path "/home/ubuntu/Shadow/results/0727/result-Qwen3-8B-Base-0727/I-2k-lora-rank128-lr0.00001-Shadow_2k" \
  --target_base "Qwen/Qwen3-8B" \
  --merge_tag "I2I" \
  --template "qwen3"

run_eval_for_relpath "0727/result-Qwen3-8B-Base-0727/merged-I2I-lr0.00001" 1
python3 /home/ubuntu/Shadow/src/shadow/merge_lora.py \
  --adapter_path "/home/ubuntu/Shadow/results/0727/result-Qwen3-8B-Base-0727/B-2k-lora-rank128-lr0.00002-Shadow_2k" \
  --target_base "Qwen/Qwen3-8B" \
  --merge_tag "B2I" \
  --template "qwen3"

run_eval_for_relpath "0727/result-Qwen3-8B-Base-0727/merged-B2I-lr0.00002" 1
python3 /home/ubuntu/Shadow/src/shadow/merge_lora.py \
  --adapter_path "/home/ubuntu/Shadow/results/0727/result-Qwen3-8B-Base-0727/I-2k-lora-rank128-lr0.00002-Shadow_2k" \
  --target_base "Qwen/Qwen3-8B" \
  --merge_tag "I2I" \
  --template "qwen3"

run_eval_for_relpath "0727/result-Qwen3-8B-Base-0727/merged-I2I-lr0.00002" 1
python3 /home/ubuntu/Shadow/src/shadow/merge_lora.py \
  --adapter_path "/home/ubuntu/Shadow/results/0727/result-Qwen3-8B-Base-0727/B-2k-lora-rank128-lr0.00005-Shadow_2k" \
  --target_base "Qwen/Qwen3-8B" \
  --merge_tag "B2I" \
  --template "qwen3"

run_eval_for_relpath "0727/result-Qwen3-8B-Base-0727/merged-B2I-lr0.00005" 1
python3 /home/ubuntu/Shadow/src/shadow/merge_lora.py \
  --adapter_path "/home/ubuntu/Shadow/results/0727/result-Qwen3-8B-Base-0727/I-2k-lora-rank128-lr0.00005-Shadow_2k" \
  --target_base "Qwen/Qwen3-8B" \
  --merge_tag "I2I" \
  --template "qwen3"

run_eval_for_relpath "0727/result-Qwen3-8B-Base-0727/merged-I2I-lr0.00005" 1
python3 /home/ubuntu/Shadow/src/shadow/merge_lora.py \
  --adapter_path "/home/ubuntu/Shadow/results/0727/result-Qwen3-8B-Base-0727/B-2k-lora-rank128-lr0.0001-Shadow_2k" \
  --target_base "Qwen/Qwen3-8B" \
  --merge_tag "B2I" \
  --template "qwen3"

run_eval_for_relpath "0727/result-Qwen3-8B-Base-0727/merged-B2I-lr0.0001" 1
python3 /home/ubuntu/Shadow/src/shadow/merge_lora.py \
  --adapter_path "/home/ubuntu/Shadow/results/0727/result-Qwen3-8B-Base-0727/I-2k-lora-rank128-lr0.0001-Shadow_2k" \
  --target_base "Qwen/Qwen3-8B" \
  --merge_tag "I2I" \
  --template "qwen3"

run_eval_for_relpath "0727/result-Qwen3-8B-Base-0727/merged-I2I-lr0.0001" 1
python3 /home/ubuntu/Shadow/src/shadow/merge_lora.py \
  --adapter_path "/home/ubuntu/Shadow/results/0727/result-Qwen3-8B-Base-0727/B-2k-lora-rank128-lr0.0002-Shadow_2k" \
  --target_base "Qwen/Qwen3-8B" \
  --merge_tag "B2I" \
  --template "qwen3"

run_eval_for_relpath "0727/result-Qwen3-8B-Base-0727/merged-B2I-lr0.0002" 1
python3 /home/ubuntu/Shadow/src/shadow/merge_lora.py \
  --adapter_path "/home/ubuntu/Shadow/results/0727/result-Qwen3-8B-Base-0727/I-2k-lora-rank128-lr0.0002-Shadow_2k" \
  --target_base "Qwen/Qwen3-8B" \
  --merge_tag "I2I" \
  --template "qwen3"

run_eval_for_relpath "0727/result-Qwen3-8B-Base-0727/merged-I2I-lr0.0002" 1
python3 /home/ubuntu/Shadow/src/shadow/merge_lora.py \
  --adapter_path "/home/ubuntu/Shadow/results/0727/result-Qwen3-8B-Base-0727/B-2k-lora-rank128-lr0.0005-Shadow_2k" \
  --target_base "Qwen/Qwen3-8B" \
  --merge_tag "B2I" \
  --template "qwen3"

run_eval_for_relpath "0727/result-Qwen3-8B-Base-0727/merged-B2I-lr0.0005" 1
python3 /home/ubuntu/Shadow/src/shadow/merge_lora.py \
  --adapter_path "/home/ubuntu/Shadow/results/0727/result-Qwen3-8B-Base-0727/I-2k-lora-rank128-lr0.0005-Shadow_2k" \
  --target_base "Qwen/Qwen3-8B" \
  --merge_tag "I2I" \
  --template "qwen3"

run_eval_for_relpath "0727/result-Qwen3-8B-Base-0727/merged-I2I-lr0.0005" 1
python3 /home/ubuntu/Shadow/src/shadow/merge_lora.py \
  --adapter_path "/home/ubuntu/Shadow/results/0727/result-Qwen3-8B-Base-0727/B-2k-lora-rank128-lr0.001-Shadow_2k" \
  --target_base "Qwen/Qwen3-8B" \
  --merge_tag "B2I" \
  --template "qwen3"

run_eval_for_relpath "0727/result-Qwen3-8B-Base-0727/merged-B2I-lr0.001" 1
python3 /home/ubuntu/Shadow/src/shadow/merge_lora.py \
  --adapter_path "/home/ubuntu/Shadow/results/0727/result-Qwen3-8B-Base-0727/I-2k-lora-rank128-lr0.001-Shadow_2k" \
  --target_base "Qwen/Qwen3-8B" \
  --merge_tag "I2I" \
  --template "qwen3"

run_eval_for_relpath "0727/result-Qwen3-8B-Base-0727/merged-I2I-lr0.001" 1
python3 /home/ubuntu/Shadow/src/shadow/merge_lora.py \
  --adapter_path "/home/ubuntu/Shadow/results/0727/result-Qwen3-8B-Base-0727/B-2k-lora-rank128-lr0.002-Shadow_2k" \
  --target_base "Qwen/Qwen3-8B" \
  --merge_tag "B2I" \
  --template "qwen3"

run_eval_for_relpath "0727/result-Qwen3-8B-Base-0727/merged-B2I-lr0.002" 1
python3 /home/ubuntu/Shadow/src/shadow/merge_lora.py \
  --adapter_path "/home/ubuntu/Shadow/results/0727/result-Qwen3-8B-Base-0727/I-2k-lora-rank128-lr0.002-Shadow_2k" \
  --target_base "Qwen/Qwen3-8B" \
  --merge_tag "I2I" \
  --template "qwen3"

run_eval_for_relpath "0727/result-Qwen3-8B-Base-0727/merged-I2I-lr0.002" 1
##### Case merge & eval - PART 2 (optional, commented): B2B / I2B #####
#python3 /home/ubuntu/Shadow/src/shadow/merge_lora.py \
#  --adapter_path "/home/ubuntu/Shadow/results/0727/result-Qwen3-8B-Base-0727/B-2k-lora-rank128-lr0.000005-Shadow_2k" \
#  --target_base "Qwen/Qwen3-8B-Base" \
#  --merge_tag "B2B" \
#  --template "qwen3"

#run_eval_for_relpath "0727/result-Qwen3-8B-Base-0727/merged-B2B-lr0.000005" 2
#python3 /home/ubuntu/Shadow/src/shadow/merge_lora.py \
#  --adapter_path "/home/ubuntu/Shadow/results/0727/result-Qwen3-8B-Base-0727/I-2k-lora-rank128-lr0.000005-Shadow_2k" \
#  --target_base "Qwen/Qwen3-8B-Base" \
#  --merge_tag "I2B" \
#  --template "qwen3"

#run_eval_for_relpath "0727/result-Qwen3-8B-Base-0727/merged-I2B-lr0.000005" 2
#python3 /home/ubuntu/Shadow/src/shadow/merge_lora.py \
#  --adapter_path "/home/ubuntu/Shadow/results/0727/result-Qwen3-8B-Base-0727/B-2k-lora-rank128-lr0.00001-Shadow_2k" \
#  --target_base "Qwen/Qwen3-8B-Base" \
#  --merge_tag "B2B" \
#  --template "qwen3"

#run_eval_for_relpath "0727/result-Qwen3-8B-Base-0727/merged-B2B-lr0.00001" 2
#python3 /home/ubuntu/Shadow/src/shadow/merge_lora.py \
#  --adapter_path "/home/ubuntu/Shadow/results/0727/result-Qwen3-8B-Base-0727/I-2k-lora-rank128-lr0.00001-Shadow_2k" \
#  --target_base "Qwen/Qwen3-8B-Base" \
#  --merge_tag "I2B" \
#  --template "qwen3"

#run_eval_for_relpath "0727/result-Qwen3-8B-Base-0727/merged-I2B-lr0.00001" 2
#python3 /home/ubuntu/Shadow/src/shadow/merge_lora.py \
#  --adapter_path "/home/ubuntu/Shadow/results/0727/result-Qwen3-8B-Base-0727/B-2k-lora-rank128-lr0.00002-Shadow_2k" \
#  --target_base "Qwen/Qwen3-8B-Base" \
#  --merge_tag "B2B" \
#  --template "qwen3"

#run_eval_for_relpath "0727/result-Qwen3-8B-Base-0727/merged-B2B-lr0.00002" 2
#python3 /home/ubuntu/Shadow/src/shadow/merge_lora.py \
#  --adapter_path "/home/ubuntu/Shadow/results/0727/result-Qwen3-8B-Base-0727/I-2k-lora-rank128-lr0.00002-Shadow_2k" \
#  --target_base "Qwen/Qwen3-8B-Base" \
#  --merge_tag "I2B" \
#  --template "qwen3"

#run_eval_for_relpath "0727/result-Qwen3-8B-Base-0727/merged-I2B-lr0.00002" 2
#python3 /home/ubuntu/Shadow/src/shadow/merge_lora.py \
#  --adapter_path "/home/ubuntu/Shadow/results/0727/result-Qwen3-8B-Base-0727/B-2k-lora-rank128-lr0.00005-Shadow_2k" \
#  --target_base "Qwen/Qwen3-8B-Base" \
#  --merge_tag "B2B" \
#  --template "qwen3"

#run_eval_for_relpath "0727/result-Qwen3-8B-Base-0727/merged-B2B-lr0.00005" 2
#python3 /home/ubuntu/Shadow/src/shadow/merge_lora.py \
#  --adapter_path "/home/ubuntu/Shadow/results/0727/result-Qwen3-8B-Base-0727/I-2k-lora-rank128-lr0.00005-Shadow_2k" \
#  --target_base "Qwen/Qwen3-8B-Base" \
#  --merge_tag "I2B" \
#  --template "qwen3"

#run_eval_for_relpath "0727/result-Qwen3-8B-Base-0727/merged-I2B-lr0.00005" 2
#python3 /home/ubuntu/Shadow/src/shadow/merge_lora.py \
#  --adapter_path "/home/ubuntu/Shadow/results/0727/result-Qwen3-8B-Base-0727/B-2k-lora-rank128-lr0.0001-Shadow_2k" \
#  --target_base "Qwen/Qwen3-8B-Base" \
#  --merge_tag "B2B" \
#  --template "qwen3"

#run_eval_for_relpath "0727/result-Qwen3-8B-Base-0727/merged-B2B-lr0.0001" 2
#python3 /home/ubuntu/Shadow/src/shadow/merge_lora.py \
#  --adapter_path "/home/ubuntu/Shadow/results/0727/result-Qwen3-8B-Base-0727/I-2k-lora-rank128-lr0.0001-Shadow_2k" \
#  --target_base "Qwen/Qwen3-8B-Base" \
#  --merge_tag "I2B" \
#  --template "qwen3"

#run_eval_for_relpath "0727/result-Qwen3-8B-Base-0727/merged-I2B-lr0.0001" 2
#python3 /home/ubuntu/Shadow/src/shadow/merge_lora.py \
#  --adapter_path "/home/ubuntu/Shadow/results/0727/result-Qwen3-8B-Base-0727/B-2k-lora-rank128-lr0.0002-Shadow_2k" \
#  --target_base "Qwen/Qwen3-8B-Base" \
#  --merge_tag "B2B" \
#  --template "qwen3"

#run_eval_for_relpath "0727/result-Qwen3-8B-Base-0727/merged-B2B-lr0.0002" 2
#python3 /home/ubuntu/Shadow/src/shadow/merge_lora.py \
#  --adapter_path "/home/ubuntu/Shadow/results/0727/result-Qwen3-8B-Base-0727/I-2k-lora-rank128-lr0.0002-Shadow_2k" \
#  --target_base "Qwen/Qwen3-8B-Base" \
#  --merge_tag "I2B" \
#  --template "qwen3"

#run_eval_for_relpath "0727/result-Qwen3-8B-Base-0727/merged-I2B-lr0.0002" 2
#python3 /home/ubuntu/Shadow/src/shadow/merge_lora.py \
#  --adapter_path "/home/ubuntu/Shadow/results/0727/result-Qwen3-8B-Base-0727/B-2k-lora-rank128-lr0.0005-Shadow_2k" \
#  --target_base "Qwen/Qwen3-8B-Base" \
#  --merge_tag "B2B" \
#  --template "qwen3"

#run_eval_for_relpath "0727/result-Qwen3-8B-Base-0727/merged-B2B-lr0.0005" 2
#python3 /home/ubuntu/Shadow/src/shadow/merge_lora.py \
#  --adapter_path "/home/ubuntu/Shadow/results/0727/result-Qwen3-8B-Base-0727/I-2k-lora-rank128-lr0.0005-Shadow_2k" \
#  --target_base "Qwen/Qwen3-8B-Base" \
#  --merge_tag "I2B" \
#  --template "qwen3"

#run_eval_for_relpath "0727/result-Qwen3-8B-Base-0727/merged-I2B-lr0.0005" 2
#python3 /home/ubuntu/Shadow/src/shadow/merge_lora.py \
#  --adapter_path "/home/ubuntu/Shadow/results/0727/result-Qwen3-8B-Base-0727/B-2k-lora-rank128-lr0.001-Shadow_2k" \
#  --target_base "Qwen/Qwen3-8B-Base" \
#  --merge_tag "B2B" \
#  --template "qwen3"

#run_eval_for_relpath "0727/result-Qwen3-8B-Base-0727/merged-B2B-lr0.001" 2
#python3 /home/ubuntu/Shadow/src/shadow/merge_lora.py \
#  --adapter_path "/home/ubuntu/Shadow/results/0727/result-Qwen3-8B-Base-0727/I-2k-lora-rank128-lr0.001-Shadow_2k" \
#  --target_base "Qwen/Qwen3-8B-Base" \
#  --merge_tag "I2B" \
#  --template "qwen3"

#run_eval_for_relpath "0727/result-Qwen3-8B-Base-0727/merged-I2B-lr0.001" 2
#python3 /home/ubuntu/Shadow/src/shadow/merge_lora.py \
#  --adapter_path "/home/ubuntu/Shadow/results/0727/result-Qwen3-8B-Base-0727/B-2k-lora-rank128-lr0.002-Shadow_2k" \
#  --target_base "Qwen/Qwen3-8B-Base" \
#  --merge_tag "B2B" \
#  --template "qwen3"

#run_eval_for_relpath "0727/result-Qwen3-8B-Base-0727/merged-B2B-lr0.002" 2
#python3 /home/ubuntu/Shadow/src/shadow/merge_lora.py \
#  --adapter_path "/home/ubuntu/Shadow/results/0727/result-Qwen3-8B-Base-0727/I-2k-lora-rank128-lr0.002-Shadow_2k" \
#  --target_base "Qwen/Qwen3-8B-Base" \
#  --merge_tag "I2B" \
#  --template "qwen3"

#run_eval_for_relpath "0727/result-Qwen3-8B-Base-0727/merged-I2B-lr0.002" 2
##### Evaluation list - PART 1 (B2I / I2I) #####
# ('0727/result-Qwen3-8B-Base-0727/merged-B2I-lr0.000005','/home/ubuntu/Shadow/results/0727/result-Qwen3-8B-Base-0727/merged-B2I-lr0.000005'),
# ('0727/result-Qwen3-8B-Base-0727/merged-I2I-lr0.000005','/home/ubuntu/Shadow/results/0727/result-Qwen3-8B-Base-0727/merged-I2I-lr0.000005'),
# ('0727/result-Qwen3-8B-Base-0727/merged-B2I-lr0.00001','/home/ubuntu/Shadow/results/0727/result-Qwen3-8B-Base-0727/merged-B2I-lr0.00001'),
# ('0727/result-Qwen3-8B-Base-0727/merged-I2I-lr0.00001','/home/ubuntu/Shadow/results/0727/result-Qwen3-8B-Base-0727/merged-I2I-lr0.00001'),
# ('0727/result-Qwen3-8B-Base-0727/merged-B2I-lr0.00002','/home/ubuntu/Shadow/results/0727/result-Qwen3-8B-Base-0727/merged-B2I-lr0.00002'),
# ('0727/result-Qwen3-8B-Base-0727/merged-I2I-lr0.00002','/home/ubuntu/Shadow/results/0727/result-Qwen3-8B-Base-0727/merged-I2I-lr0.00002'),
# ('0727/result-Qwen3-8B-Base-0727/merged-B2I-lr0.00005','/home/ubuntu/Shadow/results/0727/result-Qwen3-8B-Base-0727/merged-B2I-lr0.00005'),
# ('0727/result-Qwen3-8B-Base-0727/merged-I2I-lr0.00005','/home/ubuntu/Shadow/results/0727/result-Qwen3-8B-Base-0727/merged-I2I-lr0.00005'),
# ('0727/result-Qwen3-8B-Base-0727/merged-B2I-lr0.0001','/home/ubuntu/Shadow/results/0727/result-Qwen3-8B-Base-0727/merged-B2I-lr0.0001'),
# ('0727/result-Qwen3-8B-Base-0727/merged-I2I-lr0.0001','/home/ubuntu/Shadow/results/0727/result-Qwen3-8B-Base-0727/merged-I2I-lr0.0001'),
# ('0727/result-Qwen3-8B-Base-0727/merged-B2I-lr0.0002','/home/ubuntu/Shadow/results/0727/result-Qwen3-8B-Base-0727/merged-B2I-lr0.0002'),
# ('0727/result-Qwen3-8B-Base-0727/merged-I2I-lr0.0002','/home/ubuntu/Shadow/results/0727/result-Qwen3-8B-Base-0727/merged-I2I-lr0.0002'),
# ('0727/result-Qwen3-8B-Base-0727/merged-B2I-lr0.0005','/home/ubuntu/Shadow/results/0727/result-Qwen3-8B-Base-0727/merged-B2I-lr0.0005'),
# ('0727/result-Qwen3-8B-Base-0727/merged-I2I-lr0.0005','/home/ubuntu/Shadow/results/0727/result-Qwen3-8B-Base-0727/merged-I2I-lr0.0005'),
# ('0727/result-Qwen3-8B-Base-0727/merged-B2I-lr0.001','/home/ubuntu/Shadow/results/0727/result-Qwen3-8B-Base-0727/merged-B2I-lr0.001'),
# ('0727/result-Qwen3-8B-Base-0727/merged-I2I-lr0.001','/home/ubuntu/Shadow/results/0727/result-Qwen3-8B-Base-0727/merged-I2I-lr0.001'),
# ('0727/result-Qwen3-8B-Base-0727/merged-B2I-lr0.002','/home/ubuntu/Shadow/results/0727/result-Qwen3-8B-Base-0727/merged-B2I-lr0.002'),
# ('0727/result-Qwen3-8B-Base-0727/merged-I2I-lr0.002','/home/ubuntu/Shadow/results/0727/result-Qwen3-8B-Base-0727/merged-I2I-lr0.002'),

##### Evaluation list - PART 2 (B2B / I2B, commented) #####
## ('0727/result-Qwen3-8B-Base-0727/merged-B2B-lr0.000005','/home/ubuntu/Shadow/results/0727/result-Qwen3-8B-Base-0727/merged-B2B-lr0.000005'),
## ('0727/result-Qwen3-8B-Base-0727/merged-I2B-lr0.000005','/home/ubuntu/Shadow/results/0727/result-Qwen3-8B-Base-0727/merged-I2B-lr0.000005'),
## ('0727/result-Qwen3-8B-Base-0727/merged-B2B-lr0.00001','/home/ubuntu/Shadow/results/0727/result-Qwen3-8B-Base-0727/merged-B2B-lr0.00001'),
## ('0727/result-Qwen3-8B-Base-0727/merged-I2B-lr0.00001','/home/ubuntu/Shadow/results/0727/result-Qwen3-8B-Base-0727/merged-I2B-lr0.00001'),
## ('0727/result-Qwen3-8B-Base-0727/merged-B2B-lr0.00002','/home/ubuntu/Shadow/results/0727/result-Qwen3-8B-Base-0727/merged-B2B-lr0.00002'),
## ('0727/result-Qwen3-8B-Base-0727/merged-I2B-lr0.00002','/home/ubuntu/Shadow/results/0727/result-Qwen3-8B-Base-0727/merged-I2B-lr0.00002'),
## ('0727/result-Qwen3-8B-Base-0727/merged-B2B-lr0.00005','/home/ubuntu/Shadow/results/0727/result-Qwen3-8B-Base-0727/merged-B2B-lr0.00005'),
## ('0727/result-Qwen3-8B-Base-0727/merged-I2B-lr0.00005','/home/ubuntu/Shadow/results/0727/result-Qwen3-8B-Base-0727/merged-I2B-lr0.00005'),
## ('0727/result-Qwen3-8B-Base-0727/merged-B2B-lr0.0001','/home/ubuntu/Shadow/results/0727/result-Qwen3-8B-Base-0727/merged-B2B-lr0.0001'),
## ('0727/result-Qwen3-8B-Base-0727/merged-I2B-lr0.0001','/home/ubuntu/Shadow/results/0727/result-Qwen3-8B-Base-0727/merged-I2B-lr0.0001'),
## ('0727/result-Qwen3-8B-Base-0727/merged-B2B-lr0.0002','/home/ubuntu/Shadow/results/0727/result-Qwen3-8B-Base-0727/merged-B2B-lr0.0002'),
## ('0727/result-Qwen3-8B-Base-0727/merged-I2B-lr0.0002','/home/ubuntu/Shadow/results/0727/result-Qwen3-8B-Base-0727/merged-I2B-lr0.0002'),
## ('0727/result-Qwen3-8B-Base-0727/merged-B2B-lr0.0005','/home/ubuntu/Shadow/results/0727/result-Qwen3-8B-Base-0727/merged-B2B-lr0.0005'),
## ('0727/result-Qwen3-8B-Base-0727/merged-I2B-lr0.0005','/home/ubuntu/Shadow/results/0727/result-Qwen3-8B-Base-0727/merged-I2B-lr0.0005'),
## ('0727/result-Qwen3-8B-Base-0727/merged-B2B-lr0.001','/home/ubuntu/Shadow/results/0727/result-Qwen3-8B-Base-0727/merged-B2B-lr0.001'),
## ('0727/result-Qwen3-8B-Base-0727/merged-I2B-lr0.001','/home/ubuntu/Shadow/results/0727/result-Qwen3-8B-Base-0727/merged-I2B-lr0.001'),
## ('0727/result-Qwen3-8B-Base-0727/merged-B2B-lr0.002','/home/ubuntu/Shadow/results/0727/result-Qwen3-8B-Base-0727/merged-B2B-lr0.002'),
## ('0727/result-Qwen3-8B-Base-0727/merged-I2B-lr0.002','/home/ubuntu/Shadow/results/0727/result-Qwen3-8B-Base-0727/merged-I2B-lr0.002'),

##### Abbrev list - PART 1 (B2I / I2I) #####
# B2I-lr0.000005 : '0727/result-Qwen3-8B-Base-0727/merged-B2I-lr0.000005'
# I2I-lr0.000005 : '0727/result-Qwen3-8B-Base-0727/merged-I2I-lr0.000005'
# B2I-lr0.00001 : '0727/result-Qwen3-8B-Base-0727/merged-B2I-lr0.00001'
# I2I-lr0.00001 : '0727/result-Qwen3-8B-Base-0727/merged-I2I-lr0.00001'
# B2I-lr0.00002 : '0727/result-Qwen3-8B-Base-0727/merged-B2I-lr0.00002'
# I2I-lr0.00002 : '0727/result-Qwen3-8B-Base-0727/merged-I2I-lr0.00002'
# B2I-lr0.00005 : '0727/result-Qwen3-8B-Base-0727/merged-B2I-lr0.00005'
# I2I-lr0.00005 : '0727/result-Qwen3-8B-Base-0727/merged-I2I-lr0.00005'
# B2I-lr0.0001 : '0727/result-Qwen3-8B-Base-0727/merged-B2I-lr0.0001'
# I2I-lr0.0001 : '0727/result-Qwen3-8B-Base-0727/merged-I2I-lr0.0001'
# B2I-lr0.0002 : '0727/result-Qwen3-8B-Base-0727/merged-B2I-lr0.0002'
# I2I-lr0.0002 : '0727/result-Qwen3-8B-Base-0727/merged-I2I-lr0.0002'
# B2I-lr0.0005 : '0727/result-Qwen3-8B-Base-0727/merged-B2I-lr0.0005'
# I2I-lr0.0005 : '0727/result-Qwen3-8B-Base-0727/merged-I2I-lr0.0005'
# B2I-lr0.001 : '0727/result-Qwen3-8B-Base-0727/merged-B2I-lr0.001'
# I2I-lr0.001 : '0727/result-Qwen3-8B-Base-0727/merged-I2I-lr0.001'
# B2I-lr0.002 : '0727/result-Qwen3-8B-Base-0727/merged-B2I-lr0.002'
# I2I-lr0.002 : '0727/result-Qwen3-8B-Base-0727/merged-I2I-lr0.002'

##### Abbrev list - PART 2 (B2B / I2B, commented) #####
## B2B-lr0.000005 : '0727/result-Qwen3-8B-Base-0727/merged-B2B-lr0.000005'
## I2B-lr0.000005 : '0727/result-Qwen3-8B-Base-0727/merged-I2B-lr0.000005'
## B2B-lr0.00001 : '0727/result-Qwen3-8B-Base-0727/merged-B2B-lr0.00001'
## I2B-lr0.00001 : '0727/result-Qwen3-8B-Base-0727/merged-I2B-lr0.00001'
## B2B-lr0.00002 : '0727/result-Qwen3-8B-Base-0727/merged-B2B-lr0.00002'
## I2B-lr0.00002 : '0727/result-Qwen3-8B-Base-0727/merged-I2B-lr0.00002'
## B2B-lr0.00005 : '0727/result-Qwen3-8B-Base-0727/merged-B2B-lr0.00005'
## I2B-lr0.00005 : '0727/result-Qwen3-8B-Base-0727/merged-I2B-lr0.00005'
## B2B-lr0.0001 : '0727/result-Qwen3-8B-Base-0727/merged-B2B-lr0.0001'
## I2B-lr0.0001 : '0727/result-Qwen3-8B-Base-0727/merged-I2B-lr0.0001'
## B2B-lr0.0002 : '0727/result-Qwen3-8B-Base-0727/merged-B2B-lr0.0002'
## I2B-lr0.0002 : '0727/result-Qwen3-8B-Base-0727/merged-I2B-lr0.0002'
## B2B-lr0.0005 : '0727/result-Qwen3-8B-Base-0727/merged-B2B-lr0.0005'
## I2B-lr0.0005 : '0727/result-Qwen3-8B-Base-0727/merged-I2B-lr0.0005'
## B2B-lr0.001 : '0727/result-Qwen3-8B-Base-0727/merged-B2B-lr0.001'
## I2B-lr0.001 : '0727/result-Qwen3-8B-Base-0727/merged-I2B-lr0.001'
## B2B-lr0.002 : '0727/result-Qwen3-8B-Base-0727/merged-B2B-lr0.002'
## I2B-lr0.002 : '0727/result-Qwen3-8B-Base-0727/merged-I2B-lr0.002'

# please copy this eval_config to opencompass/examples/eval_shadow_202505.py and then run

cd ./opencompass
python3 ./run.py ./examples/eval_shadow_202505.py -r 20250727_200010

