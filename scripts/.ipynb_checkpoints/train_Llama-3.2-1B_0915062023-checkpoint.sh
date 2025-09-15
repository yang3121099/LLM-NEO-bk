##### Auto-generated 2025-09-15 06:20:23 #####
# Model     : Llama-3.2-1B
# LoRA mode : true
# KD mode   : true
# KD ratio  : 0.5
# KD temp   : 1.0
# Template  : llama3

##### Environment #####
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export HF_HUB_OFFLINE=0
export HF_DATASETS_OFFLINE=0
export HF_DATASETS_TRUST_REMOTE_CODE=1
export TRUST_REMOTE_CODE=True
export HF_ALLOW_CODE_EVAL=1

##### Training #####
##### Standard Training #####
###### B  max=1000  lr=0.0002 ######
mkdir -p "/workspace/LLM-NEO-bk/results/0915/result-Llama-3.2-1B-0915/B-1k-lora-rank128-lr0.0002-limo"
cd "/workspace/LLM-NEO-bk"
llamafactory-cli train \
  --model_name_or_path "meta-llama/Llama-3.2-1B" \
  --stage sft \
  --do_train true \
  --finetuning_type lora --lora_rank 128 \
  --dataset "limo" \
  --template "llama3" \
  --cutoff_len 4096 \
  --max_samples 1000 \
  --output_dir "/workspace/LLM-NEO-bk/results/0915/result-Llama-3.2-1B-0915/B-1k-lora-rank128-lr0.0002-limo" \
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

###### I  max=1000  lr=0.0002 ######
mkdir -p "/workspace/LLM-NEO-bk/results/0915/result-Llama-3.2-1B-0915/I-1k-lora-rank128-lr0.0002-limo"
cd "/workspace/LLM-NEO-bk"
llamafactory-cli train \
  --model_name_or_path "meta-llama/Llama-3.2-1B-Instruct" \
  --stage sft \
  --do_train true \
  --finetuning_type lora --lora_rank 128 \
  --dataset "limo" \
  --template "llama3" \
  --cutoff_len 4096 \
  --max_samples 1000 \
  --output_dir "/workspace/LLM-NEO-bk/results/0915/result-Llama-3.2-1B-0915/I-1k-lora-rank128-lr0.0002-limo" \
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


##### Knowledge Distillation Training #####
###### B with KD  max=1000  lr=0.0002 ######
mkdir -p "/workspace/LLM-NEO-bk/results/0915/result-Llama-3.2-1B-0915/B-kd-1k-lora-rank128-lr0.0002-limo"
cd "/workspace/LLM-NEO-bk"
llamafactory-cli train \
  --model_name_or_path "meta-llama/Llama-3.2-1B" \
  --stage sft \
  --do_train true \
  --finetuning_type lora --lora_rank 128 \
  --dataset "limo" \
  --template "llama3" \
  --cutoff_len 4096 \
  --max_samples 1000 \
  --output_dir "/workspace/LLM-NEO-bk/results/0915/result-Llama-3.2-1B-0915/B-kd-1k-lora-rank128-lr0.0002-limo" \
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
  --teacher_model_name_or_path "meta-llama/Llama-3.2-1B-Instruct" \
  --kd_ratio 0.5 \
  --kd_temperature 1.0 \
  --use_fast_tokenizer True

###### I with KD  max=1000  lr=0.0002 ######
mkdir -p "/workspace/LLM-NEO-bk/results/0915/result-Llama-3.2-1B-0915/I-kd-1k-lora-rank128-lr0.0002-limo"
cd "/workspace/LLM-NEO-bk"
llamafactory-cli train \
  --model_name_or_path "meta-llama/Llama-3.2-1B-Instruct" \
  --stage sft \
  --do_train true \
  --finetuning_type lora --lora_rank 128 \
  --dataset "limo" \
  --template "llama3" \
  --cutoff_len 4096 \
  --max_samples 1000 \
  --output_dir "/workspace/LLM-NEO-bk/results/0915/result-Llama-3.2-1B-0915/I-kd-1k-lora-rank128-lr0.0002-limo" \
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
  --teacher_model_name_or_path "meta-llama/Llama-3.2-1B-Instruct" \
  --kd_ratio 0.5 \
  --kd_temperature 1.0 \
  --use_fast_tokenizer True

##### LoRA delta-merge #####
### Standard LoRA Merges ###
mkdir -p "/workspace/LLM-NEO-bk/results/0915/result-Llama-3.2-1B-0915/B-1k-lora-rank128-lr0.0002-limo/merged-B2I"
python3 /workspace/LLM-NEO-bk/src/shadow/merge_lora.py \
  --adapter_path "/workspace/LLM-NEO-bk/results/0915/result-Llama-3.2-1B-0915/B-1k-lora-rank128-lr0.0002-limo" \
  --target_base "meta-llama/Llama-3.2-1B-Instruct" \
  --merge_tag "B2I" \
  --template "llama3"

mkdir -p "/workspace/LLM-NEO-bk/results/0915/result-Llama-3.2-1B-0915/I-1k-lora-rank128-lr0.0002-limo/merged-I2I"
python3 /workspace/LLM-NEO-bk/src/shadow/merge_lora.py \
  --adapter_path "/workspace/LLM-NEO-bk/results/0915/result-Llama-3.2-1B-0915/I-1k-lora-rank128-lr0.0002-limo" \
  --target_base "meta-llama/Llama-3.2-1B-Instruct" \
  --merge_tag "I2I" \
  --template "llama3"

mkdir -p "/workspace/LLM-NEO-bk/results/0915/result-Llama-3.2-1B-0915/I-1k-lora-rank128-lr0.0002-limo/merged-I2B"
#python3 /workspace/LLM-NEO-bk/src/shadow/merge_lora.py \
#  --adapter_path "/workspace/LLM-NEO-bk/results/0915/result-Llama-3.2-1B-0915/I-1k-lora-rank128-lr0.0002-limo" \
#  --target_base "meta-llama/Llama-3.2-1B" \
#  --merge_tag "I2B" \
#  --template "llama3"

mkdir -p "/workspace/LLM-NEO-bk/results/0915/result-Llama-3.2-1B-0915/B-1k-lora-rank128-lr0.0002-limo/merged-B2B"
#python3 /workspace/LLM-NEO-bk/src/shadow/merge_lora.py \
#  --adapter_path "/workspace/LLM-NEO-bk/results/0915/result-Llama-3.2-1B-0915/B-1k-lora-rank128-lr0.0002-limo" \
#  --target_base "meta-llama/Llama-3.2-1B" \
#  --merge_tag "B2B" \
#  --template "llama3"


### KD LoRA Merges ###
mkdir -p "/workspace/LLM-NEO-bk/results/0915/result-Llama-3.2-1B-0915/B-kd-1k-lora-rank128-lr0.0002-limo/merged-B-kd2I"
python3 /workspace/LLM-NEO-bk/src/shadow/merge_lora.py \
  --adapter_path "/workspace/LLM-NEO-bk/results/0915/result-Llama-3.2-1B-0915/B-kd-1k-lora-rank128-lr0.0002-limo" \
  --target_base "meta-llama/Llama-3.2-1B-Instruct" \
  --merge_tag "B-kd2I" \
  --template "llama3"

mkdir -p "/workspace/LLM-NEO-bk/results/0915/result-Llama-3.2-1B-0915/I-kd-1k-lora-rank128-lr0.0002-limo/merged-I-kd2I"
python3 /workspace/LLM-NEO-bk/src/shadow/merge_lora.py \
  --adapter_path "/workspace/LLM-NEO-bk/results/0915/result-Llama-3.2-1B-0915/I-kd-1k-lora-rank128-lr0.0002-limo" \
  --target_base "meta-llama/Llama-3.2-1B-Instruct" \
  --merge_tag "I-kd2I" \
  --template "llama3"

mkdir -p "/workspace/LLM-NEO-bk/results/0915/result-Llama-3.2-1B-0915/I-kd-1k-lora-rank128-lr0.0002-limo/merged-I-kd2B"
#python3 /workspace/LLM-NEO-bk/src/shadow/merge_lora.py \
#  --adapter_path "/workspace/LLM-NEO-bk/results/0915/result-Llama-3.2-1B-0915/I-kd-1k-lora-rank128-lr0.0002-limo" \
#  --target_base "meta-llama/Llama-3.2-1B" \
#  --merge_tag "I-kd2B" \
#  --template "llama3"

mkdir -p "/workspace/LLM-NEO-bk/results/0915/result-Llama-3.2-1B-0915/B-kd-1k-lora-rank128-lr0.0002-limo/merged-B-kd2B"
#python3 /workspace/LLM-NEO-bk/src/shadow/merge_lora.py \
#  --adapter_path "/workspace/LLM-NEO-bk/results/0915/result-Llama-3.2-1B-0915/B-kd-1k-lora-rank128-lr0.0002-limo" \
#  --target_base "meta-llama/Llama-3.2-1B" \
#  --merge_tag "B-kd2B" \
#  --template "llama3"

##### Evaluation list #####
# ('/workspace/LLM-NEO-bk/results/0915/result-Llama-3.2-1B-0915/B-kd-1k-lora-rank128-lr0.0002-limo','/workspace/LLM-NEO-bk/results/0915/result-Llama-3.2-1B-0915/B-kd-1k-lora-rank128-lr0.0002-limo'),
# ('/workspace/LLM-NEO-bk/results/0915/result-Llama-3.2-1B-0915/I-kd-1k-lora-rank128-lr0.0002-limo','/workspace/LLM-NEO-bk/results/0915/result-Llama-3.2-1B-0915/I-kd-1k-lora-rank128-lr0.0002-limo'),
# ('/workspace/LLM-NEO-bk/results/0915/result-Llama-3.2-1B-0915/B-1k-lora-rank128-lr0.0002-limo/merged-B2I','/workspace/LLM-NEO-bk/results/0915/result-Llama-3.2-1B-0915/B-1k-lora-rank128-lr0.0002-limo/merged-B2I'),
# ('/workspace/LLM-NEO-bk/results/0915/result-Llama-3.2-1B-0915/I-1k-lora-rank128-lr0.0002-limo/merged-I2I','/workspace/LLM-NEO-bk/results/0915/result-Llama-3.2-1B-0915/I-1k-lora-rank128-lr0.0002-limo/merged-I2I'),
## ('/workspace/LLM-NEO-bk/results/0915/result-Llama-3.2-1B-0915/I-1k-lora-rank128-lr0.0002-limo/merged-I2B','/workspace/LLM-NEO-bk/results/0915/result-Llama-3.2-1B-0915/I-1k-lora-rank128-lr0.0002-limo/merged-I2B'),
## ('/workspace/LLM-NEO-bk/results/0915/result-Llama-3.2-1B-0915/B-1k-lora-rank128-lr0.0002-limo/merged-B2B','/workspace/LLM-NEO-bk/results/0915/result-Llama-3.2-1B-0915/B-1k-lora-rank128-lr0.0002-limo/merged-B2B'),
# ('/workspace/LLM-NEO-bk/results/0915/result-Llama-3.2-1B-0915/B-kd-1k-lora-rank128-lr0.0002-limo/merged-B-kd2I','/workspace/LLM-NEO-bk/results/0915/result-Llama-3.2-1B-0915/B-kd-1k-lora-rank128-lr0.0002-limo/merged-B-kd2I'),
# ('/workspace/LLM-NEO-bk/results/0915/result-Llama-3.2-1B-0915/I-kd-1k-lora-rank128-lr0.0002-limo/merged-I-kd2I','/workspace/LLM-NEO-bk/results/0915/result-Llama-3.2-1B-0915/I-kd-1k-lora-rank128-lr0.0002-limo/merged-I-kd2I'),
## ('/workspace/LLM-NEO-bk/results/0915/result-Llama-3.2-1B-0915/I-kd-1k-lora-rank128-lr0.0002-limo/merged-I-kd2B','/workspace/LLM-NEO-bk/results/0915/result-Llama-3.2-1B-0915/I-kd-1k-lora-rank128-lr0.0002-limo/merged-I-kd2B'),
## ('/workspace/LLM-NEO-bk/results/0915/result-Llama-3.2-1B-0915/B-kd-1k-lora-rank128-lr0.0002-limo/merged-B-kd2B','/workspace/LLM-NEO-bk/results/0915/result-Llama-3.2-1B-0915/B-kd-1k-lora-rank128-lr0.0002-limo/merged-B-kd2B'),

# please copy this eval_config to opencompass/examples/eval_shadow_202505.py and then run

cd ./opencompass
python3 ./run.py ./examples/eval_shadow_202505.py

