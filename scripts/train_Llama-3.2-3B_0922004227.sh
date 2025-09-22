##### Auto-generated 2025-09-22 00:42:27 #####
# Model     : Llama-3.2-3B
# LoRA mode : true
# KD mode   : true
# KD ratio  : 0.1
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
###### B  max=2000  lr=0.0002 ######
mkdir -p "/dockerdata/LLM-NEO-bk/results/0922/result-Llama-3.2-3B-0922/B-2k-lora-rank128-lr0.0002-Shadow_2k"
cd "/dockerdata/LLM-NEO-bk"
llamafactory-cli train \
  --model_name_or_path "meta-llama/Llama-3.2-3B" \
  --stage sft \
  --do_train true \
  --finetuning_type lora --lora_rank 128 \
  --dataset "Shadow_2k" \
  --template "llama3" \
  --cutoff_len 4096 \
  --max_samples 2000 \
  --output_dir "/dockerdata/LLM-NEO-bk/results/0922/result-Llama-3.2-3B-0922/B-2k-lora-rank128-lr0.0002-Shadow_2k" \
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
mkdir -p "/dockerdata/LLM-NEO-bk/results/0922/result-Llama-3.2-3B-0922/I-2k-lora-rank128-lr0.0002-Shadow_2k"
cd "/dockerdata/LLM-NEO-bk"
llamafactory-cli train \
  --model_name_or_path "meta-llama/Llama-3.2-3B-Instruct" \
  --stage sft \
  --do_train true \
  --finetuning_type lora --lora_rank 128 \
  --dataset "Shadow_2k" \
  --template "llama3" \
  --cutoff_len 4096 \
  --max_samples 2000 \
  --output_dir "/dockerdata/LLM-NEO-bk/results/0922/result-Llama-3.2-3B-0922/I-2k-lora-rank128-lr0.0002-Shadow_2k" \
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
###### B with KD  max=2000  lr=0.0002 ######
mkdir -p "/dockerdata/LLM-NEO-bk/results/0922/result-Llama-3.2-3B-0922/B-kd-2k-lora-rank128-lr0.0002-Shadow_2k"
cd "/dockerdata/LLM-NEO-bk"
llamafactory-cli train \
  --model_name_or_path "meta-llama/Llama-3.2-3B" \
  --stage sft \
  --do_train true \
  --finetuning_type lora --lora_rank 128 \
  --dataset "Shadow_2k" \
  --template "llama3" \
  --cutoff_len 4096 \
  --max_samples 2000 \
  --output_dir "/dockerdata/LLM-NEO-bk/results/0922/result-Llama-3.2-3B-0922/B-kd-2k-lora-rank128-lr0.0002-Shadow_2k" \
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
  --teacher_model_name_or_path "meta-llama/Llama-3.1-8B-Instruct" \
  --kd_ratio 0.1 \
  --use_fast_tokenizer True

###### I with KD  max=2000  lr=0.0002 ######
mkdir -p "/dockerdata/LLM-NEO-bk/results/0922/result-Llama-3.2-3B-0922/I-kd-2k-lora-rank128-lr0.0002-Shadow_2k"
cd "/dockerdata/LLM-NEO-bk"
llamafactory-cli train \
  --model_name_or_path "meta-llama/Llama-3.2-3B-Instruct" \
  --stage sft \
  --do_train true \
  --finetuning_type lora --lora_rank 128 \
  --dataset "Shadow_2k" \
  --template "llama3" \
  --cutoff_len 4096 \
  --max_samples 2000 \
  --output_dir "/dockerdata/LLM-NEO-bk/results/0922/result-Llama-3.2-3B-0922/I-kd-2k-lora-rank128-lr0.0002-Shadow_2k" \
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
  --teacher_model_name_or_path "meta-llama/Llama-3.1-8B-Instruct" \
  --kd_ratio 0.1 \
  --use_fast_tokenizer True

##### LoRA delta-merge #####
### Standard LoRA Merges ###
mkdir -p "/dockerdata/LLM-NEO-bk/results/0922/result-Llama-3.2-3B-0922/B-2k-lora-rank128-lr0.0002-Shadow_2k/merged-B2I"
python3 /dockerdata/LLM-NEO-bk/src/shadow/merge_lora.py \
  --adapter_path "/dockerdata/LLM-NEO-bk/results/0922/result-Llama-3.2-3B-0922/B-2k-lora-rank128-lr0.0002-Shadow_2k" \
  --target_base "meta-llama/Llama-3.2-3B-Instruct" \
  --merge_tag "B2I" \
  --template "llama3"

mkdir -p "/dockerdata/LLM-NEO-bk/results/0922/result-Llama-3.2-3B-0922/I-2k-lora-rank128-lr0.0002-Shadow_2k/merged-I2I"
python3 /dockerdata/LLM-NEO-bk/src/shadow/merge_lora.py \
  --adapter_path "/dockerdata/LLM-NEO-bk/results/0922/result-Llama-3.2-3B-0922/I-2k-lora-rank128-lr0.0002-Shadow_2k" \
  --target_base "meta-llama/Llama-3.2-3B-Instruct" \
  --merge_tag "I2I" \
  --template "llama3"

mkdir -p "/dockerdata/LLM-NEO-bk/results/0922/result-Llama-3.2-3B-0922/I-2k-lora-rank128-lr0.0002-Shadow_2k/merged-I2B"
#python3 /dockerdata/LLM-NEO-bk/src/shadow/merge_lora.py \
#  --adapter_path "/dockerdata/LLM-NEO-bk/results/0922/result-Llama-3.2-3B-0922/I-2k-lora-rank128-lr0.0002-Shadow_2k" \
#  --target_base "meta-llama/Llama-3.2-3B" \
#  --merge_tag "I2B" \
#  --template "llama3"

mkdir -p "/dockerdata/LLM-NEO-bk/results/0922/result-Llama-3.2-3B-0922/B-2k-lora-rank128-lr0.0002-Shadow_2k/merged-B2B"
#python3 /dockerdata/LLM-NEO-bk/src/shadow/merge_lora.py \
#  --adapter_path "/dockerdata/LLM-NEO-bk/results/0922/result-Llama-3.2-3B-0922/B-2k-lora-rank128-lr0.0002-Shadow_2k" \
#  --target_base "meta-llama/Llama-3.2-3B" \
#  --merge_tag "B2B" \
#  --template "llama3"


### KD LoRA Merges ###
mkdir -p "/dockerdata/LLM-NEO-bk/results/0922/result-Llama-3.2-3B-0922/B-kd-2k-lora-rank128-lr0.0002-Shadow_2k/merged-B-kd2I"
python3 /dockerdata/LLM-NEO-bk/src/shadow/merge_lora.py \
  --adapter_path "/dockerdata/LLM-NEO-bk/results/0922/result-Llama-3.2-3B-0922/B-kd-2k-lora-rank128-lr0.0002-Shadow_2k" \
  --target_base "meta-llama/Llama-3.2-3B-Instruct" \
  --merge_tag "B-kd2I" \
  --template "llama3"

mkdir -p "/dockerdata/LLM-NEO-bk/results/0922/result-Llama-3.2-3B-0922/I-kd-2k-lora-rank128-lr0.0002-Shadow_2k/merged-I-kd2I"
python3 /dockerdata/LLM-NEO-bk/src/shadow/merge_lora.py \
  --adapter_path "/dockerdata/LLM-NEO-bk/results/0922/result-Llama-3.2-3B-0922/I-kd-2k-lora-rank128-lr0.0002-Shadow_2k" \
  --target_base "meta-llama/Llama-3.2-3B-Instruct" \
  --merge_tag "I-kd2I" \
  --template "llama3"

mkdir -p "/dockerdata/LLM-NEO-bk/results/0922/result-Llama-3.2-3B-0922/I-kd-2k-lora-rank128-lr0.0002-Shadow_2k/merged-I-kd2B"
#python3 /dockerdata/LLM-NEO-bk/src/shadow/merge_lora.py \
#  --adapter_path "/dockerdata/LLM-NEO-bk/results/0922/result-Llama-3.2-3B-0922/I-kd-2k-lora-rank128-lr0.0002-Shadow_2k" \
#  --target_base "meta-llama/Llama-3.2-3B" \
#  --merge_tag "I-kd2B" \
#  --template "llama3"

mkdir -p "/dockerdata/LLM-NEO-bk/results/0922/result-Llama-3.2-3B-0922/B-kd-2k-lora-rank128-lr0.0002-Shadow_2k/merged-B-kd2B"
#python3 /dockerdata/LLM-NEO-bk/src/shadow/merge_lora.py \
#  --adapter_path "/dockerdata/LLM-NEO-bk/results/0922/result-Llama-3.2-3B-0922/B-kd-2k-lora-rank128-lr0.0002-Shadow_2k" \
#  --target_base "meta-llama/Llama-3.2-3B" \
#  --merge_tag "B-kd2B" \
#  --template "llama3"

##### Evaluation list #####
# ('0922/result-Llama-3.2-3B-0922/B-kd-2k-lora-rank128-lr0.0002-Shadow_2k','/dockerdata/LLM-NEO-bk/results/0922/result-Llama-3.2-3B-0922/B-kd-2k-lora-rank128-lr0.0002-Shadow_2k'),
# ('0922/result-Llama-3.2-3B-0922/I-kd-2k-lora-rank128-lr0.0002-Shadow_2k','/dockerdata/LLM-NEO-bk/results/0922/result-Llama-3.2-3B-0922/I-kd-2k-lora-rank128-lr0.0002-Shadow_2k'),
# ('result-Llama-3.2-3B-0922/B-2k-lora-rank128-lr0.0002-Shadow_2k/merged-B2I','/dockerdata/LLM-NEO-bk/results/0922/result-Llama-3.2-3B-0922/B-2k-lora-rank128-lr0.0002-Shadow_2k/merged-B2I'),
# ('result-Llama-3.2-3B-0922/I-2k-lora-rank128-lr0.0002-Shadow_2k/merged-I2I','/dockerdata/LLM-NEO-bk/results/0922/result-Llama-3.2-3B-0922/I-2k-lora-rank128-lr0.0002-Shadow_2k/merged-I2I'),
## ('result-Llama-3.2-3B-0922/I-2k-lora-rank128-lr0.0002-Shadow_2k/merged-I2B','/dockerdata/LLM-NEO-bk/results/0922/result-Llama-3.2-3B-0922/I-2k-lora-rank128-lr0.0002-Shadow_2k/merged-I2B'),
## ('result-Llama-3.2-3B-0922/B-2k-lora-rank128-lr0.0002-Shadow_2k/merged-B2B','/dockerdata/LLM-NEO-bk/results/0922/result-Llama-3.2-3B-0922/B-2k-lora-rank128-lr0.0002-Shadow_2k/merged-B2B'),
# ('result-Llama-3.2-3B-0922/B-kd-2k-lora-rank128-lr0.0002-Shadow_2k/merged-B-kd2I','/dockerdata/LLM-NEO-bk/results/0922/result-Llama-3.2-3B-0922/B-kd-2k-lora-rank128-lr0.0002-Shadow_2k/merged-B-kd2I'),
# ('result-Llama-3.2-3B-0922/I-kd-2k-lora-rank128-lr0.0002-Shadow_2k/merged-I-kd2I','/dockerdata/LLM-NEO-bk/results/0922/result-Llama-3.2-3B-0922/I-kd-2k-lora-rank128-lr0.0002-Shadow_2k/merged-I-kd2I'),
## ('result-Llama-3.2-3B-0922/I-kd-2k-lora-rank128-lr0.0002-Shadow_2k/merged-I-kd2B','/dockerdata/LLM-NEO-bk/results/0922/result-Llama-3.2-3B-0922/I-kd-2k-lora-rank128-lr0.0002-Shadow_2k/merged-I-kd2B'),
## ('result-Llama-3.2-3B-0922/B-kd-2k-lora-rank128-lr0.0002-Shadow_2k/merged-B-kd2B','/dockerdata/LLM-NEO-bk/results/0922/result-Llama-3.2-3B-0922/B-kd-2k-lora-rank128-lr0.0002-Shadow_2k/merged-B-kd2B'),

# please copy this eval_config to opencompass/examples/eval_shadow_202505.py and then run

cd ./opencompass
python3 ./run.py ./examples/eval_shadow_202505.py

