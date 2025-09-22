##### Auto-generated 2025-09-21 01:01:54 #####
# Model     : Llama-3.2-1B
# LoRA mode : false
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
###### B  max=2000  lr=0.00001 ######
mkdir -p "/dockerdata/LLM-NEO-bk/results/0921/result-Llama-3.2-1B-0921/B-2k-sft-lr0.00001-Shadow_2k"
cd "/dockerdata/LLM-NEO-bk"
llamafactory-cli train \
  --model_name_or_path "meta-llama/Llama-3.2-1B" \
  --stage sft \
  --do_train true \
  --finetuning_type full \
  --dataset "Shadow_2k" \
  --template "llama3" \
  --cutoff_len 4096 \
  --max_samples 2000 \
  --output_dir "/dockerdata/LLM-NEO-bk/results/0921/result-Llama-3.2-1B-0921/B-2k-sft-lr0.00001-Shadow_2k" \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 16 \
  --learning_rate 0.00001 \
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

###### I  max=2000  lr=0.00001 ######
mkdir -p "/dockerdata/LLM-NEO-bk/results/0921/result-Llama-3.2-1B-0921/I-2k-sft-lr0.00001-Shadow_2k"
cd "/dockerdata/LLM-NEO-bk"
llamafactory-cli train \
  --model_name_or_path "meta-llama/Llama-3.2-1B-Instruct" \
  --stage sft \
  --do_train true \
  --finetuning_type full \
  --dataset "Shadow_2k" \
  --template "llama3" \
  --cutoff_len 4096 \
  --max_samples 2000 \
  --output_dir "/dockerdata/LLM-NEO-bk/results/0921/result-Llama-3.2-1B-0921/I-2k-sft-lr0.00001-Shadow_2k" \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 16 \
  --learning_rate 0.00001 \
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
###### B with KD  max=2000  lr=0.00001 ######
mkdir -p "/dockerdata/LLM-NEO-bk/results/0921/result-Llama-3.2-1B-0921/B-kd-2k-sft-lr0.00001-Shadow_2k"
cd "/dockerdata/LLM-NEO-bk"
llamafactory-cli train \
  --model_name_or_path "meta-llama/Llama-3.2-1B" \
  --stage sft \
  --do_train true \
  --finetuning_type full \
  --dataset "Shadow_2k" \
  --template "llama3" \
  --cutoff_len 4096 \
  --max_samples 2000 \
  --output_dir "/dockerdata/LLM-NEO-bk/results/0921/result-Llama-3.2-1B-0921/B-kd-2k-sft-lr0.00001-Shadow_2k" \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 32 \
  --learning_rate 0.00001 \
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

###### I with KD  max=2000  lr=0.00001 ######
mkdir -p "/dockerdata/LLM-NEO-bk/results/0921/result-Llama-3.2-1B-0921/I-kd-2k-sft-lr0.00001-Shadow_2k"
cd "/dockerdata/LLM-NEO-bk"
llamafactory-cli train \
  --model_name_or_path "meta-llama/Llama-3.2-1B-Instruct" \
  --stage sft \
  --do_train true \
  --finetuning_type full \
  --dataset "Shadow_2k" \
  --template "llama3" \
  --cutoff_len 4096 \
  --max_samples 2000 \
  --output_dir "/dockerdata/LLM-NEO-bk/results/0921/result-Llama-3.2-1B-0921/I-kd-2k-sft-lr0.00001-Shadow_2k" \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 32 \
  --learning_rate 0.00001 \
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

##### SFT delta-merge #####
### Standard SFT Merge ###
mkdir -p "/dockerdata/LLM-NEO-bk/results/0921/result-Llama-3.2-1B-0921/B-2k-sft-lr0.00001-Shadow_2k/merged-B2I"
python3 /dockerdata/LLM-NEO-bk/src/shadow/apply_diff.py \
  --tuned_model "/dockerdata/LLM-NEO-bk/results/0921/result-Llama-3.2-1B-0921/B-2k-sft-lr0.00001-Shadow_2k" \
  --target_model "meta-llama/Llama-3.2-1B-Instruct" \
  --base_model "meta-llama/Llama-3.2-1B"

### No-op SFT Merge (I2I) ###
mkdir -p "/dockerdata/LLM-NEO-bk/results/0921/result-Llama-3.2-1B-0921/I-2k-sft-lr0.00001-Shadow_2k/merged-I2I"
ln -sfn "/dockerdata/LLM-NEO-bk/results/0921/result-Llama-3.2-1B-0921/I-2k-sft-lr0.00001-Shadow_2k" "/dockerdata/LLM-NEO-bk/results/0921/result-Llama-3.2-1B-0921/I-2k-sft-lr0.00001-Shadow_2k/merged-I2I/model"
echo 'SFT I2I is identity (no delta to merge)' > "/dockerdata/LLM-NEO-bk/results/0921/result-Llama-3.2-1B-0921/I-2k-sft-lr0.00001-Shadow_2k/merged-I2I/README.txt"

### KD SFT Merge ###
mkdir -p "/dockerdata/LLM-NEO-bk/results/0921/result-Llama-3.2-1B-0921/B-kd-2k-sft-lr0.00001-Shadow_2k/merged-B-kd2I"
python3 /dockerdata/LLM-NEO-bk/src/shadow/apply_diff.py \
  --tuned_model "/dockerdata/LLM-NEO-bk/results/0921/result-Llama-3.2-1B-0921/B-kd-2k-sft-lr0.00001-Shadow_2k" \
  --target_model "meta-llama/Llama-3.2-1B-Instruct" \
  --base_model "meta-llama/Llama-3.2-1B"

### No-op KD SFT Merge (I-kd2I) ###
mkdir -p "/dockerdata/LLM-NEO-bk/results/0921/result-Llama-3.2-1B-0921/I-kd-2k-sft-lr0.00001-Shadow_2k/merged-I-kd2I"
ln -sfn "/dockerdata/LLM-NEO-bk/results/0921/result-Llama-3.2-1B-0921/I-kd-2k-sft-lr0.00001-Shadow_2k" "/dockerdata/LLM-NEO-bk/results/0921/result-Llama-3.2-1B-0921/I-kd-2k-sft-lr0.00001-Shadow_2k/merged-I-kd2I/model"
echo 'KD SFT I2I is identity (no delta to merge)' > "/dockerdata/LLM-NEO-bk/results/0921/result-Llama-3.2-1B-0921/I-kd-2k-sft-lr0.00001-Shadow_2k/merged-I-kd2I/README.txt"

##### Evaluation list #####
# ('0921/result-Llama-3.2-1B-0921/B-kd-2k-sft-lr0.00001-Shadow_2k','/dockerdata/LLM-NEO-bk/results/0921/result-Llama-3.2-1B-0921/B-kd-2k-sft-lr0.00001-Shadow_2k'),
# ('0921/result-Llama-3.2-1B-0921/I-kd-2k-sft-lr0.00001-Shadow_2k','/dockerdata/LLM-NEO-bk/results/0921/result-Llama-3.2-1B-0921/I-kd-2k-sft-lr0.00001-Shadow_2k'),
# ('result-Llama-3.2-1B-0921/B-2k-sft-lr0.00001-Shadow_2k/merged-B2I','/dockerdata/LLM-NEO-bk/results/0921/result-Llama-3.2-1B-0921/B-2k-sft-lr0.00001-Shadow_2k/merged-B2I'),
# ('result-Llama-3.2-1B-0921/I-2k-sft-lr0.00001-Shadow_2k/merged-I2I','/dockerdata/LLM-NEO-bk/results/0921/result-Llama-3.2-1B-0921/I-2k-sft-lr0.00001-Shadow_2k/merged-I2I'),
# ('result-Llama-3.2-1B-0921/B-kd-2k-sft-lr0.00001-Shadow_2k/merged-B-kd2I','/dockerdata/LLM-NEO-bk/results/0921/result-Llama-3.2-1B-0921/B-kd-2k-sft-lr0.00001-Shadow_2k/merged-B-kd2I'),
# ('result-Llama-3.2-1B-0921/I-kd-2k-sft-lr0.00001-Shadow_2k/merged-I-kd2I','/dockerdata/LLM-NEO-bk/results/0921/result-Llama-3.2-1B-0921/I-kd-2k-sft-lr0.00001-Shadow_2k/merged-I-kd2I'),

# please copy this eval_config to opencompass/examples/eval_shadow_202505.py and then run

cd ./opencompass
python3 ./run.py ./examples/eval_shadow_202505.py

