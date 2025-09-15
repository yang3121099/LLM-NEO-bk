##### Auto-generated 2025-08-09 09:28:44 #####
# Model     : Qwen3-4B-Thinking-2507
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
###### B  max=1000  lr=0.0002 ######
mkdir -p "/home/ubuntu/Shadow/results/0809/result-Qwen3-4B-Thinking-2507-0809/B-1k-lora-rank128-lr0.0002-s1k_gptoss20b_low"
cd "/home/ubuntu/Shadow"
llamafactory-cli train \
  --model_name_or_path "Qwen/Qwen3-4B-Thinking-2507" \
  --stage sft \
  --do_train true \
  --finetuning_type lora --lora_rank 128 \
  --dataset "s1k_gptoss20b_low" \
  --template "qwen3" \
  --cutoff_len 4096 \
  --max_samples 1000 \
  --output_dir "/home/ubuntu/Shadow/results/0809/result-Qwen3-4B-Thinking-2507-0809/B-1k-lora-rank128-lr0.0002-s1k_gptoss20b_low" \
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
  --overwrite_cache false

###### I  max=1000  lr=0.0002 ######
mkdir -p "/home/ubuntu/Shadow/results/0809/result-Qwen3-4B-Thinking-2507-0809/I-1k-lora-rank128-lr0.0002-s1k_gptoss20b_low"
cd "/home/ubuntu/Shadow"
llamafactory-cli train \
  --model_name_or_path "Qwen/Qwen3-4B-Instruct-2507" \
  --stage sft \
  --do_train true \
  --finetuning_type lora --lora_rank 128 \
  --dataset "s1k_gptoss20b_low" \
  --template "qwen3" \
  --cutoff_len 4096 \
  --max_samples 1000 \
  --output_dir "/home/ubuntu/Shadow/results/0809/result-Qwen3-4B-Thinking-2507-0809/I-1k-lora-rank128-lr0.0002-s1k_gptoss20b_low" \
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
  --overwrite_cache false

##### LoRA delta-merge #####
mkdir -p "/home/ubuntu/Shadow/results/0809/result-Qwen3-4B-Thinking-2507-0809/B-1k-lora-rank128-lr0.0002-s1k_gptoss20b_low/merged-B2I"
python3 /home/ubuntu/Shadow/src/shadow/merge_lora.py \
  --adapter_path "/home/ubuntu/Shadow/results/0809/result-Qwen3-4B-Thinking-2507-0809/B-1k-lora-rank128-lr0.0002-s1k_gptoss20b_low" \
  --target_base "Qwen/Qwen3-4B-Instruct-2507" \
  --merge_tag "B2I" \
  --template "qwen3"

mkdir -p "/home/ubuntu/Shadow/results/0809/result-Qwen3-4B-Thinking-2507-0809/I-1k-lora-rank128-lr0.0002-s1k_gptoss20b_low/merged-I2I"
python3 /home/ubuntu/Shadow/src/shadow/merge_lora.py \
  --adapter_path "/home/ubuntu/Shadow/results/0809/result-Qwen3-4B-Thinking-2507-0809/I-1k-lora-rank128-lr0.0002-s1k_gptoss20b_low" \
  --target_base "Qwen/Qwen3-4B-Instruct-2507" \
  --merge_tag "I2I" \
  --template "qwen3"

mkdir -p "/home/ubuntu/Shadow/results/0809/result-Qwen3-4B-Thinking-2507-0809/I-1k-lora-rank128-lr0.0002-s1k_gptoss20b_low/merged-I2B"
python3 /home/ubuntu/Shadow/src/shadow/merge_lora.py \
 --adapter_path "/home/ubuntu/Shadow/results/0809/result-Qwen3-4B-Thinking-2507-0809/I-1k-lora-rank128-lr0.0002-s1k_gptoss20b_low" \
 --target_base "Qwen/Qwen3-4B-Thinking-2507" \
 --merge_tag "I2B" \
 --template "qwen3"

mkdir -p "/home/ubuntu/Shadow/results/0809/result-Qwen3-4B-Thinking-2507-0809/B-1k-lora-rank128-lr0.0002-s1k_gptoss20b_low/merged-B2B"
python3 /home/ubuntu/Shadow/src/shadow/merge_lora.py \
 --adapter_path "/home/ubuntu/Shadow/results/0809/result-Qwen3-4B-Thinking-2507-0809/B-1k-lora-rank128-lr0.0002-s1k_gptoss20b_low" \
 --target_base "Qwen/Qwen3-4B-Thinking-2507" \
 --merge_tag "B2B" \
 --template "qwen3"

#### Evaluation list #####
('/home/ubuntu/Shadow/results/0809/result-Qwen3-4B-Thinking-2507-0809/B-1k-lora-rank128-lr0.0002-s1k_gptoss20b_low/merged-B2I','/home/ubuntu/Shadow/results/0809/result-Qwen3-4B-Thinking-2507-0809/B-1k-lora-rank128-lr0.0002-s1k_gptoss20b_low/merged-B2I'),
('/home/ubuntu/Shadow/results/0809/result-Qwen3-4B-Thinking-2507-0809/I-1k-lora-rank128-lr0.0002-s1k_gptoss20b_low/merged-I2I','/home/ubuntu/Shadow/results/0809/result-Qwen3-4B-Thinking-2507-0809/I-1k-lora-rank128-lr0.0002-s1k_gptoss20b_low/merged-I2I'),
('/home/ubuntu/Shadow/results/0809/result-Qwen3-4B-Thinking-2507-0809/I-1k-lora-rank128-lr0.0002-s1k_gptoss20b_low/merged-I2B','/home/ubuntu/Shadow/results/0809/result-Qwen3-4B-Thinking-2507-0809/I-1k-lora-rank128-lr0.0002-s1k_gptoss20b_low/merged-I2B'),
('/home/ubuntu/Shadow/results/0809/result-Qwen3-4B-Thinking-2507-0809/B-1k-lora-rank128-lr0.0002-s1k_gptoss20b_low/merged-B2B','/home/ubuntu/Shadow/results/0809/result-Qwen3-4B-Thinking-2507-0809/B-1k-lora-rank128-lr0.0002-s1k_gptoss20b_low/merged-B2B'),

# please copy this eval_config to opencompass/examples/eval_shadow_202505.py and then run

cd ./opencompass
python3 ./run.py ./examples/eval_shadow_202505.py

