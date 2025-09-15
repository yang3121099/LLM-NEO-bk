##### Auto-generated 2025-07-26 06:30:28 #####
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

##### Training #####
###### B  max=2000  lr=2e-4 ######
mkdir -p "/home/ubuntu/Shadow/results/0726/result-Qwen3-8B-Base-0726/B-2k-lora-rank128-lr0.0002-Shadow_2k"
cd "/home/ubuntu/Shadow"
llamafactory-cli train \
  --model_name_or_path "/home/ubuntu/models//Qwen3-8B-Base" \
  --stage sft \
  --do_train true \
  --finetuning_type lora --lora_rank 128 \
  --dataset "Shadow_2k" \
  --template "qwen3" \
  --cutoff_len 4096 \
  --max_samples 2000 \
  --output_dir "/home/ubuntu/Shadow/results/0726/result-Qwen3-8B-Base-0726/B-2k-lora-rank128-lr0.0002-Shadow_2k" \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 16 \
  --learning_rate 2e-4 \
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
  --overwrite_cache false

###### I  max=2000  lr=2e-4 ######
mkdir -p "/home/ubuntu/Shadow/results/0726/result-Qwen3-8B-Base-0726/I-2k-lora-rank128-lr0.0002-Shadow_2k"
cd "/home/ubuntu/Shadow"
llamafactory-cli train \
  --model_name_or_path "/home/ubuntu/models//Qwen3-8B" \
  --stage sft \
  --do_train true \
  --finetuning_type lora --lora_rank 128 \
  --dataset "Shadow_2k" \
  --template "qwen3" \
  --cutoff_len 4096 \
  --max_samples 2000 \
  --output_dir "/home/ubuntu/Shadow/results/0726/result-Qwen3-8B-Base-0726/I-2k-lora-rank128-lr0.0002-Shadow_2k" \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 16 \
  --learning_rate 2e-4 \
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
  --overwrite_cache false

##### LoRA delta-merge #####
python3 /home/ubuntu/Shadow/src/shadow/merge_lora.py \
  --adapter_path "/home/ubuntu/Shadow/results/0726/result-Qwen3-8B-Base-0726/B-2k-lora-rank128-lr0.0002-Shadow_2k/checkpoint-62" \
  --target_base "Qwen/Qwen3-8B" \
  --merge_tag "B2I" \
  --template "qwen3"
# result-Qwen3-8B-Base-0726/B-2.0k-lora-rank128-lr-Shadow_2k
# result-Qwen3-8B-Base-0726/B-2k-lora-rank128-lr0.0002-Shadow_2k

python3 /home/ubuntu/Shadow/src/shadow/merge_lora.py \
  --adapter_path "/home/ubuntu/Shadow/results/0726/result-Qwen3-8B-Base-0726/I-2k-lora-rank128-lr0.0002-Shadow_2k/checkpoint-62" \
  --target_base "Qwen/Qwen3-8B" \
  --merge_tag "I2I" \
  --template "qwen3"


python3 /home/ubuntu/Shadow/src/shadow/merge_lora.py \
  --adapter_path "/home/ubuntu/Shadow/results/0726/result-Qwen3-8B-Base-0726/B-2k-lora-rank128-lr0.0002-Shadow_2k/checkpoint-62" \
  --target_base "Qwen/Qwen3-8B-Base" \
  --merge_tag "B2B" \
  --template "qwen3"

python3 /home/ubuntu/Shadow/src/shadow/merge_lora.py \
  --adapter_path "/home/ubuntu/Shadow/results/0726/result-Qwen3-8B-Base-0726/I-2k-lora-rank128-lr0.0002-Shadow_2k/checkpoint-62" \
  --target_base "Qwen/Qwen3-8B-Base" \
  --merge_tag "I2B" \
  --template "qwen3"

##### Evaluation list #####
# ('Qwen3-8B-Base_merged_B2I_lora128_lr0.0002_Shadow_2k','/home/ubuntu/Shadow/results/0726/result-Qwen3-8B-Base-0726/merged-B2I'),
# ('Qwen3-8B-Base_merged_I2I_lora128_lr0.0002_Shadow_2k','/home/ubuntu/Shadow/results/0726/result-Qwen3-8B-Base-0726/merged-I2I'),

# please copy this eval_config to opencompass/examples/eval_shadow_202505.py and then run

cd ./opencompass
python3 ./run.py ./examples/eval_shadow_202505.py -r 20250726_074016
