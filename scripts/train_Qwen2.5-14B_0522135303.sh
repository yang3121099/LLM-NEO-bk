##### Auto-generated 2025-05-22 13:53:03 #####
# Model     : Qwen2.5-14B
# LoRA mode : true
# Template  : qwen

##### Environment #####
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export HF_HUB_OFFLINE=0
export HF_DATASETS_OFFLINE=0
export HF_DATASETS_TRUST_REMOTE_CODE=1
export TRUST_REMOTE_CODE=True
export HF_ALLOW_CODE_EVAL=1

##### Training #####
----- B  max=2000  lr=1e-5 -----
mkdir -p "/apdcephfs_qy3/share_301069248/users/rummyyang/open-instruct/opensource/Shadow/results/0522/result-Qwen2.5-14B-0522/B-2k-lora-rank128-lr0.00001-Shadow_2k"
conda activate openrlhf
cd "/apdcephfs_qy3/share_301069248/users/rummyyang/open-instruct/opensource/Shadow"
llamafactory-cli train \
  --model_name_or_path "/apdcephfs_qy3/share_301069248/users/rummyyang/minillm/checkpoints/Qwen/Qwen2.5-14B" \
  --stage sft \
  --do_train true \
  --finetuning_type lora --lora_rank 128 \
  --deepspeed "/apdcephfs_qy3/share_301069248/users/rummyyang/open-instruct/opensource/Shadow/examples/deepspeed/ds_z3_config.json" \
  --dataset "Shadow_2k" \
  --template "qwen" \
  --cutoff_len 4096 \
  --max_samples 2000 \
  --output_dir "/apdcephfs_qy3/share_301069248/users/rummyyang/open-instruct/opensource/Shadow/results/0522/result-Qwen2.5-14B-0522/B-2k-lora-rank128-lr0.00001-Shadow_2k" \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 16 \
  --learning_rate 1e-5 \
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

----- I  max=2000  lr=1e-5 -----
mkdir -p "/apdcephfs_qy3/share_301069248/users/rummyyang/open-instruct/opensource/Shadow/results/0522/result-Qwen2.5-14B-0522/I-2k-lora-rank128-lr0.00001-Shadow_2k"
conda activate openrlhf
cd "/apdcephfs_qy3/share_301069248/users/rummyyang/open-instruct/opensource/Shadow"
llamafactory-cli train \
  --model_name_or_path "/apdcephfs_qy3/share_301069248/users/rummyyang/minillm/checkpoints/Qwen/Qwen2.5-14B-Instruct" \
  --stage sft \
  --do_train true \
  --finetuning_type lora --lora_rank 128 \
  --deepspeed "/apdcephfs_qy3/share_301069248/users/rummyyang/open-instruct/opensource/Shadow/examples/deepspeed/ds_z3_config.json" \
  --dataset "Shadow_2k" \
  --template "qwen" \
  --cutoff_len 4096 \
  --max_samples 2000 \
  --output_dir "/apdcephfs_qy3/share_301069248/users/rummyyang/open-instruct/opensource/Shadow/results/0522/result-Qwen2.5-14B-0522/I-2k-lora-rank128-lr0.00001-Shadow_2k" \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 16 \
  --learning_rate 1e-5 \
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
python3 /apdcephfs_qy3/share_301069248/users/rummyyang/open-instruct/opensource/Shadow/LLM-Neo/Shadow/src/merge_lora.py \
  --adapter_path "/apdcephfs_qy3/share_301069248/users/rummyyang/open-instruct/opensource/Shadow/results/0522/result-Qwen2.5-14B-0522/B-2.0k-lora-rank128-ratio0.5-lr-Shadow_2k" \
  --target_base "/apdcephfs_qy3/share_301069248/users/rummyyang/minillm/checkpoints/Qwen/Qwen2.5-14B-Instruct" \
  --merge_tag "B2I" \
  --template "qwen"

python3 /apdcephfs_qy3/share_301069248/users/rummyyang/open-instruct/opensource/Shadow/LLM-Neo/Shadow/src/merge_lora.py \
  --adapter_path "/apdcephfs_qy3/share_301069248/users/rummyyang/open-instruct/opensource/Shadow/results/0522/result-Qwen2.5-14B-0522/I-2.0k-lora-rank128-ratio0.5-lr-Shadow_2k" \
  --target_base "/apdcephfs_qy3/share_301069248/users/rummyyang/minillm/checkpoints/Qwen/Qwen2.5-14B-Instruct" \
  --merge_tag "I2I" \
  --template "qwen"

##### Evaluation list #####
('Qwen2.5-14B_merged_B2I_lora128_lr0.00001_Shadow_2k','/apdcephfs_qy3/share_301069248/users/rummyyang/open-instruct/opensource/Shadow/results/0522/result-Qwen2.5-14B-0522/merged-B2I'),
('Qwen2.5-14B_merged_I2I_lora128_lr0.00001_Shadow_2k','/apdcephfs_qy3/share_301069248/users/rummyyang/open-instruct/opensource/Shadow/results/0522/result-Qwen2.5-14B-0522/merged-I2I'),

