##### Auto-generated 2025-07-28 09:20:50 #####
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
###### B  max=2000  lr=5e-6 ######
mkdir -p "/home/ubuntu/Shadow/results/0728/result-Qwen3-8B-Base-0728/B-2k-lora-rank128-lr5e-6-Shadow_2k"
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
  --output_dir "/home/ubuntu/Shadow/results/0728/result-Qwen3-8B-Base-0728/B-2k-lora-rank128-lr5e-6-Shadow_2k" \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 16 \
  --learning_rate 5e-6 \
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

###### I  max=2000  lr=5e-6 ######
mkdir -p "/home/ubuntu/Shadow/results/0728/result-Qwen3-8B-Base-0728/I-2k-lora-rank128-lr5e-6-Shadow_2k"
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
  --output_dir "/home/ubuntu/Shadow/results/0728/result-Qwen3-8B-Base-0728/I-2k-lora-rank128-lr5e-6-Shadow_2k" \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 16 \
  --learning_rate 5e-6 \
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

###### B  max=2000  lr=1e-5 ######
mkdir -p "/home/ubuntu/Shadow/results/0728/result-Qwen3-8B-Base-0728/B-2k-lora-rank128-lr1e-5-Shadow_2k"
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
  --output_dir "/home/ubuntu/Shadow/results/0728/result-Qwen3-8B-Base-0728/B-2k-lora-rank128-lr1e-5-Shadow_2k" \
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

###### I  max=2000  lr=1e-5 ######
mkdir -p "/home/ubuntu/Shadow/results/0728/result-Qwen3-8B-Base-0728/I-2k-lora-rank128-lr1e-5-Shadow_2k"
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
  --output_dir "/home/ubuntu/Shadow/results/0728/result-Qwen3-8B-Base-0728/I-2k-lora-rank128-lr1e-5-Shadow_2k" \
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

###### B  max=2000  lr=2e-5 ######
mkdir -p "/home/ubuntu/Shadow/results/0728/result-Qwen3-8B-Base-0728/B-2k-lora-rank128-lr2e-5-Shadow_2k"
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
  --output_dir "/home/ubuntu/Shadow/results/0728/result-Qwen3-8B-Base-0728/B-2k-lora-rank128-lr2e-5-Shadow_2k" \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 16 \
  --learning_rate 2e-5 \
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

###### I  max=2000  lr=2e-5 ######
mkdir -p "/home/ubuntu/Shadow/results/0728/result-Qwen3-8B-Base-0728/I-2k-lora-rank128-lr2e-5-Shadow_2k"
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
  --output_dir "/home/ubuntu/Shadow/results/0728/result-Qwen3-8B-Base-0728/I-2k-lora-rank128-lr2e-5-Shadow_2k" \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 16 \
  --learning_rate 2e-5 \
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

###### B  max=2000  lr=5e-5 ######
mkdir -p "/home/ubuntu/Shadow/results/0728/result-Qwen3-8B-Base-0728/B-2k-lora-rank128-lr5e-5-Shadow_2k"
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
  --output_dir "/home/ubuntu/Shadow/results/0728/result-Qwen3-8B-Base-0728/B-2k-lora-rank128-lr5e-5-Shadow_2k" \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 16 \
  --learning_rate 5e-5 \
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

###### I  max=2000  lr=5e-5 ######
mkdir -p "/home/ubuntu/Shadow/results/0728/result-Qwen3-8B-Base-0728/I-2k-lora-rank128-lr5e-5-Shadow_2k"
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
  --output_dir "/home/ubuntu/Shadow/results/0728/result-Qwen3-8B-Base-0728/I-2k-lora-rank128-lr5e-5-Shadow_2k" \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 16 \
  --learning_rate 5e-5 \
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

###### B  max=2000  lr=1e-4 ######
mkdir -p "/home/ubuntu/Shadow/results/0728/result-Qwen3-8B-Base-0728/B-2k-lora-rank128-lr1e-4-Shadow_2k"
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
  --output_dir "/home/ubuntu/Shadow/results/0728/result-Qwen3-8B-Base-0728/B-2k-lora-rank128-lr1e-4-Shadow_2k" \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 16 \
  --learning_rate 1e-4 \
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

###### I  max=2000  lr=1e-4 ######
mkdir -p "/home/ubuntu/Shadow/results/0728/result-Qwen3-8B-Base-0728/I-2k-lora-rank128-lr1e-4-Shadow_2k"
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
  --output_dir "/home/ubuntu/Shadow/results/0728/result-Qwen3-8B-Base-0728/I-2k-lora-rank128-lr1e-4-Shadow_2k" \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 16 \
  --learning_rate 1e-4 \
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

###### B  max=2000  lr=2e-4 ######
mkdir -p "/home/ubuntu/Shadow/results/0728/result-Qwen3-8B-Base-0728/B-2k-lora-rank128-lr2e-4-Shadow_2k"
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
  --output_dir "/home/ubuntu/Shadow/results/0728/result-Qwen3-8B-Base-0728/B-2k-lora-rank128-lr2e-4-Shadow_2k" \
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
mkdir -p "/home/ubuntu/Shadow/results/0728/result-Qwen3-8B-Base-0728/I-2k-lora-rank128-lr2e-4-Shadow_2k"
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
  --output_dir "/home/ubuntu/Shadow/results/0728/result-Qwen3-8B-Base-0728/I-2k-lora-rank128-lr2e-4-Shadow_2k" \
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

###### B  max=2000  lr=5e-4 ######
mkdir -p "/home/ubuntu/Shadow/results/0728/result-Qwen3-8B-Base-0728/B-2k-lora-rank128-lr5e-4-Shadow_2k"
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
  --output_dir "/home/ubuntu/Shadow/results/0728/result-Qwen3-8B-Base-0728/B-2k-lora-rank128-lr5e-4-Shadow_2k" \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 16 \
  --learning_rate 5e-4 \
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

###### I  max=2000  lr=5e-4 ######
mkdir -p "/home/ubuntu/Shadow/results/0728/result-Qwen3-8B-Base-0728/I-2k-lora-rank128-lr5e-4-Shadow_2k"
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
  --output_dir "/home/ubuntu/Shadow/results/0728/result-Qwen3-8B-Base-0728/I-2k-lora-rank128-lr5e-4-Shadow_2k" \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 16 \
  --learning_rate 5e-4 \
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

###### B  max=2000  lr=1e-3 ######
mkdir -p "/home/ubuntu/Shadow/results/0728/result-Qwen3-8B-Base-0728/B-2k-lora-rank128-lr1e-3-Shadow_2k"
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
  --output_dir "/home/ubuntu/Shadow/results/0728/result-Qwen3-8B-Base-0728/B-2k-lora-rank128-lr1e-3-Shadow_2k" \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 16 \
  --learning_rate 1e-3 \
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

###### I  max=2000  lr=1e-3 ######
mkdir -p "/home/ubuntu/Shadow/results/0728/result-Qwen3-8B-Base-0728/I-2k-lora-rank128-lr1e-3-Shadow_2k"
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
  --output_dir "/home/ubuntu/Shadow/results/0728/result-Qwen3-8B-Base-0728/I-2k-lora-rank128-lr1e-3-Shadow_2k" \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 16 \
  --learning_rate 1e-3 \
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

###### B  max=2000  lr=2e-3 ######
mkdir -p "/home/ubuntu/Shadow/results/0728/result-Qwen3-8B-Base-0728/B-2k-lora-rank128-lr2e-3-Shadow_2k"
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
  --output_dir "/home/ubuntu/Shadow/results/0728/result-Qwen3-8B-Base-0728/B-2k-lora-rank128-lr2e-3-Shadow_2k" \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 16 \
  --learning_rate 2e-3 \
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

###### I  max=2000  lr=2e-3 ######
mkdir -p "/home/ubuntu/Shadow/results/0728/result-Qwen3-8B-Base-0728/I-2k-lora-rank128-lr2e-3-Shadow_2k"
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
  --output_dir "/home/ubuntu/Shadow/results/0728/result-Qwen3-8B-Base-0728/I-2k-lora-rank128-lr2e-3-Shadow_2k" \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 16 \
  --learning_rate 2e-3 \
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
  --adapter_path "/home/ubuntu/Shadow/results/0728/result-Qwen3-8B-Base-0728/B-2k-lora-rank128-lr5e-6-Shadow_2k" \
  --target_base "Qwen/Qwen3-8B" \
  --merge_tag "B2I" \
  --template "qwen3"

python3 /home/ubuntu/Shadow/src/shadow/merge_lora.py \
  --adapter_path "/home/ubuntu/Shadow/results/0728/result-Qwen3-8B-Base-0728/I-2k-lora-rank128-lr5e-6-Shadow_2k" \
  --target_base "Qwen/Qwen3-8B" \
  --merge_tag "I2I" \
  --template "qwen3"

#python3 /home/ubuntu/Shadow/src/shadow/merge_lora.py \
#  --adapter_path "/home/ubuntu/Shadow/results/0728/result-Qwen3-8B-Base-0728/I-2k-lora-rank128-lr5e-6-Shadow_2k" \
#  --target_base "Qwen/Qwen3-8B-Base" \
#  --merge_tag "I2B" \
#  --template "qwen3"

#python3 /home/ubuntu/Shadow/src/shadow/merge_lora.py \
#  --adapter_path "/home/ubuntu/Shadow/results/0728/result-Qwen3-8B-Base-0728/B-2k-lora-rank128-lr5e-6-Shadow_2k" \
#  --target_base "Qwen/Qwen3-8B-Base" \
#  --merge_tag "B2B" \
#  --template "qwen3"

python3 /home/ubuntu/Shadow/src/shadow/merge_lora.py \
  --adapter_path "/home/ubuntu/Shadow/results/0728/result-Qwen3-8B-Base-0728/B-2k-lora-rank128-lr1e-5-Shadow_2k" \
  --target_base "Qwen/Qwen3-8B" \
  --merge_tag "B2I" \
  --template "qwen3"

python3 /home/ubuntu/Shadow/src/shadow/merge_lora.py \
  --adapter_path "/home/ubuntu/Shadow/results/0728/result-Qwen3-8B-Base-0728/I-2k-lora-rank128-lr1e-5-Shadow_2k" \
  --target_base "Qwen/Qwen3-8B" \
  --merge_tag "I2I" \
  --template "qwen3"

#python3 /home/ubuntu/Shadow/src/shadow/merge_lora.py \
#  --adapter_path "/home/ubuntu/Shadow/results/0728/result-Qwen3-8B-Base-0728/I-2k-lora-rank128-lr1e-5-Shadow_2k" \
#  --target_base "Qwen/Qwen3-8B-Base" \
#  --merge_tag "I2B" \
#  --template "qwen3"

#python3 /home/ubuntu/Shadow/src/shadow/merge_lora.py \
#  --adapter_path "/home/ubuntu/Shadow/results/0728/result-Qwen3-8B-Base-0728/B-2k-lora-rank128-lr1e-5-Shadow_2k" \
#  --target_base "Qwen/Qwen3-8B-Base" \
#  --merge_tag "B2B" \
#  --template "qwen3"

python3 /home/ubuntu/Shadow/src/shadow/merge_lora.py \
  --adapter_path "/home/ubuntu/Shadow/results/0728/result-Qwen3-8B-Base-0728/B-2k-lora-rank128-lr2e-5-Shadow_2k" \
  --target_base "Qwen/Qwen3-8B" \
  --merge_tag "B2I" \
  --template "qwen3"

python3 /home/ubuntu/Shadow/src/shadow/merge_lora.py \
  --adapter_path "/home/ubuntu/Shadow/results/0728/result-Qwen3-8B-Base-0728/I-2k-lora-rank128-lr2e-5-Shadow_2k" \
  --target_base "Qwen/Qwen3-8B" \
  --merge_tag "I2I" \
  --template "qwen3"

#python3 /home/ubuntu/Shadow/src/shadow/merge_lora.py \
#  --adapter_path "/home/ubuntu/Shadow/results/0728/result-Qwen3-8B-Base-0728/I-2k-lora-rank128-lr2e-5-Shadow_2k" \
#  --target_base "Qwen/Qwen3-8B-Base" \
#  --merge_tag "I2B" \
#  --template "qwen3"

#python3 /home/ubuntu/Shadow/src/shadow/merge_lora.py \
#  --adapter_path "/home/ubuntu/Shadow/results/0728/result-Qwen3-8B-Base-0728/B-2k-lora-rank128-lr2e-5-Shadow_2k" \
#  --target_base "Qwen/Qwen3-8B-Base" \
#  --merge_tag "B2B" \
#  --template "qwen3"

python3 /home/ubuntu/Shadow/src/shadow/merge_lora.py \
  --adapter_path "/home/ubuntu/Shadow/results/0728/result-Qwen3-8B-Base-0728/B-2k-lora-rank128-lr5e-5-Shadow_2k" \
  --target_base "Qwen/Qwen3-8B" \
  --merge_tag "B2I" \
  --template "qwen3"

python3 /home/ubuntu/Shadow/src/shadow/merge_lora.py \
  --adapter_path "/home/ubuntu/Shadow/results/0728/result-Qwen3-8B-Base-0728/I-2k-lora-rank128-lr5e-5-Shadow_2k" \
  --target_base "Qwen/Qwen3-8B" \
  --merge_tag "I2I" \
  --template "qwen3"

#python3 /home/ubuntu/Shadow/src/shadow/merge_lora.py \
#  --adapter_path "/home/ubuntu/Shadow/results/0728/result-Qwen3-8B-Base-0728/I-2k-lora-rank128-lr5e-5-Shadow_2k" \
#  --target_base "Qwen/Qwen3-8B-Base" \
#  --merge_tag "I2B" \
#  --template "qwen3"

#python3 /home/ubuntu/Shadow/src/shadow/merge_lora.py \
#  --adapter_path "/home/ubuntu/Shadow/results/0728/result-Qwen3-8B-Base-0728/B-2k-lora-rank128-lr5e-5-Shadow_2k" \
#  --target_base "Qwen/Qwen3-8B-Base" \
#  --merge_tag "B2B" \
#  --template "qwen3"

python3 /home/ubuntu/Shadow/src/shadow/merge_lora.py \
  --adapter_path "/home/ubuntu/Shadow/results/0728/result-Qwen3-8B-Base-0728/B-2k-lora-rank128-lr1e-4-Shadow_2k" \
  --target_base "Qwen/Qwen3-8B" \
  --merge_tag "B2I" \
  --template "qwen3"

python3 /home/ubuntu/Shadow/src/shadow/merge_lora.py \
  --adapter_path "/home/ubuntu/Shadow/results/0728/result-Qwen3-8B-Base-0728/I-2k-lora-rank128-lr1e-4-Shadow_2k" \
  --target_base "Qwen/Qwen3-8B" \
  --merge_tag "I2I" \
  --template "qwen3"

#python3 /home/ubuntu/Shadow/src/shadow/merge_lora.py \
#  --adapter_path "/home/ubuntu/Shadow/results/0728/result-Qwen3-8B-Base-0728/I-2k-lora-rank128-lr1e-4-Shadow_2k" \
#  --target_base "Qwen/Qwen3-8B-Base" \
#  --merge_tag "I2B" \
#  --template "qwen3"

#python3 /home/ubuntu/Shadow/src/shadow/merge_lora.py \
#  --adapter_path "/home/ubuntu/Shadow/results/0728/result-Qwen3-8B-Base-0728/B-2k-lora-rank128-lr1e-4-Shadow_2k" \
#  --target_base "Qwen/Qwen3-8B-Base" \
#  --merge_tag "B2B" \
#  --template "qwen3"

python3 /home/ubuntu/Shadow/src/shadow/merge_lora.py \
  --adapter_path "/home/ubuntu/Shadow/results/0728/result-Qwen3-8B-Base-0728/B-2k-lora-rank128-lr2e-4-Shadow_2k" \
  --target_base "Qwen/Qwen3-8B" \
  --merge_tag "B2I" \
  --template "qwen3"

python3 /home/ubuntu/Shadow/src/shadow/merge_lora.py \
  --adapter_path "/home/ubuntu/Shadow/results/0728/result-Qwen3-8B-Base-0728/I-2k-lora-rank128-lr2e-4-Shadow_2k" \
  --target_base "Qwen/Qwen3-8B" \
  --merge_tag "I2I" \
  --template "qwen3"

#python3 /home/ubuntu/Shadow/src/shadow/merge_lora.py \
#  --adapter_path "/home/ubuntu/Shadow/results/0728/result-Qwen3-8B-Base-0728/I-2k-lora-rank128-lr2e-4-Shadow_2k" \
#  --target_base "Qwen/Qwen3-8B-Base" \
#  --merge_tag "I2B" \
#  --template "qwen3"

#python3 /home/ubuntu/Shadow/src/shadow/merge_lora.py \
#  --adapter_path "/home/ubuntu/Shadow/results/0728/result-Qwen3-8B-Base-0728/B-2k-lora-rank128-lr2e-4-Shadow_2k" \
#  --target_base "Qwen/Qwen3-8B-Base" \
#  --merge_tag "B2B" \
#  --template "qwen3"

python3 /home/ubuntu/Shadow/src/shadow/merge_lora.py \
  --adapter_path "/home/ubuntu/Shadow/results/0728/result-Qwen3-8B-Base-0728/B-2k-lora-rank128-lr5e-4-Shadow_2k" \
  --target_base "Qwen/Qwen3-8B" \
  --merge_tag "B2I" \
  --template "qwen3"

python3 /home/ubuntu/Shadow/src/shadow/merge_lora.py \
  --adapter_path "/home/ubuntu/Shadow/results/0728/result-Qwen3-8B-Base-0728/I-2k-lora-rank128-lr5e-4-Shadow_2k" \
  --target_base "Qwen/Qwen3-8B" \
  --merge_tag "I2I" \
  --template "qwen3"

#python3 /home/ubuntu/Shadow/src/shadow/merge_lora.py \
#  --adapter_path "/home/ubuntu/Shadow/results/0728/result-Qwen3-8B-Base-0728/I-2k-lora-rank128-lr5e-4-Shadow_2k" \
#  --target_base "Qwen/Qwen3-8B-Base" \
#  --merge_tag "I2B" \
#  --template "qwen3"

#python3 /home/ubuntu/Shadow/src/shadow/merge_lora.py \
#  --adapter_path "/home/ubuntu/Shadow/results/0728/result-Qwen3-8B-Base-0728/B-2k-lora-rank128-lr5e-4-Shadow_2k" \
#  --target_base "Qwen/Qwen3-8B-Base" \
#  --merge_tag "B2B" \
#  --template "qwen3"

python3 /home/ubuntu/Shadow/src/shadow/merge_lora.py \
  --adapter_path "/home/ubuntu/Shadow/results/0728/result-Qwen3-8B-Base-0728/B-2k-lora-rank128-lr1e-3-Shadow_2k" \
  --target_base "Qwen/Qwen3-8B" \
  --merge_tag "B2I" \
  --template "qwen3"

python3 /home/ubuntu/Shadow/src/shadow/merge_lora.py \
  --adapter_path "/home/ubuntu/Shadow/results/0728/result-Qwen3-8B-Base-0728/I-2k-lora-rank128-lr1e-3-Shadow_2k" \
  --target_base "Qwen/Qwen3-8B" \
  --merge_tag "I2I" \
  --template "qwen3"

#python3 /home/ubuntu/Shadow/src/shadow/merge_lora.py \
#  --adapter_path "/home/ubuntu/Shadow/results/0728/result-Qwen3-8B-Base-0728/I-2k-lora-rank128-lr1e-3-Shadow_2k" \
#  --target_base "Qwen/Qwen3-8B-Base" \
#  --merge_tag "I2B" \
#  --template "qwen3"

#python3 /home/ubuntu/Shadow/src/shadow/merge_lora.py \
#  --adapter_path "/home/ubuntu/Shadow/results/0728/result-Qwen3-8B-Base-0728/B-2k-lora-rank128-lr1e-3-Shadow_2k" \
#  --target_base "Qwen/Qwen3-8B-Base" \
#  --merge_tag "B2B" \
#  --template "qwen3"

python3 /home/ubuntu/Shadow/src/shadow/merge_lora.py \
  --adapter_path "/home/ubuntu/Shadow/results/0728/result-Qwen3-8B-Base-0728/B-2k-lora-rank128-lr2e-3-Shadow_2k" \
  --target_base "Qwen/Qwen3-8B" \
  --merge_tag "B2I" \
  --template "qwen3"

python3 /home/ubuntu/Shadow/src/shadow/merge_lora.py \
  --adapter_path "/home/ubuntu/Shadow/results/0728/result-Qwen3-8B-Base-0728/I-2k-lora-rank128-lr2e-3-Shadow_2k" \
  --target_base "Qwen/Qwen3-8B" \
  --merge_tag "I2I" \
  --template "qwen3"

#python3 /home/ubuntu/Shadow/src/shadow/merge_lora.py \
#  --adapter_path "/home/ubuntu/Shadow/results/0728/result-Qwen3-8B-Base-0728/I-2k-lora-rank128-lr2e-3-Shadow_2k" \
#  --target_base "Qwen/Qwen3-8B-Base" \
#  --merge_tag "I2B" \
#  --template "qwen3"

#python3 /home/ubuntu/Shadow/src/shadow/merge_lora.py \
#  --adapter_path "/home/ubuntu/Shadow/results/0728/result-Qwen3-8B-Base-0728/B-2k-lora-rank128-lr2e-3-Shadow_2k" \
#  --target_base "Qwen/Qwen3-8B-Base" \
#  --merge_tag "B2B" \
#  --template "qwen3"

##### Evaluation list #####
# ('0728/result-Qwen3-8B-Base-0728/merged-B2I','/home/ubuntu/Shadow/results/0728/result-Qwen3-8B-Base-0728/merged-B2I'),
# ('0728/result-Qwen3-8B-Base-0728/merged-I2I','/home/ubuntu/Shadow/results/0728/result-Qwen3-8B-Base-0728/merged-I2I'),
# ('0728/result-Qwen3-8B-Base-0728/merged-B2I','/home/ubuntu/Shadow/results/0728/result-Qwen3-8B-Base-0728/merged-B2I'),
# ('0728/result-Qwen3-8B-Base-0728/merged-I2I','/home/ubuntu/Shadow/results/0728/result-Qwen3-8B-Base-0728/merged-I2I'),
# ('0728/result-Qwen3-8B-Base-0728/merged-B2I','/home/ubuntu/Shadow/results/0728/result-Qwen3-8B-Base-0728/merged-B2I'),
# ('0728/result-Qwen3-8B-Base-0728/merged-I2I','/home/ubuntu/Shadow/results/0728/result-Qwen3-8B-Base-0728/merged-I2I'),
# ('0728/result-Qwen3-8B-Base-0728/merged-B2I','/home/ubuntu/Shadow/results/0728/result-Qwen3-8B-Base-0728/merged-B2I'),
# ('0728/result-Qwen3-8B-Base-0728/merged-I2I','/home/ubuntu/Shadow/results/0728/result-Qwen3-8B-Base-0728/merged-I2I'),
# ('0728/result-Qwen3-8B-Base-0728/merged-B2I','/home/ubuntu/Shadow/results/0728/result-Qwen3-8B-Base-0728/merged-B2I'),
# ('0728/result-Qwen3-8B-Base-0728/merged-I2I','/home/ubuntu/Shadow/results/0728/result-Qwen3-8B-Base-0728/merged-I2I'),
# ('0728/result-Qwen3-8B-Base-0728/merged-B2I','/home/ubuntu/Shadow/results/0728/result-Qwen3-8B-Base-0728/merged-B2I'),
# ('0728/result-Qwen3-8B-Base-0728/merged-I2I','/home/ubuntu/Shadow/results/0728/result-Qwen3-8B-Base-0728/merged-I2I'),
# ('0728/result-Qwen3-8B-Base-0728/merged-B2I','/home/ubuntu/Shadow/results/0728/result-Qwen3-8B-Base-0728/merged-B2I'),
# ('0728/result-Qwen3-8B-Base-0728/merged-I2I','/home/ubuntu/Shadow/results/0728/result-Qwen3-8B-Base-0728/merged-I2I'),
# ('0728/result-Qwen3-8B-Base-0728/merged-B2I','/home/ubuntu/Shadow/results/0728/result-Qwen3-8B-Base-0728/merged-B2I'),
# ('0728/result-Qwen3-8B-Base-0728/merged-I2I','/home/ubuntu/Shadow/results/0728/result-Qwen3-8B-Base-0728/merged-I2I'),
# ('0728/result-Qwen3-8B-Base-0728/merged-B2I','/home/ubuntu/Shadow/results/0728/result-Qwen3-8B-Base-0728/merged-B2I'),
# ('0728/result-Qwen3-8B-Base-0728/merged-I2I','/home/ubuntu/Shadow/results/0728/result-Qwen3-8B-Base-0728/merged-I2I'),

# please copy this eval_config to opencompass/examples/eval_shadow_202505.py and then run

cd ./opencompass
python3 ./run.py ./examples/eval_shadow_202505.py

