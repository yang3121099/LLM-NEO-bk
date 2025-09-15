##### Auto-generated 2025-07-29 11:06:42 #####
# Model     : Llama-3.1-8B
# LoRA mode : true
# Template  : llama3

##### Environment #####
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export HF_HUB_OFFLINE=0
export HF_DATASETS_OFFLINE=0
export HF_DATASETS_TRUST_REMOTE_CODE=1
export TRUST_REMOTE_CODE=True
export HF_ALLOW_CODE_EVAL=1

##### Training #####
###### B  max=2000  lr=0.0002 ######
mkdir -p "/home/ubuntu/Shadow/results/0729/result-Llama-3.1-8B-0729/B-2k-lora-rank128-lr0.0002-Shadow_2k"
cd "/home/ubuntu/Shadow"
llamafactory-cli train \
  --model_name_or_path "meta-llama/Llama-3.1-8B" \
  --stage sft \
  --do_train true \
  --finetuning_type lora --lora_rank 128 \
  --deepspeed examples/deepspeed/ds_z3_config.json \
  --dataset "Shadow_2k" \
  --template "llama3" \
  --cutoff_len 4096 \
  --max_samples 2000 \
  --output_dir "/home/ubuntu/Shadow/results/0729/result-Llama-3.1-8B-0729/B-2k-lora-rank128-lr0.0002-Shadow_2k" \
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

###### I  max=2000  lr=0.0002 ######
mkdir -p "/home/ubuntu/Shadow/results/0729/result-Llama-3.1-8B-0729/I-2k-lora-rank128-lr0.0002-Shadow_2k"
cd "/home/ubuntu/Shadow"
llamafactory-cli train \
  --model_name_or_path "meta-llama/Llama-3.1-8B-Instruct" \
  --stage sft \
  --do_train true \
  --finetuning_type lora --lora_rank 128 \
  --deepspeed examples/deepspeed/ds_z3_config.json \
  --dataset "Shadow_2k" \
  --template "llama3" \
  --cutoff_len 4096 \
  --max_samples 2000 \
  --output_dir "/home/ubuntu/Shadow/results/0729/result-Llama-3.1-8B-0729/I-2k-lora-rank128-lr0.0002-Shadow_2k" \
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
mkdir -p "/home/ubuntu/Shadow/results/0729/result-Llama-3.1-8B-0729/B-2k-lora-rank128-lr0.0002-Shadow_2k/merge-B2I"
python3 /home/ubuntu/Shadow/src/shadow/merge_lora.py \
  --adapter_path "/home/ubuntu/Shadow/results/0729/result-Llama-3.1-8B-0729/B-2k-lora-rank128-lr0.0002-Shadow_2k" \
  --target_base "meta-llama/Llama-3.1-8B-Instruct" \
  --merge_tag "B2I" \
  --template "llama3"

mkdir -p "/home/ubuntu/Shadow/results/0729/result-Llama-3.1-8B-0729/I-2k-lora-rank128-lr0.0002-Shadow_2k/merge-I2I"
python3 /home/ubuntu/Shadow/src/shadow/merge_lora.py \
  --adapter_path "/home/ubuntu/Shadow/results/0729/result-Llama-3.1-8B-0729/I-2k-lora-rank128-lr0.0002-Shadow_2k" \
  --target_base "meta-llama/Llama-3.1-8B-Instruct" \
  --merge_tag "I2I" \
  --template "llama3"

mkdir -p "/home/ubuntu/Shadow/results/0729/result-Llama-3.1-8B-0729/I-2k-lora-rank128-lr0.0002-Shadow_2k/merge-I2B"
python3 /home/ubuntu/Shadow/src/shadow/merge_lora.py \
  --adapter_path "/home/ubuntu/Shadow/results/0729/result-Llama-3.1-8B-0729/I-2k-lora-rank128-lr0.0002-Shadow_2k" \
  --target_base "meta-llama/Llama-3.1-8B" \
  --merge_tag "I2B" \
  --template "llama3"

mkdir -p "/home/ubuntu/Shadow/results/0729/result-Llama-3.1-8B-0729/B-2k-lora-rank128-lr0.0002-Shadow_2k/merge-B2B"
python3 /home/ubuntu/Shadow/src/shadow/merge_lora.py \
  --adapter_path "/home/ubuntu/Shadow/results/0729/result-Llama-3.1-8B-0729/B-2k-lora-rank128-lr0.0002-Shadow_2k" \
  --target_base "meta-llama/Llama-3.1-8B" \
  --merge_tag "B2B" \
  --template "llama3"

##### Evaluation list #####
# ('/home/ubuntu/Shadow/results/0729/result-Llama-3.1-8B-0729/B-2k-lora-rank128-lr0.0002-Shadow_2k/merge-B2I','/home/ubuntu/Shadow/results/0729/result-Llama-3.1-8B-0729/B-2k-lora-rank128-lr0.0002-Shadow_2k/merge-B2I'),
# ('/home/ubuntu/Shadow/results/0729/result-Llama-3.1-8B-0729/I-2k-lora-rank128-lr0.0002-Shadow_2k/merge-I2I','/home/ubuntu/Shadow/results/0729/result-Llama-3.1-8B-0729/I-2k-lora-rank128-lr0.0002-Shadow_2k/merge-I2I'),
# ('/home/ubuntu/Shadow/results/0729/result-Llama-3.1-8B-0729/I-2k-lora-rank128-lr0.0002-Shadow_2k/merge-I2B','/home/ubuntu/Shadow/results/0729/result-Llama-3.1-8B-0729/I-2k-lora-rank128-lr0.0002-Shadow_2k/merge-I2B'),
# ('/home/ubuntu/Shadow/results/0729/result-Llama-3.1-8B-0729/B-2k-lora-rank128-lr0.0002-Shadow_2k/merge-B2B','/home/ubuntu/Shadow/results/0729/result-Llama-3.1-8B-0729/B-2k-lora-rank128-lr0.0002-Shadow_2k/merge-B2B'),

# please copy this eval_config to opencompass/examples/eval_shadow_202505.py and then run

cd ./opencompass
python3 ./run.py ./examples/eval_shadow_202505.py

