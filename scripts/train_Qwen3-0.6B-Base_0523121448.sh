##### Auto-generated 2025-05-23 12:14:48 #####
# Model     : Qwen3-0.6B-Base
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
##### LoRA delta-merge #####
##### Evaluation list #####

# please copy this 'Evaluation list' to opencompass/examples/eval_shadow_202505.py models part and then run

cd ./opencompass
python3 ./run.py ./examples/eval_shadow_202505.py
