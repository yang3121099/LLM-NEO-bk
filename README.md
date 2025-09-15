

## 0729实验脚本



```
# 环境安装
cd /root
mkdir -p /root/shadow_exp/new_shadow
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
eval "$(/root/miniconda3/bin/conda shell.bash hook)"

conda create -n factory python=3.10 -y
conda activate factory

cd /root/shadow_exp/new_shadow
git clone https://github.com/yang3121099/Shadow-FT-bk
mv Shadow-FT-bk Shadow
cd Shadow
pip install -e ".[torch,metrics]"
pip install importlib_metadata omegaconf
pip install torch==2.6.0 transformers==4.52.1 torchvision  deepspeed -U
cd ./opencompass
# pip install -U opencompass
pip install -e .
export COMPASS_DATA_CACHE="/root/shadow_exp/new_shadow/opencompass" 

pip install lmdeploy evalplus==0.3.1 latex2sympy2_extended math_verify prettytable jieba rouge_chinese rank_bm25 gradio_client tree_sitter_languages  fuzzywuzzy  h5py peft==0.15.2
git clone git@github.com:open-compass/human-eval.git #安装HumanEval
cd human-eval && pip install -e .
pip install git+https://github.com/EleutherAI/lm-evaluation-harness.git


#以下正文
git pull origin main #更新代码

conda activate factory
pip install aqlm[gpu,cpu]

cd /root/shadow_exp/new_shadow/Shadow
bash ./run.sh

bash ./scripts/train_Llama-3.2-1B_0729110642.sh
bash ./scripts/train_Llama-3.1-8B_0729110023.sh


cd ./opencompass
pip install huggingface_hub
huggingface-cli login


python3 ./run.py ./eval_shadow_202505.py -r 20250727200010 # 请与文件内的参考指标 校准后再继续


python3 ./run.py ./eval_shadow_20250729.py -r 20250727200010 
python /root/shadow_exp/new_shadow/Shadow/upload_hf.py #自动上传hf  请重新传入hf_ token


#若有上次的ckpt，则执行下面的内容part1
python3 ./run.py ./eval_shadow_20250727_part1.py -r 20250727200010 
python /root/shadow_exp/new_shadow/Shadow/upload_hf.py #自动上传hf

# #如果没有上次的ckpt，则执行下面这组part3
# bash ./scripts/train_Qwen3-8B-Base_0730042234.sh
# python3 ./run.py ./eval_shadow_20250727_part3.py -r 20250727200010  
# python /root/shadow_exp/new_shadow/Shadow/upload_hf.py #自动上传hf

python3 ./run.py ./eval_shadow_20250727_part2.py -r 20250727200010 
python /root/shadow_exp/new_shadow/Shadow/upload_hf.py #自动上传hf





```




## 0728实验脚本




```
pip install huggingface_hub
huggingface-cli login

git pull origin main #更新代码

cd /root/shadow_exp/new_shadow/Shadow
cd ./src
bash ./copy_files.sh /root/miniconda3/envs/factory/lib/python3.10/site-packages
cd ../opencompass


python3 ./run.py ./eval_shadow_202505.py -r 20250727200010 # 请与文件内的参考指标 校准后再继续
python /root/shadow_exp/new_shadow/Shadow/upload_hf.py #自动上传hf



bash ./eval_instruct_0427.sh  ./eval_shadow_20250727_part1.py
python /root/shadow_exp/new_shadow/Shadow/upload_hf.py #自动上传hf

bash ./eval_instruct_0427.sh  ./eval_shadow_20250727.py
python /root/shadow_exp/new_shadow/Shadow/upload_hf.py #自动上传hf

bash ./eval_instruct_0427.sh  ./eval_quantw_20250727.py
python /root/shadow_exp/new_shadow/Shadow/upload_hf.py #自动上传hf

bash ./eval_instruct_0427.sh  ./eval_shadow_20250727_part2.py
python /root/shadow_exp/new_shadow/Shadow/upload_hf.py #自动上传hf
 

```


## 0727实验脚本


```
# 创建环境
conda create -n factory python=3.10 -y
conda activate factory

# 复制本仓库
git clone https://github.com/yang3121099/Shadow-FT-bk
mv Shadow-FT-bk Shadow
cd Shadow
pip install -e ".[torch,metrics]"
pip install importlib_metadata omegaconf
pip install torch==2.6.0 transformers==4.52.1 torchvision  deepspeed -U
cd ./opencompass
pip install -U opencompass
pip install -e .
export COMPASS_DATA_CACHE="." # 若报错则改为绝对路径的 "YOURS/opencompass/data"
# wget https://github.com/open-compass/opencompass/releases/download/0.2.2.rc1/OpenCompassData-core-20240207.zip
# python3 -m zipfile -e  OpenCompassData-core-20240207.zip  ./data

pip install lmdeploy evalplus==0.3.1 latex2sympy2_extended math_verify prettytable jieba rouge_chinese rank_bm25 gradio_client tree_sitter_languages  fuzzywuzzy  h5py peft==0.15.2
git clone github.com/open-compass/human-eval
cd human-eval && pip install -e .
pip install git+https://github.com/EleutherAI/lm-evaluation-harness.git
```

# 生成训练代码

```
cd ../../../Shadow #或绝对路径
bash run.sh

# 打开新创建的sh文件，全选粘贴到命令行即可
```



# 执行测评代码

```
# git add . && git commit -m "update $(date +%Y-%m-%d)" && git push origin main
git pull origin main #更新代码

pip install "opencompass[lmdeploy]"

cd ./src
bash ./copy_files.sh /YOURENV/miniconda3/envs/factory/lib/python3.10/site-packages

export COMPASS_DATA_CACHE="YOURS/opencompass" # 若报错则改为绝对路径


python3 ./run.py ./eval_shadow_202505.py -r 20250727200010 # 校准 确认batch大小


python3 ./run.py ./eval_shadow_20250727_part1.py -r 20250727200010
python3 ./run.py ./eval_shadow_20250727_part2.py -r 20250727200010
python3 ./run.py ./eval_quantw_20250727.py -r 20250727200011
python3 ./run.py ./eval_shadow_20250727.py -r 20250727200013


cd ../
pip install huggingface_hub #上传到hf
python upload_hf.py



```

以下是Readme，师兄无需阅读


# Shadow-FT

Official repository for the paper **Shadow-FT: Tuning Instruct via Base**.

Shadow-FT fine‑tunes a *Base* language model with LoRA to obtain lightweight Δ (delta) parameters, then **merges** them onto its *Instruct* counterpart to boost instruction following. Training is powered by **LLaMA‑Factory**, evaluation by **OpenCompass**.

---

## Install

```bash
# Clone the repo
git clone https://github.com/wutaiqiang/Shadow-FT && cd Shadow

# Core dependencies (inherited from LLaMA‑Factory)
pip install -e ".[torch]" --no-build-isolation
```

---

## One‑click workflow

All operations are wrapped in **./run.sh**.  

---

## Sample `run.sh` log

```text
##### Auto-generated 2025-05-22 13:54:08 #####
# Model     : Qwen2.5-14B
# LoRA mode : true
# Template  : qwen

##### Environment #####
export VLLM_WORKER_MULTIPROC_METHOD=spawn

##### Training #####
###### I  max=2000  lr=1e-5 ######
llamafactory-cli train \
  --model_name_or_path "${MODEL_ROOT}/Qwen2.5-14B-Instruct" \
  --finetuning_type lora --lora_rank 128 \
  --dataset "Shadow_2k" \
  --output_dir "${OUTPUT_ROOT}/instruct_lora" ...

##### LoRA delta‑merge #####
llamafactory-cli export \
  --base_model "${MODEL_ROOT}/Qwen2.5-14B-Instruct" \
  --lora_dir   "${OUTPUT_ROOT}/delta" \
  --output_dir "${OUTPUT_ROOT}/shadow_instruct"

##### Evaluation list #####
# ('short_name', 'model_path')
```

*(The log is auto‑generated by ****\`\`****; edit the script to match your paths and GPUs.)*

---

## Cite

```bibtex
@article{shadowft2025,
  title  = {Shadow-FT: Tuning Instruct via Base},
  author = {Wu et al.},
  year   = {2025},
  eprint = {2505.12716},
  archivePrefix = {arXiv}
}
```

## License

Apache‑2.0.  Please also comply with the licenses of any upstream models and datasets.