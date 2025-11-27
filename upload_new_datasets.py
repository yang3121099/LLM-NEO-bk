
import os
from huggingface_hub import HfApi, HfFolder, upload_folder

# 从环境变量中读取 HF Token
HF_TOKEN = os.environ.get("HF_TOKEN")
if not HF_TOKEN:
    raise ValueError("环境变量 HF_TOKEN 未设置，请先运行：export HF_TOKEN=''")

HfFolder.save_token(HF_TOKEN)
api = HfApi()

# ✅ 数据集文件夹路径
dataset_paths = [
       "/dockerdata/LLM-NEO-bk/opencompass/outputs/Subjective-Evaluation/shadow-comprehensive/", 
        ]
        # ✅ HF 用户名或组织名
HF_USERNAME = "yang31210999"

for path in dataset_paths:
    dataset_name = "ICLR-1127_BenchMark-H20-d6"
    repo_id = f"{HF_USERNAME}/{dataset_name}"
    print(f"🚀 正在上传数据集 {dataset_name} 到 {repo_id} ...")

    # ✅ 如果仓库不存在，则创建
    try:
        api.create_repo(repo_id, repo_type="dataset", private=False)
    except Exception as e:
        print(f"⚠️ 数据集仓库 {repo_id} 已存在，跳过创建")

    # ✅ 上传整个数据集目录
    upload_folder(
        folder_path=path,
        repo_id=repo_id,
        repo_type="dataset",
        commit_message="Upload dataset"
    )

print("✅ 全部数据集上传完成！")
