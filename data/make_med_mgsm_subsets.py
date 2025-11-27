import json
import random
from pathlib import Path

from datasets import load_dataset  # pip install datasets

random.seed(42)

ROOT = Path(".")

# =========================
# 1. medical_o1_reasoning 抽样 2k
# =========================
print("Loading medical_o1_reasoning (HF)...")
ds_med = load_dataset(
    "FreedomIntelligence/medical-o1-reasoning-SFT",
    "en",
    split="train"
)

n_med = len(ds_med)
k_med = min(2000, n_med)
idx_med = random.sample(range(n_med), k_med)

med_2k = []
for i in idx_med:
    ex = ds_med[i]
    # 保留原始字段名：Question / Response
    med_2k.append(
        {
            "Question": ex["Question"],
            "Response": ex["Response"],
        }
    )

med_2k_path = ROOT / "medical_o1_reasoning_2k.json"
with med_2k_path.open("w", encoding="utf-8") as f:
    json.dump(med_2k, f, ensure_ascii=False, indent=2)

print(f"Saved {len(med_2k)} examples to {med_2k_path}")

# =========================
# 2. mgsm 抽样 2k
# =========================

mgsm_path = ROOT / "mgsm.json"
print(f"Loading {mgsm_path} ...")
with mgsm_path.open("r", encoding="utf-8") as f:
    mgsm_all = json.load(f)

n_mgsm = len(mgsm_all)
k_mgsm = min(2000, n_mgsm)
idx_mgsm = random.sample(range(n_mgsm), k_mgsm)

mgsm_2k = [mgsm_all[i] for i in idx_mgsm]

mgsm_2k_path = ROOT / "mgsm_2k.json"
with mgsm_2k_path.open("w", encoding="utf-8") as f:
    json.dump(mgsm_2k, f, ensure_ascii=False, indent=2)

print(f"Saved {len(mgsm_2k)} examples to {mgsm_2k_path}")

# =========================
# 3. 构造 4k 混合数据集
#    统一成 prompt/response 字段
# =========================

mix = []

for ex in med_2k:
    mix.append(
        {
            "prompt": ex["Question"],
            "response": ex["Response"],
            "source": "medical_o1_reasoning_en",
        }
    )

for ex in mgsm_2k:
    mix.append(
        {
            "prompt": ex["question"],
            "response": ex["answer"],
            "source": "mgsm",
        }
    )

random.shuffle(mix)

mix_path = ROOT / "medical_mgsm_mix_4k.json"
with mix_path.open("w", encoding="utf-8") as f:
    json.dump(mix, f, ensure_ascii=False, indent=2)

print(f"Saved {len(mix)} examples to {mix_path}")
print("Done.")

