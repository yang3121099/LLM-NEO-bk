import json
import random
from pathlib import Path

import pandas as pd
import numpy as np  # 新增：用于类型判断

random.seed(42)

ROOT = Path(".")

K_SHADOW = 2000
K_MED = 2000

# =========================================================
# 1. 读取 Shadow_2k.parquet（ShareGPT 格式）
# =========================================================
shadow_path = ROOT / "Shadow_2k.parquet"
print(f"Loading {shadow_path} ...")
df_shadow = pd.read_parquet(shadow_path)

if "conversations" not in df_shadow.columns:
    raise ValueError("Shadow_2k.parquet 中找不到列 `conversations`")

n_shadow = len(df_shadow)
k_shadow = min(K_SHADOW, n_shadow)
print(f"[Shadow] loaded {n_shadow}, sampling {k_shadow}")

idx_shadow = random.sample(range(n_shadow), k_shadow)

shadow_samples = []
for i in idx_shadow:
    conv = df_shadow.iloc[i]["conversations"]

    # 关键修复：把 numpy.ndarray 转成普通 list
    if isinstance(conv, np.ndarray):
        conv = conv.tolist()

    # 如果 conv 是 pandas 的 object 里套了 ndarray/list，也转成 list
    # 比如 conv 是 list-like 但不是纯 list
    if not isinstance(conv, list):
        try:
            conv = list(conv)
        except TypeError:
            raise TypeError(f"conversations 第 {i} 行类型 {type(conv)} 无法转换为 list，请检查数据格式。")

    shadow_samples.append(
        {
            "conversations": conv,   # 已经是 Python list，可以安全写入 JSON
            "source": "Shadow_2k",
        }
    )

shadow_out = ROOT / "Shadow_2k_sharegpt_2k.json"
with shadow_out.open("w", encoding="utf-8") as f:
    json.dump(shadow_samples, f, ensure_ascii=False, indent=2)

print(f"Saved Shadow subset: {len(shadow_samples)} → {shadow_out}")


# =========================================================
# 2. 读取 medical_o1_reasoning_2k.json（本地）
#    格式: {"Question": "...", "Response": "..."}
#    转换 → ShareGPT 格式
# =========================================================
med_path = ROOT / "medical_o1_reasoning_2k.json"
print(f"Loading {med_path} ...")

with med_path.open("r", encoding="utf-8") as f:
    med_raw = json.load(f)

n_med = len(med_raw)
k_med = min(K_MED, n_med)

print(f"[Medical] loaded {n_med}, sampling {k_med}")

idx_med = random.sample(range(n_med), k_med)

med_samples = []
for i in idx_med:
    ex = med_raw[i]
    q = ex["Question"]
    a = ex["Response"]

    conv = [
        {"role": "user", "content": q},
        {"role": "assistant", "content": a},
    ]

    med_samples.append(
        {
            "conversations": conv,
            "source": "medical_o1_reasoning",
        }
    )

med_out = ROOT / "medical_o1_reasoning_sharegpt_2k.json"
with med_out.open("w", encoding="utf-8") as f:
    json.dump(med_samples, f, ensure_ascii=False, indent=2)

print(f"Saved Medical subset: {len(med_samples)} → {med_out}")


# =========================================================
# 3. 合并、混洗
# =========================================================
mix = shadow_samples + med_samples
random.shuffle(mix)

mix_out = ROOT / "shadow_medical_mix_sharegpt_4k.json"
with mix_out.open("w", encoding="utf-8") as f:
    json.dump(mix, f, ensure_ascii=False, indent=2)

print(f"Saved MIX dataset: {len(mix)} → {mix_out}")

# =========================================================
# 4. 打印最终数量
# =========================================================
print("\n================== RESULT ==================")
print(f"Shadow subset size:   {len(shadow_samples)}")
print(f"Medical subset size:  {len(med_samples)}")
print(f"Final MIX size:       {len(mix)}")
print("============================================\n")

