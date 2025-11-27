import json
from pathlib import Path

path = Path("shadow_medical_mix_sharegpt_4k.json")

with path.open("r", encoding="utf-8") as f:
    data = json.load(f)

def normalize_msg(msg):
    # 已经是 role/content 的，直接返回
    if "role" in msg and "content" in msg:
        return msg

    # 旧格式: from/value
    if "from" in msg and "value" in msg:
        role = msg["from"]
        if role == "human":
            role = "user"
        elif role == "gpt":
            role = "assistant"
        # 其他角色按原样保留
        return {
            "role": role,
            "content": msg["value"],
        }

    # 其他奇怪情况：原样返回，方便你后续排查
    return msg

for ex in data:
    conv = ex.get("conversations", [])
    ex["conversations"] = [normalize_msg(m) for m in conv]

# 覆盖写回 / 或者你想写成新文件也可以
out_path = Path("shadow_medical_mix_sharegpt_4k.normalized.json")
with out_path.open("w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

print(f"Normalized and saved to {out_path}, total samples: {len(data)}")

