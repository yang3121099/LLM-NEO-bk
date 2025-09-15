
import os
import json

# 原始列式 JSON 路径
input_root = "/home/ubuntu/.cache/huggingface/hub/models--yang31210999--gptoss-0807_BenchMark-H100-d1/snapshots/4fa933c4c2dd4c61e92a84e7ca01d4f76ace0320/data/predictions"

# 行式输出的根目录
output_root = os.path.join(os.path.dirname(input_root), "predictions-vert")

# 遍历所有 json 文件
for dirpath, _, filenames in os.walk(input_root):
    for filename in filenames:
        if filename.endswith(".json"):
            input_path = os.path.join(dirpath, filename)

            # 构造对应输出路径
            relative_path = os.path.relpath(input_path, input_root)
            output_path = os.path.join(output_root, relative_path)
            output_path = output_path.replace(".json", ".jsonl")

            # 确保输出子目录存在
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            print(f"🔄 Converting: {relative_path}")

            try:
                with open(input_path, "r") as f:
                    data = json.load(f)

                with open(output_path, "w") as out_f:
                    for key, value in data.items():
                        out_f.write(json.dumps({
                            "id": key,
                            "prompt": key,
                            "response": value
                        }) + "\n")

                print(f"✅ Saved to: {output_path}")

            except Exception as e:
                print(f"❌ Error processing {input_path}: {e}")

print("🎉 All JSON files converted to vertical format.")
