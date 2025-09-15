# # Convert Parquet to JSON (AIME)
# import pandas as pd

# # Specify the Parquet file path
# # link: https://huggingface.co/datasets/AI-MO/aimo-validation-aime
# parquet_file = "/home/ubuntu/Shadow/opencompass/data/train-00000-of-00001.parquet"

# # Use pandas to read the Parquet file
# df = pd.read_parquet(parquet_file)

# # Filter the DataFrame to keep only rows where '2024_AIME' appears in the 'url' column
# filtered_df = df[df['url'].str.contains('2024_AIME', na=False)]

# # Print the first few rows of the filtered DataFrame to confirm
# print(filtered_df.head())

# # Export to a JSON file with indentation
# json_file = "/home/ubuntu/Shadow/opencompass/data/aime_2024.json"
# filtered_df.to_json(json_file, orient='records', force_ascii=False, indent=4)

# print(f"Filtered data has been saved to {json_file}")



import pandas as pd
import json
import os
import numpy as np
import glob
from tqdm import tqdm

# 用于处理JSON序列化NumPy数组的函数
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, (np.bool_)):
            return bool(obj)
        if isinstance(obj, (bytes, bytearray)):
            return obj.decode('utf-8')
        return super(NumpyEncoder, self).default(obj)

def convert_parquet_to_jsonl(parquet_path, output_dir=None):
    """
    将Parquet文件转换为JSONL格式并保存
    
    Args:
        parquet_path: Parquet文件路径
        output_dir: 输出目录，若为None则在数据集目录下创建jsonl目录
    """
    # 获取目录和不带扩展名的文件名
    dir_path = os.path.dirname(parquet_path)
    file_base = os.path.basename(parquet_path).split('.')[0]
    
    # 如果未指定输出目录，则在数据集目录下创建jsonl目录
    if output_dir is None:
        output_dir = os.path.join(dir_path, "jsonl")
        
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 构建输出路径
    jsonl_path = os.path.join(output_dir, f"{file_base}.jsonl")
    
    print(f"Converting {parquet_path} to {jsonl_path}")
    
    try:
        # 读取Parquet文件
        df = pd.read_parquet(parquet_path)
        
        # 写入JSONL
        with open(jsonl_path, 'w', encoding='utf-8') as f:
            for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing rows"):
                # 使用自定义编码器处理NumPy类型
                json_str = json.dumps(row.to_dict(), ensure_ascii=False, cls=NumpyEncoder)
                f.write(json_str + '\n')
        
        print(f"Conversion complete. JSONL file saved at: {jsonl_path}")
        print(f"Total records processed: {len(df)}")
        return True
        
    except Exception as e:
        print(f"Error converting {parquet_path}: {e}")
        return False

def process_directory(base_dir, pattern="**/*.parquet"):
    """
    处理目录中所有符合模式的Parquet文件
    
    Args:
        base_dir: 基础目录
        pattern: glob模式，默认查找所有.parquet文件
    """
    # 查找所有Parquet文件
    parquet_files = glob.glob(os.path.join(base_dir, pattern), recursive=True)
    
    # 按数据集分组
    dataset_groups = {}
    for file_path in parquet_files:
        # 获取数据集目录路径
        dataset_dir = os.path.dirname(file_path)
        if dataset_dir not in dataset_groups:
            dataset_groups[dataset_dir] = []
        dataset_groups[dataset_dir].append(file_path)
    
    # 处理每个数据集
    success_count = 0
    fail_count = 0
    
    for dataset_dir, files in dataset_groups.items():
        # 为每个数据集创建jsonl子目录
        output_dir = os.path.join(dataset_dir, "jsonl")
        
        print(f"\nProcessing dataset: {dataset_dir}")
        print(f"Found {len(files)} Parquet files")
        
        # 处理每个Parquet文件
        for file_path in files:
            if convert_parquet_to_jsonl(file_path, output_dir):
                success_count += 1
            else:
                fail_count += 1
                
    print(f"\nSummary:")
    print(f"Successfully converted: {success_count} files")
    print(f"Failed: {fail_count} files")
    
    # 创建或更新配置文件示例
    print("\nConfiguration example for LLaMA-Factory:")
    print("dataset:")
    for dataset_dir in dataset_groups:
        rel_path = os.path.relpath(os.path.join(dataset_dir, "jsonl"))
        print(f"  - path: {rel_path}")
        print(f"    name: default")
        print(f"    type: json")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Convert Parquet files to JSONL format")
    parser.add_argument("--base_dir", type=str, 
                        default="/apdcephfs_qy3/share_301069248/users/rummyyang/open-instruct/lm-evaluation-harness/lm-evaluation-harness/dataset/mbppplus/data",
                        help="Base directory to search for Parquet files")
    parser.add_argument("--pattern", type=str, default="**/*.parquet", 
                        help="Glob pattern to find Parquet files")
    
    args = parser.parse_args()
    
    process_directory(args.base_dir, args.pattern)