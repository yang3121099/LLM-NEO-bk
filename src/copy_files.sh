#!/bin/bash
# 用法: bash copy_files.sh /home/ubuntu/miniconda3/envs/factory/lib/python3.10/site-packages

TARGET_PATH=$1

if [ -z "$TARGET_PATH" ]; then
    echo "❌ 请输入目标路径，例如:"
    echo "   bash copy_files.sh /root/miniconda3/envs/factory/lib/python3.10/site-packages"
    exit 1
fi

# ✅ 依次复制文件（这里使用相对路径）
cp ./mbpp.py   $TARGET_PATH/opencompass/datasets/mbpp.py
cp ./humaneval.py   $TARGET_PATH/opencompass/datasets/humaneval.py
# cp ./evaluate.py    $TARGET_PATH/evalplus/evaluate.py
# cp ./utils.py       $TARGET_PATH/evalplus/data/utils.py
cp ./turbomind_with_tf_above_v4_33.py   $TARGET_PATH/opencompass/models/turbomind_with_tf_above_v4_33.py

cp ./mbpp.py   /home/ubuntu/Shadow/opencompass/opencompass/datasets/mbpp.py
cp ./humaneval.py   /home/ubuntu/Shadow/opencompass/opencompass/datasets/humaneval.py
cp ./turbomind_with_tf_above_v4_33.py   /home/ubuntu/Shadow/opencompass/opencompass/models/turbomind_with_tf_above_v4_33.py
# cp ./utils.py       /home/ubuntu/Shadow/evalplus/data/utils.py
# cp ./turbomind_with_tf_above_v4_33.py   /home/ubuntu/Shadow/opencompass/models/turbomind_with_tf_above_v4_33.py

echo "✅ 文件已复制到: $TARGET_PATH"
