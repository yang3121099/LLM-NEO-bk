#!/bin/bash

# 配置路径
WORK_ROOT="/dockerdata/LLM-NEO-bk"
SCRIPT_PY="$WORK_ROOT/src/gen_analysis_loras.py"
MERGE_PY="$WORK_ROOT/src/shadow/merge_lora.py"
# 所有的输出都在这里 (由 Python 脚本定义)
ANALYSIS_DIR="$WORK_ROOT/results/1123/result-Qwen3-8B-Base-1123/analysis_suite_1124"
TEMPLATE="qwen3"

# 1. 运行 Python 生成 LoRA 文件 (处理 CodeZ1 和 Shadow2k)
echo ">>> Step 1: Generating Analysis LoRAs (CodeZ1 & Shadow2k)..."
python3 "$SCRIPT_PY"

# 2. 定义前缀和子类型，用于双重循环生成列表
PREFIXES=("CodeZ1" "Shadow2k")
SUB_TYPES=(
    "Single_ModelA_AttnOnly"
    "Single_ModelA_MlpOnly"
    "Single_ModelA_Shallow"
    "Single_ModelA_Deep"
    "Single_ModelB_AttnOnly"
    "Single_ModelB_MlpOnly"
    "Single_ModelB_Shallow"
    "Single_ModelB_Deep"
    "Hybrid_Attn-A_Mlp-B"
    "Hybrid_Mlp-A_Attn-B"
    "Hybrid_Shallow-A_Deep-B"
    "Hybrid_Deep-A_Shallow-B"
)

OUTPUT_SCRIPT="run_merge_eval.sh"
echo "# Auto-generated Merge Script" > $OUTPUT_SCRIPT

# 3. 生成 Merge 命令
echo ">>> Step 2: Generating Merge Commands..."

for PREFIX in "${PREFIXES[@]}"; do
    for TYPE in "${SUB_TYPES[@]}"; do
        # 组合出真实的文件夹名称，例如 CodeZ1_Single_ModelA_AttnOnly
        FOLDER_NAME="${PREFIX}_${TYPE}"
        ADAPTER_PATH="$ANALYSIS_DIR/$FOLDER_NAME"
        
        # 这里的 merged 只是一个父目录，脚本通常会在里面建 merged-<TAG>
        # 但我们先建好父目录以免报错
        MERGED_PARENT="$ADAPTER_PATH/merged" 
        
        # Merge Tag 直接用文件夹名，保证唯一性
        TAG="${FOLDER_NAME}"
        
        {
            echo "echo \"Processing: $FOLDER_NAME\""
            # 注意：实际输出通常是 $ADAPTER_PATH/merged-$TAG
            # 这里 mkdir merged 只是为了保险，或者如果脚本只用 merged 目录
            echo "mkdir -p \"$MERGED_PARENT\"" 
            echo "python3 $MERGE_PY \\"
            echo "  --adapter_path \"$ADAPTER_PATH\" \\"
            echo "  --target_base \"Qwen/Qwen3-8B\" \\"
            echo "  --merge_tag \"$TAG\" \\"
            echo "  --template \"$TEMPLATE\""
            echo ""
        } >> $OUTPUT_SCRIPT
    done
done

# 4. 生成 Eval Config (修正路径后缀)
echo ">>> Step 3: Generating Eval Config List..."

echo "" >> $OUTPUT_SCRIPT
echo "echo '===== Copy below to your eval config ====='" >> $OUTPUT_SCRIPT
echo "cat << 'EOF'" >> $OUTPUT_SCRIPT
echo "EVAL_LINES = [" >> $OUTPUT_SCRIPT

for PREFIX in "${PREFIXES[@]}"; do
    for TYPE in "${SUB_TYPES[@]}"; do
        FOLDER_NAME="${PREFIX}_${TYPE}"
        TAG="${FOLDER_NAME}"
        
        # === 关键修正 ===
        # 根据你的 ls 输出: .../Single_ModelA_AttnOnly/merged-Single_ModelA_AttnOnly
        # 所以实际的模型路径是: ADAPTER_PATH + "/merged-" + TAG
        REAL_MERGED_PATH="$ANALYSIS_DIR/$FOLDER_NAME/merged-$TAG"
        
        echo "    ('$FOLDER_NAME', '$REAL_MERGED_PATH')," >> $OUTPUT_SCRIPT
    done
done

echo "]" >> $OUTPUT_SCRIPT
echo "EOF" >> $OUTPUT_SCRIPT

echo ">>> Done! Please run: bash $OUTPUT_SCRIPT"
