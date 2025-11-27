# Auto-generated Merge Script
echo "Processing: CodeZ1_Single_ModelA_AttnOnly"
mkdir -p "/dockerdata/LLM-NEO-bk/results/1123/result-Qwen3-8B-Base-1123/analysis_suite_1124/CodeZ1_Single_ModelA_AttnOnly/merged"
python3 /dockerdata/LLM-NEO-bk/src/shadow/merge_lora.py \
  --adapter_path "/dockerdata/LLM-NEO-bk/results/1123/result-Qwen3-8B-Base-1123/analysis_suite_1124/CodeZ1_Single_ModelA_AttnOnly" \
  --target_base "Qwen/Qwen3-8B" \
  --merge_tag "CodeZ1_Single_ModelA_AttnOnly" \
  --template "qwen3"

echo "Processing: CodeZ1_Single_ModelA_MlpOnly"
mkdir -p "/dockerdata/LLM-NEO-bk/results/1123/result-Qwen3-8B-Base-1123/analysis_suite_1124/CodeZ1_Single_ModelA_MlpOnly/merged"
python3 /dockerdata/LLM-NEO-bk/src/shadow/merge_lora.py \
  --adapter_path "/dockerdata/LLM-NEO-bk/results/1123/result-Qwen3-8B-Base-1123/analysis_suite_1124/CodeZ1_Single_ModelA_MlpOnly" \
  --target_base "Qwen/Qwen3-8B" \
  --merge_tag "CodeZ1_Single_ModelA_MlpOnly" \
  --template "qwen3"

echo "Processing: CodeZ1_Single_ModelA_Shallow"
mkdir -p "/dockerdata/LLM-NEO-bk/results/1123/result-Qwen3-8B-Base-1123/analysis_suite_1124/CodeZ1_Single_ModelA_Shallow/merged"
python3 /dockerdata/LLM-NEO-bk/src/shadow/merge_lora.py \
  --adapter_path "/dockerdata/LLM-NEO-bk/results/1123/result-Qwen3-8B-Base-1123/analysis_suite_1124/CodeZ1_Single_ModelA_Shallow" \
  --target_base "Qwen/Qwen3-8B" \
  --merge_tag "CodeZ1_Single_ModelA_Shallow" \
  --template "qwen3"

echo "Processing: CodeZ1_Single_ModelA_Deep"
mkdir -p "/dockerdata/LLM-NEO-bk/results/1123/result-Qwen3-8B-Base-1123/analysis_suite_1124/CodeZ1_Single_ModelA_Deep/merged"
python3 /dockerdata/LLM-NEO-bk/src/shadow/merge_lora.py \
  --adapter_path "/dockerdata/LLM-NEO-bk/results/1123/result-Qwen3-8B-Base-1123/analysis_suite_1124/CodeZ1_Single_ModelA_Deep" \
  --target_base "Qwen/Qwen3-8B" \
  --merge_tag "CodeZ1_Single_ModelA_Deep" \
  --template "qwen3"

echo "Processing: CodeZ1_Single_ModelB_AttnOnly"
mkdir -p "/dockerdata/LLM-NEO-bk/results/1123/result-Qwen3-8B-Base-1123/analysis_suite_1124/CodeZ1_Single_ModelB_AttnOnly/merged"
python3 /dockerdata/LLM-NEO-bk/src/shadow/merge_lora.py \
  --adapter_path "/dockerdata/LLM-NEO-bk/results/1123/result-Qwen3-8B-Base-1123/analysis_suite_1124/CodeZ1_Single_ModelB_AttnOnly" \
  --target_base "Qwen/Qwen3-8B" \
  --merge_tag "CodeZ1_Single_ModelB_AttnOnly" \
  --template "qwen3"

echo "Processing: CodeZ1_Single_ModelB_MlpOnly"
mkdir -p "/dockerdata/LLM-NEO-bk/results/1123/result-Qwen3-8B-Base-1123/analysis_suite_1124/CodeZ1_Single_ModelB_MlpOnly/merged"
python3 /dockerdata/LLM-NEO-bk/src/shadow/merge_lora.py \
  --adapter_path "/dockerdata/LLM-NEO-bk/results/1123/result-Qwen3-8B-Base-1123/analysis_suite_1124/CodeZ1_Single_ModelB_MlpOnly" \
  --target_base "Qwen/Qwen3-8B" \
  --merge_tag "CodeZ1_Single_ModelB_MlpOnly" \
  --template "qwen3"

echo "Processing: CodeZ1_Single_ModelB_Shallow"
mkdir -p "/dockerdata/LLM-NEO-bk/results/1123/result-Qwen3-8B-Base-1123/analysis_suite_1124/CodeZ1_Single_ModelB_Shallow/merged"
python3 /dockerdata/LLM-NEO-bk/src/shadow/merge_lora.py \
  --adapter_path "/dockerdata/LLM-NEO-bk/results/1123/result-Qwen3-8B-Base-1123/analysis_suite_1124/CodeZ1_Single_ModelB_Shallow" \
  --target_base "Qwen/Qwen3-8B" \
  --merge_tag "CodeZ1_Single_ModelB_Shallow" \
  --template "qwen3"

echo "Processing: CodeZ1_Single_ModelB_Deep"
mkdir -p "/dockerdata/LLM-NEO-bk/results/1123/result-Qwen3-8B-Base-1123/analysis_suite_1124/CodeZ1_Single_ModelB_Deep/merged"
python3 /dockerdata/LLM-NEO-bk/src/shadow/merge_lora.py \
  --adapter_path "/dockerdata/LLM-NEO-bk/results/1123/result-Qwen3-8B-Base-1123/analysis_suite_1124/CodeZ1_Single_ModelB_Deep" \
  --target_base "Qwen/Qwen3-8B" \
  --merge_tag "CodeZ1_Single_ModelB_Deep" \
  --template "qwen3"

echo "Processing: CodeZ1_Hybrid_Attn-A_Mlp-B"
mkdir -p "/dockerdata/LLM-NEO-bk/results/1123/result-Qwen3-8B-Base-1123/analysis_suite_1124/CodeZ1_Hybrid_Attn-A_Mlp-B/merged"
python3 /dockerdata/LLM-NEO-bk/src/shadow/merge_lora.py \
  --adapter_path "/dockerdata/LLM-NEO-bk/results/1123/result-Qwen3-8B-Base-1123/analysis_suite_1124/CodeZ1_Hybrid_Attn-A_Mlp-B" \
  --target_base "Qwen/Qwen3-8B" \
  --merge_tag "CodeZ1_Hybrid_Attn-A_Mlp-B" \
  --template "qwen3"

echo "Processing: CodeZ1_Hybrid_Mlp-A_Attn-B"
mkdir -p "/dockerdata/LLM-NEO-bk/results/1123/result-Qwen3-8B-Base-1123/analysis_suite_1124/CodeZ1_Hybrid_Mlp-A_Attn-B/merged"
python3 /dockerdata/LLM-NEO-bk/src/shadow/merge_lora.py \
  --adapter_path "/dockerdata/LLM-NEO-bk/results/1123/result-Qwen3-8B-Base-1123/analysis_suite_1124/CodeZ1_Hybrid_Mlp-A_Attn-B" \
  --target_base "Qwen/Qwen3-8B" \
  --merge_tag "CodeZ1_Hybrid_Mlp-A_Attn-B" \
  --template "qwen3"

echo "Processing: CodeZ1_Hybrid_Shallow-A_Deep-B"
mkdir -p "/dockerdata/LLM-NEO-bk/results/1123/result-Qwen3-8B-Base-1123/analysis_suite_1124/CodeZ1_Hybrid_Shallow-A_Deep-B/merged"
python3 /dockerdata/LLM-NEO-bk/src/shadow/merge_lora.py \
  --adapter_path "/dockerdata/LLM-NEO-bk/results/1123/result-Qwen3-8B-Base-1123/analysis_suite_1124/CodeZ1_Hybrid_Shallow-A_Deep-B" \
  --target_base "Qwen/Qwen3-8B" \
  --merge_tag "CodeZ1_Hybrid_Shallow-A_Deep-B" \
  --template "qwen3"

echo "Processing: CodeZ1_Hybrid_Deep-A_Shallow-B"
mkdir -p "/dockerdata/LLM-NEO-bk/results/1123/result-Qwen3-8B-Base-1123/analysis_suite_1124/CodeZ1_Hybrid_Deep-A_Shallow-B/merged"
python3 /dockerdata/LLM-NEO-bk/src/shadow/merge_lora.py \
  --adapter_path "/dockerdata/LLM-NEO-bk/results/1123/result-Qwen3-8B-Base-1123/analysis_suite_1124/CodeZ1_Hybrid_Deep-A_Shallow-B" \
  --target_base "Qwen/Qwen3-8B" \
  --merge_tag "CodeZ1_Hybrid_Deep-A_Shallow-B" \
  --template "qwen3"

echo "Processing: Shadow2k_Single_ModelA_AttnOnly"
mkdir -p "/dockerdata/LLM-NEO-bk/results/1123/result-Qwen3-8B-Base-1123/analysis_suite_1124/Shadow2k_Single_ModelA_AttnOnly/merged"
python3 /dockerdata/LLM-NEO-bk/src/shadow/merge_lora.py \
  --adapter_path "/dockerdata/LLM-NEO-bk/results/1123/result-Qwen3-8B-Base-1123/analysis_suite_1124/Shadow2k_Single_ModelA_AttnOnly" \
  --target_base "Qwen/Qwen3-8B" \
  --merge_tag "Shadow2k_Single_ModelA_AttnOnly" \
  --template "qwen3"

echo "Processing: Shadow2k_Single_ModelA_MlpOnly"
mkdir -p "/dockerdata/LLM-NEO-bk/results/1123/result-Qwen3-8B-Base-1123/analysis_suite_1124/Shadow2k_Single_ModelA_MlpOnly/merged"
python3 /dockerdata/LLM-NEO-bk/src/shadow/merge_lora.py \
  --adapter_path "/dockerdata/LLM-NEO-bk/results/1123/result-Qwen3-8B-Base-1123/analysis_suite_1124/Shadow2k_Single_ModelA_MlpOnly" \
  --target_base "Qwen/Qwen3-8B" \
  --merge_tag "Shadow2k_Single_ModelA_MlpOnly" \
  --template "qwen3"

echo "Processing: Shadow2k_Single_ModelA_Shallow"
mkdir -p "/dockerdata/LLM-NEO-bk/results/1123/result-Qwen3-8B-Base-1123/analysis_suite_1124/Shadow2k_Single_ModelA_Shallow/merged"
python3 /dockerdata/LLM-NEO-bk/src/shadow/merge_lora.py \
  --adapter_path "/dockerdata/LLM-NEO-bk/results/1123/result-Qwen3-8B-Base-1123/analysis_suite_1124/Shadow2k_Single_ModelA_Shallow" \
  --target_base "Qwen/Qwen3-8B" \
  --merge_tag "Shadow2k_Single_ModelA_Shallow" \
  --template "qwen3"

echo "Processing: Shadow2k_Single_ModelA_Deep"
mkdir -p "/dockerdata/LLM-NEO-bk/results/1123/result-Qwen3-8B-Base-1123/analysis_suite_1124/Shadow2k_Single_ModelA_Deep/merged"
python3 /dockerdata/LLM-NEO-bk/src/shadow/merge_lora.py \
  --adapter_path "/dockerdata/LLM-NEO-bk/results/1123/result-Qwen3-8B-Base-1123/analysis_suite_1124/Shadow2k_Single_ModelA_Deep" \
  --target_base "Qwen/Qwen3-8B" \
  --merge_tag "Shadow2k_Single_ModelA_Deep" \
  --template "qwen3"

echo "Processing: Shadow2k_Single_ModelB_AttnOnly"
mkdir -p "/dockerdata/LLM-NEO-bk/results/1123/result-Qwen3-8B-Base-1123/analysis_suite_1124/Shadow2k_Single_ModelB_AttnOnly/merged"
python3 /dockerdata/LLM-NEO-bk/src/shadow/merge_lora.py \
  --adapter_path "/dockerdata/LLM-NEO-bk/results/1123/result-Qwen3-8B-Base-1123/analysis_suite_1124/Shadow2k_Single_ModelB_AttnOnly" \
  --target_base "Qwen/Qwen3-8B" \
  --merge_tag "Shadow2k_Single_ModelB_AttnOnly" \
  --template "qwen3"

echo "Processing: Shadow2k_Single_ModelB_MlpOnly"
mkdir -p "/dockerdata/LLM-NEO-bk/results/1123/result-Qwen3-8B-Base-1123/analysis_suite_1124/Shadow2k_Single_ModelB_MlpOnly/merged"
python3 /dockerdata/LLM-NEO-bk/src/shadow/merge_lora.py \
  --adapter_path "/dockerdata/LLM-NEO-bk/results/1123/result-Qwen3-8B-Base-1123/analysis_suite_1124/Shadow2k_Single_ModelB_MlpOnly" \
  --target_base "Qwen/Qwen3-8B" \
  --merge_tag "Shadow2k_Single_ModelB_MlpOnly" \
  --template "qwen3"

echo "Processing: Shadow2k_Single_ModelB_Shallow"
mkdir -p "/dockerdata/LLM-NEO-bk/results/1123/result-Qwen3-8B-Base-1123/analysis_suite_1124/Shadow2k_Single_ModelB_Shallow/merged"
python3 /dockerdata/LLM-NEO-bk/src/shadow/merge_lora.py \
  --adapter_path "/dockerdata/LLM-NEO-bk/results/1123/result-Qwen3-8B-Base-1123/analysis_suite_1124/Shadow2k_Single_ModelB_Shallow" \
  --target_base "Qwen/Qwen3-8B" \
  --merge_tag "Shadow2k_Single_ModelB_Shallow" \
  --template "qwen3"

echo "Processing: Shadow2k_Single_ModelB_Deep"
mkdir -p "/dockerdata/LLM-NEO-bk/results/1123/result-Qwen3-8B-Base-1123/analysis_suite_1124/Shadow2k_Single_ModelB_Deep/merged"
python3 /dockerdata/LLM-NEO-bk/src/shadow/merge_lora.py \
  --adapter_path "/dockerdata/LLM-NEO-bk/results/1123/result-Qwen3-8B-Base-1123/analysis_suite_1124/Shadow2k_Single_ModelB_Deep" \
  --target_base "Qwen/Qwen3-8B" \
  --merge_tag "Shadow2k_Single_ModelB_Deep" \
  --template "qwen3"

echo "Processing: Shadow2k_Hybrid_Attn-A_Mlp-B"
mkdir -p "/dockerdata/LLM-NEO-bk/results/1123/result-Qwen3-8B-Base-1123/analysis_suite_1124/Shadow2k_Hybrid_Attn-A_Mlp-B/merged"
python3 /dockerdata/LLM-NEO-bk/src/shadow/merge_lora.py \
  --adapter_path "/dockerdata/LLM-NEO-bk/results/1123/result-Qwen3-8B-Base-1123/analysis_suite_1124/Shadow2k_Hybrid_Attn-A_Mlp-B" \
  --target_base "Qwen/Qwen3-8B" \
  --merge_tag "Shadow2k_Hybrid_Attn-A_Mlp-B" \
  --template "qwen3"

echo "Processing: Shadow2k_Hybrid_Mlp-A_Attn-B"
mkdir -p "/dockerdata/LLM-NEO-bk/results/1123/result-Qwen3-8B-Base-1123/analysis_suite_1124/Shadow2k_Hybrid_Mlp-A_Attn-B/merged"
python3 /dockerdata/LLM-NEO-bk/src/shadow/merge_lora.py \
  --adapter_path "/dockerdata/LLM-NEO-bk/results/1123/result-Qwen3-8B-Base-1123/analysis_suite_1124/Shadow2k_Hybrid_Mlp-A_Attn-B" \
  --target_base "Qwen/Qwen3-8B" \
  --merge_tag "Shadow2k_Hybrid_Mlp-A_Attn-B" \
  --template "qwen3"

echo "Processing: Shadow2k_Hybrid_Shallow-A_Deep-B"
mkdir -p "/dockerdata/LLM-NEO-bk/results/1123/result-Qwen3-8B-Base-1123/analysis_suite_1124/Shadow2k_Hybrid_Shallow-A_Deep-B/merged"
python3 /dockerdata/LLM-NEO-bk/src/shadow/merge_lora.py \
  --adapter_path "/dockerdata/LLM-NEO-bk/results/1123/result-Qwen3-8B-Base-1123/analysis_suite_1124/Shadow2k_Hybrid_Shallow-A_Deep-B" \
  --target_base "Qwen/Qwen3-8B" \
  --merge_tag "Shadow2k_Hybrid_Shallow-A_Deep-B" \
  --template "qwen3"

echo "Processing: Shadow2k_Hybrid_Deep-A_Shallow-B"
mkdir -p "/dockerdata/LLM-NEO-bk/results/1123/result-Qwen3-8B-Base-1123/analysis_suite_1124/Shadow2k_Hybrid_Deep-A_Shallow-B/merged"
python3 /dockerdata/LLM-NEO-bk/src/shadow/merge_lora.py \
  --adapter_path "/dockerdata/LLM-NEO-bk/results/1123/result-Qwen3-8B-Base-1123/analysis_suite_1124/Shadow2k_Hybrid_Deep-A_Shallow-B" \
  --target_base "Qwen/Qwen3-8B" \
  --merge_tag "Shadow2k_Hybrid_Deep-A_Shallow-B" \
  --template "qwen3"


echo '===== Copy below to your eval config ====='
cat << 'EOF'
EVAL_LINES = [
    ('CodeZ1_Single_ModelA_AttnOnly', '/dockerdata/LLM-NEO-bk/results/1123/result-Qwen3-8B-Base-1123/analysis_suite_1124/CodeZ1_Single_ModelA_AttnOnly/merged-CodeZ1_Single_ModelA_AttnOnly'),
    ('CodeZ1_Single_ModelA_MlpOnly', '/dockerdata/LLM-NEO-bk/results/1123/result-Qwen3-8B-Base-1123/analysis_suite_1124/CodeZ1_Single_ModelA_MlpOnly/merged-CodeZ1_Single_ModelA_MlpOnly'),
    ('CodeZ1_Single_ModelA_Shallow', '/dockerdata/LLM-NEO-bk/results/1123/result-Qwen3-8B-Base-1123/analysis_suite_1124/CodeZ1_Single_ModelA_Shallow/merged-CodeZ1_Single_ModelA_Shallow'),
    ('CodeZ1_Single_ModelA_Deep', '/dockerdata/LLM-NEO-bk/results/1123/result-Qwen3-8B-Base-1123/analysis_suite_1124/CodeZ1_Single_ModelA_Deep/merged-CodeZ1_Single_ModelA_Deep'),
    ('CodeZ1_Single_ModelB_AttnOnly', '/dockerdata/LLM-NEO-bk/results/1123/result-Qwen3-8B-Base-1123/analysis_suite_1124/CodeZ1_Single_ModelB_AttnOnly/merged-CodeZ1_Single_ModelB_AttnOnly'),
    ('CodeZ1_Single_ModelB_MlpOnly', '/dockerdata/LLM-NEO-bk/results/1123/result-Qwen3-8B-Base-1123/analysis_suite_1124/CodeZ1_Single_ModelB_MlpOnly/merged-CodeZ1_Single_ModelB_MlpOnly'),
    ('CodeZ1_Single_ModelB_Shallow', '/dockerdata/LLM-NEO-bk/results/1123/result-Qwen3-8B-Base-1123/analysis_suite_1124/CodeZ1_Single_ModelB_Shallow/merged-CodeZ1_Single_ModelB_Shallow'),
    ('CodeZ1_Single_ModelB_Deep', '/dockerdata/LLM-NEO-bk/results/1123/result-Qwen3-8B-Base-1123/analysis_suite_1124/CodeZ1_Single_ModelB_Deep/merged-CodeZ1_Single_ModelB_Deep'),
    ('CodeZ1_Hybrid_Attn-A_Mlp-B', '/dockerdata/LLM-NEO-bk/results/1123/result-Qwen3-8B-Base-1123/analysis_suite_1124/CodeZ1_Hybrid_Attn-A_Mlp-B/merged-CodeZ1_Hybrid_Attn-A_Mlp-B'),
    ('CodeZ1_Hybrid_Mlp-A_Attn-B', '/dockerdata/LLM-NEO-bk/results/1123/result-Qwen3-8B-Base-1123/analysis_suite_1124/CodeZ1_Hybrid_Mlp-A_Attn-B/merged-CodeZ1_Hybrid_Mlp-A_Attn-B'),
    ('CodeZ1_Hybrid_Shallow-A_Deep-B', '/dockerdata/LLM-NEO-bk/results/1123/result-Qwen3-8B-Base-1123/analysis_suite_1124/CodeZ1_Hybrid_Shallow-A_Deep-B/merged-CodeZ1_Hybrid_Shallow-A_Deep-B'),
    ('CodeZ1_Hybrid_Deep-A_Shallow-B', '/dockerdata/LLM-NEO-bk/results/1123/result-Qwen3-8B-Base-1123/analysis_suite_1124/CodeZ1_Hybrid_Deep-A_Shallow-B/merged-CodeZ1_Hybrid_Deep-A_Shallow-B'),
    ('Shadow2k_Single_ModelA_AttnOnly', '/dockerdata/LLM-NEO-bk/results/1123/result-Qwen3-8B-Base-1123/analysis_suite_1124/Shadow2k_Single_ModelA_AttnOnly/merged-Shadow2k_Single_ModelA_AttnOnly'),
    ('Shadow2k_Single_ModelA_MlpOnly', '/dockerdata/LLM-NEO-bk/results/1123/result-Qwen3-8B-Base-1123/analysis_suite_1124/Shadow2k_Single_ModelA_MlpOnly/merged-Shadow2k_Single_ModelA_MlpOnly'),
    ('Shadow2k_Single_ModelA_Shallow', '/dockerdata/LLM-NEO-bk/results/1123/result-Qwen3-8B-Base-1123/analysis_suite_1124/Shadow2k_Single_ModelA_Shallow/merged-Shadow2k_Single_ModelA_Shallow'),
    ('Shadow2k_Single_ModelA_Deep', '/dockerdata/LLM-NEO-bk/results/1123/result-Qwen3-8B-Base-1123/analysis_suite_1124/Shadow2k_Single_ModelA_Deep/merged-Shadow2k_Single_ModelA_Deep'),
    ('Shadow2k_Single_ModelB_AttnOnly', '/dockerdata/LLM-NEO-bk/results/1123/result-Qwen3-8B-Base-1123/analysis_suite_1124/Shadow2k_Single_ModelB_AttnOnly/merged-Shadow2k_Single_ModelB_AttnOnly'),
    ('Shadow2k_Single_ModelB_MlpOnly', '/dockerdata/LLM-NEO-bk/results/1123/result-Qwen3-8B-Base-1123/analysis_suite_1124/Shadow2k_Single_ModelB_MlpOnly/merged-Shadow2k_Single_ModelB_MlpOnly'),
    ('Shadow2k_Single_ModelB_Shallow', '/dockerdata/LLM-NEO-bk/results/1123/result-Qwen3-8B-Base-1123/analysis_suite_1124/Shadow2k_Single_ModelB_Shallow/merged-Shadow2k_Single_ModelB_Shallow'),
    ('Shadow2k_Single_ModelB_Deep', '/dockerdata/LLM-NEO-bk/results/1123/result-Qwen3-8B-Base-1123/analysis_suite_1124/Shadow2k_Single_ModelB_Deep/merged-Shadow2k_Single_ModelB_Deep'),
    ('Shadow2k_Hybrid_Attn-A_Mlp-B', '/dockerdata/LLM-NEO-bk/results/1123/result-Qwen3-8B-Base-1123/analysis_suite_1124/Shadow2k_Hybrid_Attn-A_Mlp-B/merged-Shadow2k_Hybrid_Attn-A_Mlp-B'),
    ('Shadow2k_Hybrid_Mlp-A_Attn-B', '/dockerdata/LLM-NEO-bk/results/1123/result-Qwen3-8B-Base-1123/analysis_suite_1124/Shadow2k_Hybrid_Mlp-A_Attn-B/merged-Shadow2k_Hybrid_Mlp-A_Attn-B'),
    ('Shadow2k_Hybrid_Shallow-A_Deep-B', '/dockerdata/LLM-NEO-bk/results/1123/result-Qwen3-8B-Base-1123/analysis_suite_1124/Shadow2k_Hybrid_Shallow-A_Deep-B/merged-Shadow2k_Hybrid_Shallow-A_Deep-B'),
    ('Shadow2k_Hybrid_Deep-A_Shallow-B', '/dockerdata/LLM-NEO-bk/results/1123/result-Qwen3-8B-Base-1123/analysis_suite_1124/Shadow2k_Hybrid_Deep-A_Shallow-B/merged-Shadow2k_Hybrid_Deep-A_Shallow-B'),
]
EOF
