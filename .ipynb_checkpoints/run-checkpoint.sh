# # #!/usr/bin/env bash
# # ###############################################################################
# # ##### 0. Globals                                                             #####
# # ###############################################################################
# # # WORKSPACE_DIR="$(cd "$(dirname "$0")" && pwd)"
# # WORKSPACE_DIR="/home/ubuntu/Shadow"
# # RESULTS_DIR="$WORKSPACE_DIR/results"
# # SCRIPT_OUTPUT_DIR="$WORKSPACE_DIR/scripts"
# # mkdir -p "$RESULTS_DIR" "$SCRIPT_OUTPUT_DIR"

# # # --- LoRA switch -------------------------------------------------------------
# # USE_LORA=true                   # false -> full SFT
# # is_lora()   { [[ "${USE_LORA,,}" == "true" ]]; }

# # lora_ranks=(128)
# # # ratios=(0.5)
# # learning_rates_lora=(2e-4)
# # # learning_rates_lora=(5e-6 1e-5 2e-5 5e-5 1e-4 2e-4 5e-4 1e-3 2e-3)
# # learning_rates_sft=(1e-5)

# # if is_lora; then
# #   learning_rates=("${learning_rates_lora[@]}")
# # else
# #   learning_rates=("${learning_rates_sft[@]}")
# # fi

# # # --- Helpers -----------------------------------------------------------------
# # # 统一 k 表示：整千 -> 2k，非整千 -> 2.5k（保留一位小数，不使用 2.0k）
# # format_k() {
# #   local num=$1
# #   if (( num % 1000 == 0 )); then
# #     echo "$((num/1000))k"
# #   else
# #     awk -v n="$num" 'BEGIN{printf "%.1fk", n/1000}'
# #   fi
# # }
# # # 科学计数法/十进制 -> 十进制小数（最多12位小数，去尾零）
# # to_decimal() {
# #   LC_NUMERIC=C printf "%.12f" "$1" | sed -E 's/0+$//;s/\.$/.0/'
# # }
# # # lr 标签统一为十进制：lr0.000005
# # lr_tag_dec() { echo "lr$(to_decimal "$1")"; }

# # # 生成 eval 列表项；comment_level: 1 -> "#", 2 -> "##"
# # add_eval() {
# #   local abs="$1" comment_level="${2:-1}"
# #   local rel="${abs#$RESULTS_DIR/}"
# #   local prefix="#"
# #   [[ "$comment_level" -eq 2 ]] && prefix="##"
# #   EVAL_LINES+=("$prefix ('$rel','$abs'),")
# # }

# # maybe_rel() { [[ $1 = /* ]] && echo "$1" || echo "../$1"; }

# # ###############################################################################
# # ##### 1. Base-model resolution (no jq)                                       #####
# # ###############################################################################
# # MODEL_DIR="" # ""→HF
# # MODEL_PAIR_FILE="$WORKSPACE_DIR/examples/model_pair.json"
# # BASE_MODELS=(
# #   # "Qwen2.5-14B"
# #   # "Qwen3-0.6B"
# #   # "Qwen3-8B"
# #   "Llama3.1-8B"
# # )

# # MODEL_PAIRS=()                  # will contain "<base>||<instruct>"

# # for NAME in "${BASE_MODELS[@]}"; do
# #   BLOCK=$(awk -v n="\"$NAME\"" '
# #     $0~n {print; getline;
# #            while ($0 !~ /\}/) {print; getline}; print; exit}' \
# #     "$MODEL_PAIR_FILE")
# #   [[ -z $BLOCK ]] && { echo "ERROR: '$NAME' not found in JSON"; exit 1; }

# #   HF_BASE=$(printf '%s\n' "$BLOCK" | sed -n 's/.*"hf_base_path":[[:space:]]*"\([^"]*\)".*/\1/p')
# #   HF_INST=$(printf '%s\n' "$BLOCK" | sed -n 's/.*"hf_instruct_path":[[:space:]]*"\([^"]*\)".*/\1/p')
# #   [[ -z $HF_BASE || -z $HF_INST ]] && { echo "ERROR: malformed JSON for $NAME"; exit 1; }

# #   if [[ -n $MODEL_DIR ]]; then
# #       REL_BASE=${HF_BASE#*/}
# #       REL_INST=${HF_INST#*/}
# #       BASE_PATH="$MODEL_DIR/$REL_BASE"
# #       INST_PATH="$MODEL_DIR/$REL_INST"
# #       [[ -d $BASE_PATH ]] || { echo "ERROR: $BASE_PATH missing"; exit 1; }
# #       [[ -d $INST_PATH ]] || { echo "ERROR: $INST_PATH missing"; exit 1; }
# #   else
# #       BASE_PATH=$HF_BASE
# #       INST_PATH=$HF_INST
# #   fi

# #   MODEL_PAIRS+=("${BASE_PATH}||${INST_PATH}")
# #   echo "INFO  Base=$BASE_PATH  Instruct=$INST_PATH"
# # done

# # ###############################################################################
# # ##### 2. Generate train scripts                                              #####
# # ###############################################################################
# # MONTHDAY=$(date +%m%d)
# # TIMESTAMP=$(date +%m%d%H%M%S)

# # for PAIR in "${MODEL_PAIRS[@]}"; do
# #   B_MODEL="${PAIR%%||*}"
# #   I_MODEL="${PAIR##*||}"
# #   MODEL_BASE=$(basename "$B_MODEL")
# #   SCRIPT_FILE="$SCRIPT_OUTPUT_DIR/train_${MODEL_BASE}_${TIMESTAMP}.sh"
# #   : > "$SCRIPT_FILE"            # truncate / create

# #   # ================= header =================
# #   {
# #     echo "##### Auto-generated $(date '+%F %T') #####"
# #     echo "# Model     : $MODEL_BASE"
# #     echo "# LoRA mode : $USE_LORA"
# #   } >> "$SCRIPT_FILE"

# #   # ================= template =================
# #   case "$B_MODEL" in
# #     *Llama-3*)  template="llama3" ;;
# #     *Llama3*)  template="llama3" ;;
# #     *Qwen2*)    template="qwen" ;;
# #     *Llama-2*)  template="llama2" ;;
# #     *Llama2*)  template="llama2" ;;
# #     *Qwen3*)    template="qwen3" ;;
# #     *internlm2*)template="intern2" ;;
# #     *mistral_small*) template="mistral_small" ;;
# #     *Mistral*)  template="mistral" ;;
# #     *Falcon*)   template="falcon" ;;
# #     *gemma3*)   template="gemma3" ;;
# #     *gemma*)    template="gemma" ;;
# #     *Yi*)       template="yi" ;;
# #     *Baichuan*) template="baichuan2" ;;
# #     *) echo "ERROR: unknown template for $B_MODEL"; exit 1 ;;
# #   esac
# #   echo "# Template  : $template" >> "$SCRIPT_FILE"
# #   echo "" >> "$SCRIPT_FILE"

# #   # ================= env block =================
# #   {
# #     echo "##### Environment #####"
# #     echo "export VLLM_WORKER_MULTIPROC_METHOD=spawn"
# #     echo "export HF_HUB_OFFLINE=0"
# #     echo "export HF_DATASETS_OFFLINE=0"
# #     echo "export HF_DATASETS_TRUST_REMOTE_CODE=1"
# #     echo "export TRUST_REMOTE_CODE=True"
# #     echo "export HF_ALLOW_CODE_EVAL=1"
# #     echo ""
# #   } >> "$SCRIPT_FILE"

# #   # ================= constants =================
# #   # DEEPSPEED_CFG="$WORKSPACE_DIR/examples/deepspeed/ds_z3_config.json"
# #   DATASET="Shadow_2k"
# #   suffix_name="Shadow_2k"
# #   cutoff_len=4096
# #   samples=(2000)
# #   logging_steps=1; save_steps=1000; per_device_train_batch_size=2
# #   gradient_accumulation_steps=16; num_train_epochs=1
# #   lr_scheduler_type="cosine"; warmup_ratio=0.1; bf16=true
# #   val_size=0.01; per_device_eval_batch_size=1
# #   eval_strategy="steps"; eval_steps=10000; overwrite_cache=false

# #   # ================= training =================
# #   echo "##### Training #####" >> "$SCRIPT_FILE"

# #   EVAL_LINES=()   # reset per base model

# #   generate_train() {
# #     local M_PATH=$1 TAG=$2 LR=$3
# #     local LR_DEC LR_TAG
# #     LR_DEC="$(to_decimal "$LR")"
# #     LR_TAG="$(lr_tag_dec "$LR")"
# #     local OUT_ROOT="$RESULTS_DIR/${MONTHDAY}/result-${MODEL_BASE}-${MONTHDAY}"

# #     for MAX in "${samples[@]}"; do
# #       local K; K="$(format_k "$MAX")"
# #       local DIR="${TAG}-${K}-$(is_lora && echo lora-rank${lora_ranks[0]} || echo sft)-${LR_TAG}-${suffix_name}"
# #       local OUTDIR="$OUT_ROOT/$DIR"

# #       {
# #         echo "###### ${TAG}  max=$MAX  lr=$LR_DEC ######"
# #         echo "mkdir -p \"$OUTDIR\""
# #         echo "cd \"$WORKSPACE_DIR\""
# #         echo "llamafactory-cli train \\"
# #         echo "  --model_name_or_path \"$M_PATH\" \\"
# #         echo "  --stage sft \\"
# #         echo "  --do_train true \\"
# #         if is_lora; then
# #           echo "  --finetuning_type lora --lora_rank ${lora_ranks[0]} \\"
# #         else
# #           echo "  --finetuning_type full \\"
# #         fi
# #         echo "  --deepspeed examples/deepspeed/ds_z3_config.json \\"          >> "$SCRIPT_FILE"
# #         echo "  --dataset \"$DATASET\" \\"
# #         echo "  --template \"$template\" \\"
# #         echo "  --cutoff_len $cutoff_len \\"
# #         echo "  --max_samples $MAX \\"
# #         echo "  --output_dir \"$OUTDIR\" \\"
# #         echo "  --per_device_train_batch_size $per_device_train_batch_size \\"
# #         echo "  --gradient_accumulation_steps $gradient_accumulation_steps \\"
# #         echo "  --learning_rate $LR_DEC \\"
# #         echo "  --num_train_epochs $num_train_epochs \\"
# #         echo "  --logging_steps $logging_steps \\"
# #         echo "  --save_steps $save_steps \\"
# #         echo "  --save_only_model True \\"
# #         echo "  --plot_loss true \\"
# #         echo "  --lr_scheduler_type $lr_scheduler_type \\"
# #         echo "  --warmup_ratio $warmup_ratio \\"
# #         echo "  --bf16 $bf16 \\"
# #         echo "  --val_size $val_size \\"
# #         echo "  --per_device_eval_batch_size $per_device_eval_batch_size \\"
# #         echo "  --eval_strategy $eval_strategy \\"
# #         echo "  --eval_steps $eval_steps \\"
# #         echo "  --trust_remote_code True \\"
# #         echo "  --flash_attn fa2 \\"
# #         echo "  --overwrite_cache $overwrite_cache"
# #         echo ""
# #       } >> "$SCRIPT_FILE"

# #       # evaluation tuple for SFT（仅非 LoRA 且 I 路径）
# #       if ! is_lora && [[ $TAG == I ]]; then
# #         add_eval "$OUTDIR" 1
# #       fi
# #     done
# #   }

# #   for LR in "${learning_rates[@]}"; do
# #     generate_train "$B_MODEL" B "$LR"
# #     generate_train "$I_MODEL" I "$LR"
# #   done

# #   # ================= delta-merge =================
# #   if is_lora; then
# #     echo "##### LoRA delta-merge #####" >> "$SCRIPT_FILE"

# #     # 参数6：若传入 '#' 则命令以单井号注释；evaluation 用双井号 '##'
# #     merge_lora() {
# #       local SRC=$1 SRC_TAG=$2 TGT=$3 TGT_TAG=$4 LR=$5 COMMENT_PREFIX=${6:-}
# #       local LR_DEC LR_TAG
# #       LR_DEC="$(to_decimal "$LR")"
# #       LR_TAG="$(lr_tag_dec "$LR")"
# #       local SRC_ROOT="$RESULTS_DIR/${MONTHDAY}/result-${MODEL_BASE}-${MONTHDAY}"

# #       for RANK in "${lora_ranks[@]}"; do
# #         for MAX in "${samples[@]}"; do
# #           local K; K="$(format_k "$MAX")"
# #           local ADAP="$SRC_ROOT/${SRC_TAG}-${K}-lora-rank${RANK}-${LR_TAG}-${suffix_name}"
# #           local TAG="${SRC_TAG}2${TGT_TAG}"
# #           # 合并产物目录：挂在各自训练输出目录下，并带 lr 后缀
# #           local MERGED="$ADAP/merged-${TAG}-${LR_TAG}"

# #           {
# #             echo "${COMMENT_PREFIX}# (optional) ensure target dir exists"
# #             echo "${COMMENT_PREFIX}mkdir -p \"$MERGED\""
# #             echo "${COMMENT_PREFIX}python3 $WORKSPACE_DIR/src/shadow/merge_lora.py \\"
# #             echo "${COMMENT_PREFIX}  --adapter_path \"$ADAP\" \\"
# #             echo "${COMMENT_PREFIX}  --target_base \"$TGT\" \\"
# #             echo "${COMMENT_PREFIX}  --merge_tag \"$TAG\" \\"
# #             echo "${COMMENT_PREFIX}  --template \"$template\""
# #             echo ""
# #           } >> "$SCRIPT_FILE"

# #           # evaluation：启用（B2I/I2I）-> "#"; 注释（I2B/B2B）-> "##"
# #           if [[ -z "$COMMENT_PREFIX" ]]; then
# #             add_eval "$MERGED" 1
# #           else
# #             add_eval "$MERGED" 2
# #           fi
# #         done
# #       done
# #     }

# #     for LR in "${learning_rates[@]}"; do
# #       merge_lora "$B_MODEL" B "$I_MODEL" I "$LR"          # B2I（启用，#）
# #       merge_lora "$I_MODEL" I "$I_MODEL" I "$LR"          # I2I（启用，#）
# #       merge_lora "$I_MODEL" I "$B_MODEL" B "$LR" "#"      # I2B（命令 #，evaluation ##）
# #       merge_lora "$B_MODEL" B "$B_MODEL" B "$LR" "#"      # B2B（命令 #，evaluation ##）
# #     done
# #   else
# #     echo "##### SFT delta-merge #####" >> "$SCRIPT_FILE"
# #     # SFT：仅以 samples[0]/learning_rates[0] 生成一次 B2I 差分
# #     LR="${learning_rates[0]}"
# #     LR_DEC="$(to_decimal "$LR")"
# #     LR_TAG="$(lr_tag_dec "$LR")"
# #     K="$(format_k "${samples[0]}")"
# #     B_DIR="$RESULTS_DIR/${MONTHDAY}/result-${MODEL_BASE}-${MONTHDAY}/B-${K}-sft-${LR_TAG}-${suffix_name}"
# #     # 合并结果放在训练目录下
# #     MERGED="$B_DIR/merged-B2I-${LR_TAG}"
# #     {
# #       echo "python3 $WORKSPACE_DIR/src/shadow/apply_diff.py \\"
# #       echo "  --tuned_model \"$B_DIR\" \\"
# #       echo "  --target_model \"$I_MODEL\" \\"
# #       echo "  --base_model \"$B_MODEL\""
# #       echo ""
# #     } >> "$SCRIPT_FILE"
# #     add_eval "$MERGED" 1
# #   fi


# #   # ================= evaluation list =================
# #   {
# #     echo "##### Evaluation list #####"
# #     for line in "${EVAL_LINES[@]}"; do
# #       echo "$line"
# #     done
# #     echo ""
# #     echo "# please copy this eval_config to opencompass/examples/eval_shadow_202505.py and then run"
# #     echo ""
# #     echo "cd ./opencompass"
# #     echo "python3 ./run.py ./eval_shadow_202505.py -r 20250727200010 # 校准"
# #     echo "bash ./eval_instruct_0427.sh  ./eval_shadow_20250727_part1.py"
# #     echo "bash ./eval_instruct_0427.sh  ./eval_shadow_20250727.py"
# #     echo "bash ./eval_instruct_0427.sh  ./eval_quantw_20250727.py"
# #     echo "bash ./eval_instruct_0427.sh  ./eval_shadow_20250727_part2.py"



# #     echo "cd ./opencompass"
# #     echo "pip install huggingface_hub"
# #     echo "python upload_hf.py"

# #     echo ""
# #   } >> "$SCRIPT_FILE"

# # done

# # echo "Generation finished. Check $SCRIPT_FILE"


# #!/usr/bin/env bash
# ###############################################################################
# ##### 0. Globals                                                             #####
# ###############################################################################
# WORKSPACE_DIR="$(cd "$(dirname "$0")" && pwd)"
# # WORKSPACE_DIR="/root/shadow_exp/new_shadow/Shadow"
# RESULTS_DIR="$WORKSPACE_DIR/results"
# SCRIPT_OUTPUT_DIR="$WORKSPACE_DIR/scripts"
# mkdir -p "$RESULTS_DIR" "$SCRIPT_OUTPUT_DIR"

# # --- LoRA switch -------------------------------------------------------------
# USE_LORA=true                   # false -> full SFT
# is_lora()   { [[ "${USE_LORA,,}" == "true" ]]; }

# lora_ranks=(128)
# # ratios=(0.5)
# learning_rates_lora=(2e-4)
# # learning_rates_lora=(5e-6 1e-5 2e-5 5e-4 1e-4 2e-4 5e-4 1e-3 2e-3)
# learning_rates_sft=(1e-5)

# if is_lora; then
#   learning_rates=("${learning_rates_lora[@]}")
# else
#   learning_rates=("${learning_rates_sft[@]}")
# fi

# # --- Helpers -----------------------------------------------------------------
# # 统一 k 表示：整千 -> 2k，非整千 -> 2.5k（保留一位小数，不使用 2.0k）
# format_k() {
#   local num=$1
#   if (( num % 1000 == 0 )); then
#     echo "$((num/1000))k"
#   else
#     awk -v n="$num" 'BEGIN{printf "%.1fk", n/1000}'
#   fi
# }
# # 将科学计数法/十进制统一转为十进制小数（不使用科学计数法）
# to_decimal() {
#   LC_NUMERIC=C printf "%.12f" "$1" | sed -E 's/0+$//;s/\.$/.0/'
# }
# # 统一 lr 标签：十进制小数
# lr_tag_dec() { echo "lr$(to_decimal "$1")"; }

# # 生成 eval 列表项；comment_level: 1 -> "#", 2 -> "##"
# add_eval() {
#   local abs="$1" comment_level="${2:-1}"
#   local prefix="#"
#   [[ "$comment_level" -eq 2 ]] && prefix="##"
#   EVAL_LINES+=("$prefix ('$abs','$abs'),")
# }

# maybe_rel() { [[ $1 = /* ]] && echo "$1" || echo "../$1"; }

# ###############################################################################
# ##### 1. Base-model resolution (no jq)                                       #####
# ###############################################################################
# MODEL_DIR="" # ""→HF
# MODEL_PAIR_FILE="$WORKSPACE_DIR/examples/model_pair.json"
# BASE_MODELS=(
#   # "Qwen2.5-14B"
#   # "Qwen3-0.6B"
#   # "Qwen3-8B"
# "Qwen3-4B-Thinking-2507"
#   # "Llama3.1-8B"
#   # "Llama3.2-1B"
# )

# MODEL_PAIRS=()                  # will contain "<base>||<instruct>"

# for NAME in "${BASE_MODELS[@]}"; do
#   BLOCK=$(awk -v n="\"$NAME\"" '
#     $0~n {print; getline;
#            while ($0 !~ /\}/) {print; getline}; print; exit}' \
#     "$MODEL_PAIR_FILE")
#   [[ -z $BLOCK ]] && { echo "ERROR: '$NAME' not found in JSON"; exit 1; }

#   HF_BASE=$(printf '%s\n' "$BLOCK" | sed -n 's/.*"hf_base_path":[[:space:]]*"\([^"]*\)".*/\1/p')
#   HF_INST=$(printf '%s\n' "$BLOCK" | sed -n 's/.*"hf_instruct_path":[[:space:]]*"\([^"]*\)".*/\1/p')
#   [[ -z $HF_BASE || -z $HF_INST ]] && { echo "ERROR: malformed JSON for $NAME"; exit 1; }

#   if [[ -n $MODEL_DIR ]]; then
#       REL_BASE=${HF_BASE#*/}
#       REL_INST=${HF_INST#*/}
#       BASE_PATH="$MODEL_DIR/$REL_BASE"
#       INST_PATH="$MODEL_DIR/$REL_INST"
#       [[ -d $BASE_PATH ]] || { echo "ERROR: $BASE_PATH missing"; exit 1; }
#       [[ -d $INST_PATH ]] || { echo "ERROR: $INST_PATH missing"; exit 1; }
#   else
#       BASE_PATH=$HF_BASE
#       INST_PATH=$HF_INST
#   fi

#   MODEL_PAIRS+=("${BASE_PATH}||${INST_PATH}")
#   echo "INFO  Base=$BASE_PATH  Instruct=$INST_PATH"
# done

# ###############################################################################
# ##### 2. Generate train scripts                                              #####
# ###############################################################################
# MONTHDAY=$(date +%m%d)
# TIMESTAMP=$(date +%m%d%H%M%S)

# for PAIR in "${MODEL_PAIRS[@]}"; do
#   B_MODEL="${PAIR%%||*}"
#   I_MODEL="${PAIR##*||}"
#   MODEL_BASE=$(basename "$B_MODEL")
#   SCRIPT_FILE="$SCRIPT_OUTPUT_DIR/train_${MODEL_BASE}_${TIMESTAMP}.sh"
#   : > "$SCRIPT_FILE"            # truncate / create

#   # ================= header =================
#   {
#     echo "##### Auto-generated $(date '+%F %T') #####"
#     echo "# Model     : $MODEL_BASE"
#     echo "# LoRA mode : $USE_LORA"
#   } >> "$SCRIPT_FILE"

#   # ================= template =================
#   case "$B_MODEL" in
#     *Llama-3*)  template="llama3" ;;
#     *Qwen2*)    template="qwen" ;;
#     *Llama-2*)  template="llama2" ;;
#     *Qwen3*)    template="qwen3" ;;
#     *internlm2*)template="intern2" ;;
#     *mistral_small*) template="mistral_small" ;;
#     *Mistral*)  template="mistral" ;;
#     *Falcon*)   template="falcon" ;;
#     *gemma3*)   template="gemma3" ;;
#     *gemma*)    template="gemma" ;;
#     *Yi*)       template="yi" ;;
#     *Baichuan*) template="baichuan2" ;;
#     *) echo "ERROR: unknown template for $B_MODEL"; exit 1 ;;
#   esac
#   echo "# Template  : $template" >> "$SCRIPT_FILE"
#   echo "" >> "$SCRIPT_FILE"

#   # ================= env block =================
#   {
#     echo "##### Environment #####"
#     echo "export VLLM_WORKER_MULTIPROC_METHOD=spawn"
#     echo "export HF_HUB_OFFLINE=0"
#     echo "export HF_DATASETS_OFFLINE=0"
#     echo "export HF_DATASETS_TRUST_REMOTE_CODE=1"
#     echo "export TRUST_REMOTE_CODE=True"
#     echo "export HF_ALLOW_CODE_EVAL=1"
#     echo ""
#   } >> "$SCRIPT_FILE"

#   # ================= constants =================
#   # DATASET="s1k_gptoss20b_low"
#   # suffix_name="s1k_gptoss20b_low"
#   # cutoff_len=4096

# # s1k_gptoss20b_sharegpt_think
#   DATASET="s1k_gptoss20b_high"
#   suffix_name="s1k_gptoss20b_high"
#   cutoff_len=32768

#   # samples=(2000)
#   samples=(1000)
#   logging_steps=1; save_steps=1000; per_device_train_batch_size=2
#   gradient_accumulation_steps=16; num_train_epochs=1
#   lr_scheduler_type="cosine"; warmup_ratio=0.1; bf16=true
#   val_size=0.01; per_device_eval_batch_size=1
#   eval_strategy="steps"; eval_steps=10000; overwrite_cache=false

#   # ================= training =================
#   echo "##### Training #####" >> "$SCRIPT_FILE"

#   EVAL_LINES=()   # reset per base model

#   generate_train() {
#     local M_PATH=$1 TAG=$2 LR=$3
#     local LR_DEC LR_TAG
#     LR_DEC="$(to_decimal "$LR")"
#     LR_TAG="$(lr_tag_dec "$LR")"
#     local OUT_ROOT="$RESULTS_DIR/${MONTHDAY}/result-${MODEL_BASE}-${MONTHDAY}"

#     for MAX in "${samples[@]}"; do
#       local K; K="$(format_k "$MAX")"
#       local DIR="${TAG}-${K}-$(is_lora && echo lora-rank${lora_ranks[0]} || echo sft)-${LR_TAG}-${suffix_name}"
#       local OUTDIR="$OUT_ROOT/$DIR"

#       {
#         echo "###### ${TAG}  max=$MAX  lr=$LR_DEC ######"
#         echo "mkdir -p \"$OUTDIR\""
#         echo "cd \"$WORKSPACE_DIR\""
#         echo "llamafactory-cli train \\"
#         echo "  --model_name_or_path \"$M_PATH\" \\"
#         echo "  --stage sft \\"
#         echo "  --do_train true \\"
#         if is_lora; then
#           echo "  --finetuning_type lora --lora_rank ${lora_ranks[0]} \\"
#         else
#           echo "  --finetuning_type full \\"
#         fi
#         echo "  --deepspeed examples/deepspeed/ds_z3_config.json \\"          >> "$SCRIPT_FILE"
#         echo "  --dataset \"$DATASET\" \\"
#         echo "  --template \"$template\" \\"
#         echo "  --cutoff_len $cutoff_len \\"
#         echo "  --max_samples $MAX \\"
#         echo "  --output_dir \"$OUTDIR\" \\"
#         echo "  --per_device_train_batch_size $per_device_train_batch_size \\"
#         echo "  --gradient_accumulation_steps $gradient_accumulation_steps \\"
#         echo "  --learning_rate $LR_DEC \\"
#         echo "  --num_train_epochs $num_train_epochs \\"
#         echo "  --logging_steps $logging_steps \\"
#         echo "  --save_steps $save_steps \\"
#         echo "  --save_only_model True \\"
#         echo "  --plot_loss true \\"
#         echo "  --lr_scheduler_type $lr_scheduler_type \\"
#         echo "  --warmup_ratio $warmup_ratio \\"
#         echo "  --bf16 $bf16 \\"
#         echo "  --val_size $val_size \\"
#         echo "  --per_device_eval_batch_size $per_device_eval_batch_size \\"
#         echo "  --eval_strategy $eval_strategy \\"
#         echo "  --eval_steps $eval_steps \\"
#         echo "  --trust_remote_code True \\"
#         echo "  --flash_attn fa2 \\"
#         echo "  --overwrite_cache $overwrite_cache"
#         echo ""
#       } >> "$SCRIPT_FILE"

#       # evaluation tuple for SFT (I path only)
#       if ! is_lora && [[ $TAG == I ]]; then
#         add_eval "$OUTDIR" 1
#       fi
#     done
#   }

#   for LR in "${learning_rates[@]}"; do
#     generate_train "$B_MODEL" B "$LR"
#     generate_train "$I_MODEL" I "$LR"
#   done

#   # ================= delta-merge =================
#   if is_lora; then
#     echo "##### LoRA delta-merge #####" >> "$SCRIPT_FILE"

#     # 参数6：若传入 '#' 则命令以单井号注释；同时 evaluation 用双井号 '##'
#     merge_lora() {
#       local SRC=$1 SRC_TAG=$2 TGT=$3 TGT_TAG=$4 LR=$5 COMMENT_PREFIX=${6:-}
#       local LR_DEC LR_TAG
#       LR_DEC="$(to_decimal "$LR")"
#       LR_TAG="$(lr_tag_dec "$LR")"
#       local SRC_ROOT="$RESULTS_DIR/${MONTHDAY}/result-${MODEL_BASE}-${MONTHDAY}"

#       for RANK in "${lora_ranks[@]}"; do
#         for MAX in "${samples[@]}"; do
#           local K; K="$(format_k "$MAX")"
#           local ADAP="$SRC_ROOT/${SRC_TAG}-${K}-lora-rank${RANK}-${LR_TAG}-${suffix_name}"
#           local TAG="${SRC_TAG}2${TGT_TAG}"
#           # 合并产物目录：放在对应适配器目录下，并添加 merged-<TAG> 后缀（lr 已在父目录体现）
#           local MERGED_DIR="$ADAP/merged-${TAG}"

#           {
#             echo "mkdir -p \"$MERGED_DIR\""
#             echo "${COMMENT_PREFIX}python3 $WORKSPACE_DIR/src/shadow/merge_lora.py \\"
#             echo "${COMMENT_PREFIX}  --adapter_path \"$ADAP\" \\"
#             echo "${COMMENT_PREFIX}  --target_base \"$TGT\" \\"
#             echo "${COMMENT_PREFIX}  --merge_tag \"$TAG\" \\"
#             echo "${COMMENT_PREFIX}  --template \"$template\""
#             echo ""
#           } >> "$SCRIPT_FILE"

#           # evaluation：启用的（B2I / I2I）用单 '#'; 注释的（I2B / B2B）用双 '##'
#           if [[ -z "$COMMENT_PREFIX" ]]; then
#             add_eval "$MERGED_DIR" 1
#           else
#             add_eval "$MERGED_DIR" 2
#           fi
#         done
#       done
#     }

#     for LR in "${learning_rates[@]}"; do
#       merge_lora "$B_MODEL" B "$I_MODEL" I "$LR"          # B2I（启用，#）
#       merge_lora "$I_MODEL" I "$I_MODEL" I "$LR"          # I2I（启用，#）
#       merge_lora "$I_MODEL" I "$B_MODEL" B "$LR"  "#"  # I2B（命令 #，evaluation ##）
#       merge_lora "$B_MODEL" B "$B_MODEL" B "$LR"   "#"    # B2B（命令 #，evaluation ##）
#     done
#   else
#     echo "##### SFT delta-merge #####" >> "$SCRIPT_FILE"
#     # SFT 场景默认只做一次 B2I 的 diff 应用（基于 samples[0] 与 learning_rates[0]）
#     LR="${learning_rates[0]}"
#     LR_DEC="$(to_decimal "$LR")"
#     LR_TAG="$(lr_tag_dec "$LR")"
#     K="$(format_k "${samples[0]}")"
#     B_DIR="$RESULTS_DIR/${MONTHDAY}/result-${MODEL_BASE}-${MONTHDAY}/B-${K}-sft-${LR_TAG}-${suffix_name}"
#     MERGED_DIR="$B_DIR/merged-B2I"
#     {
#       echo "mkdir -p \"$MERGED_DIR\""
#       echo "python3 $WORKSPACE_DIR/src/shadow/apply_diff.py \\"
#       echo "  --tuned_model \"$B_DIR\" \\"
#       echo "  --target_model \"$I_MODEL\" \\"
#       echo "  --base_model \"$B_MODEL\""
#       echo ""
#     } >> "$SCRIPT_FILE"
#     add_eval "$MERGED_DIR" 1
#   fi

#   # ================= evaluation list =================
#   {
#     echo "##### Evaluation list #####"
#     for line in "${EVAL_LINES[@]}"; do
#       echo "$line"
#     done
#     echo ""
#     echo "# please copy this eval_config to opencompass/examples/eval_shadow_202505.py and then run"
#     echo ""
#     echo "cd ./opencompass"
#     echo "python3 ./run.py ./examples/eval_shadow_202505.py"
#     echo ""
#   } >> "$SCRIPT_FILE"

# done

# echo "Generation finished. Check $SCRIPT_FILE"


#!/usr/bin/env bash
###############################################################################
##### 0. Globals                                                             #####
###############################################################################
WORKSPACE_DIR="$(cd "$(dirname "$0")" && pwd)"
# WORKSPACE_DIR="/root/shadow_exp/new_shadow/Shadow"
RESULTS_DIR="$WORKSPACE_DIR/results"
SCRIPT_OUTPUT_DIR="$WORKSPACE_DIR/scripts"
mkdir -p "$RESULTS_DIR" "$SCRIPT_OUTPUT_DIR"

# --- LoRA switch -------------------------------------------------------------
USE_LORA=true                   # false -> full SFT
is_lora()   { [[ "${USE_LORA,,}" == "true" ]]; }

# --- KD switch ---------------------------------------------------------------
USE_KD=true                     # false -> standard training, true -> knowledge distillation
is_kd()     { [[ "${USE_KD,,}" == "true" ]]; }

lora_ranks=(128)
# ratios=(0.5)
learning_rates_lora=(2e-4)
# learning_rates_lora=(5e-6 1e-5 2e-5 5e-4 1e-4 2e-4 5e-4 1e-3 2e-3)
learning_rates_sft=(1e-5)

if is_lora; then
  learning_rates=("${learning_rates_lora[@]}")
else
  learning_rates=("${learning_rates_sft[@]}")
fi

# --- KD Parameters -----------------------------------------------------------
KD_RATIO=0.5                    # Knowledge distillation loss ratio
KD_TEMPERATURE=1.0              # Temperature for KD softmax
TEACHER_MODEL="meta-llama/Llama-3.1-8B-Instruct"                # Will be set based on training target

# --- Helpers -----------------------------------------------------------------
# 统一 k 表示：整千 -> 2k，非整千 -> 2.5k（保留一位小数，不使用 2.0k）
format_k() {
  local num=$1
  if (( num % 1000 == 0 )); then
    echo "$((num/1000))k"
  else
    awk -v n="$num" 'BEGIN{printf "%.1fk", n/1000}'
  fi
}
# 将科学计数法/十进制统一转为十进制小数（不使用科学计数法）
to_decimal() {
  LC_NUMERIC=C printf "%.12f" "$1" | sed -E 's/0+$//;s/\.$/.0/'
}
# 统一 lr 标签：十进制小数
lr_tag_dec() { echo "lr$(to_decimal "$1")"; }

# 生成 eval 列表项；comment_level: 1 -> "#", 2 -> "##"
# add_eval() {
#   local abs="$1" comment_level="${2:-1}"
#   local prefix="#"
#   [[ "$comment_level" -eq 2 ]] && prefix="##"
#   EVAL_LINES+=("$prefix ('$abs','$abs'),")
# }

# 生成 eval 列表项；comment_level: 1 -> "#", 2 -> "##"
add_eval() {
  local abs="$1" comment_level="${2:-1}"
  local prefix="#"; [[ "$comment_level" -eq 2 ]] && prefix="##"

  # 取最后三段作为简写（不足三段时就原样返回）
  local short
  short=$(awk -F/ '{if (NF>=3) print $(NF-2)"/"$(NF-1)"/"$NF; else print $0}' <<<"${abs%/}")

  # 左边用简写，右边用完整路径
  EVAL_LINES+=("$prefix ('$short','$abs'),")
}


maybe_rel() { [[ $1 = /* ]] && echo "$1" || echo "../$1"; }

###############################################################################
##### 1. Base-model resolution (no jq)                                       #####
###############################################################################
MODEL_DIR="" # ""→HF
MODEL_PAIR_FILE="$WORKSPACE_DIR/examples/model_pair.json"
BASE_MODELS=(
  # "Qwen2.5-14B"
  # "Qwen3-0.6B"
  # "Qwen3-8B"
# "Qwen3-4B-Thinking-2507"
  # "Llama3.1-8B"
  "Llama3.2-1B"
)

MODEL_PAIRS=()                  # will contain "<base>||<instruct>"

for NAME in "${BASE_MODELS[@]}"; do
  BLOCK=$(awk -v n="\"$NAME\"" '
    $0~n {print; getline;
           while ($0 !~ /\}/) {print; getline}; print; exit}' \
    "$MODEL_PAIR_FILE")
  [[ -z $BLOCK ]] && { echo "ERROR: '$NAME' not found in JSON"; exit 1; }

  HF_BASE=$(printf '%s\n' "$BLOCK" | sed -n 's/.*"hf_base_path":[[:space:]]*"\([^"]*\)".*/\1/p')
  HF_INST=$(printf '%s\n' "$BLOCK" | sed -n 's/.*"hf_instruct_path":[[:space:]]*"\([^"]*\)".*/\1/p')
  [[ -z $HF_BASE || -z $HF_INST ]] && { echo "ERROR: malformed JSON for $NAME"; exit 1; }

  if [[ -n $MODEL_DIR ]]; then
      REL_BASE=${HF_BASE#*/}
      REL_INST=${HF_INST#*/}
      BASE_PATH="$MODEL_DIR/$REL_BASE"
      INST_PATH="$MODEL_DIR/$REL_INST"
      [[ -d $BASE_PATH ]] || { echo "ERROR: $BASE_PATH missing"; exit 1; }
      [[ -d $INST_PATH ]] || { echo "ERROR: $INST_PATH missing"; exit 1; }
  else
      BASE_PATH=$HF_BASE
      INST_PATH=$HF_INST
  fi

  MODEL_PAIRS+=("${BASE_PATH}||${INST_PATH}")
  echo "INFO  Base=$BASE_PATH  Instruct=$INST_PATH"
done

###############################################################################
##### 2. Generate train scripts                                              #####
###############################################################################
MONTHDAY=$(date +%m%d)
TIMESTAMP=$(date +%m%d%H%M%S)

for PAIR in "${MODEL_PAIRS[@]}"; do
  B_MODEL="${PAIR%%||*}"
  I_MODEL="${PAIR##*||}"
  MODEL_BASE=$(basename "$B_MODEL")
  SCRIPT_FILE="$SCRIPT_OUTPUT_DIR/train_${MODEL_BASE}_${TIMESTAMP}.sh"
  : > "$SCRIPT_FILE"            # truncate / create

  # ================= header =================
  {
    echo "##### Auto-generated $(date '+%F %T') #####"
    echo "# Model     : $MODEL_BASE"
    echo "# LoRA mode : $USE_LORA"
    echo "# KD mode   : $USE_KD"
    if is_kd; then
      echo "# KD ratio  : $KD_RATIO"
      echo "# KD temp   : $KD_TEMPERATURE"
    fi
  } >> "$SCRIPT_FILE"

  # ================= template =================
  case "$B_MODEL" in
    *Llama-3*)  template="llama3" ;;
    *Qwen2*)    template="qwen" ;;
    *Llama-2*)  template="llama2" ;;
    *Qwen3*)    template="qwen3" ;;
    *internlm2*)template="intern2" ;;
    *mistral_small*) template="mistral_small" ;;
    *Mistral*)  template="mistral" ;;
    *Falcon*)   template="falcon" ;;
    *gemma3*)   template="gemma3" ;;
    *gemma*)    template="gemma" ;;
    *Yi*)       template="yi" ;;
    *Baichuan*) template="baichuan2" ;;
    *) echo "ERROR: unknown template for $B_MODEL"; exit 1 ;;
  esac
  echo "# Template  : $template" >> "$SCRIPT_FILE"
  echo "" >> "$SCRIPT_FILE"

  # ================= env block =================
  {
    echo "##### Environment #####"
    echo "export VLLM_WORKER_MULTIPROC_METHOD=spawn"
    echo "export HF_HUB_OFFLINE=0"
    echo "export HF_DATASETS_OFFLINE=0"
    echo "export HF_DATASETS_TRUST_REMOTE_CODE=1"
    echo "export TRUST_REMOTE_CODE=True"
    echo "export HF_ALLOW_CODE_EVAL=1"
    echo ""
  } >> "$SCRIPT_FILE"

  # ================= constants =================
  # DATASET="s1k_gptoss20b_low"
  # suffix_name="s1k_gptoss20b_low"
  # cutoff_len=4096

# # limo
#   DATASET="limo"
#   suffix_name="limo"
#   cutoff_len=4096

# limo
  DATASET="Shadow_2k"
  suffix_name="Shadow_2k"
  cutoff_len=4096
  
  # samples=(2000)
  samples=(2000)
  logging_steps=1; save_steps=1000; per_device_train_batch_size=2
  gradient_accumulation_steps=16; num_train_epochs=1
  lr_scheduler_type="cosine"; warmup_ratio=0.1; bf16=true
  val_size=0.01; per_device_eval_batch_size=1
  eval_strategy="steps"; eval_steps=10000; overwrite_cache=false

  # ================= training =================
  echo "##### Training #####" >> "$SCRIPT_FILE"

  EVAL_LINES=()   # reset per base model

  generate_train() {
    local M_PATH=$1 TAG=$2 LR=$3 IS_KD=${4:-false}
    local LR_DEC LR_TAG
    LR_DEC="$(to_decimal "$LR")"
    LR_TAG="$(lr_tag_dec "$LR")"
    local OUT_ROOT="$RESULTS_DIR/${MONTHDAY}/result-${MODEL_BASE}-${MONTHDAY}"

    # Set teacher model for KD training
    local KD_SUFFIX=""
    local TEACHER_PATH=""
    if [[ "$IS_KD" == "true" ]]; then
      KD_SUFFIX="-kd"
      if [[ "$TAG" == "B" ]]; then
        TEACHER_PATH="$I_MODEL"  # For base model, use instruct as teacher
      else
        TEACHER_PATH="$I_MODEL"  # For instruct model, use itself as teacher (self-distillation)
      fi
    fi

    for MAX in "${samples[@]}"; do
      local K; K="$(format_k "$MAX")"
      local DIR="${TAG}${KD_SUFFIX}-${K}-$(is_lora && echo lora-rank${lora_ranks[0]} || echo sft)-${LR_TAG}-${suffix_name}"
      local OUTDIR="$OUT_ROOT/$DIR"

      {
        echo "###### ${TAG}$(is_kd && [[ "$IS_KD" == "true" ]] && echo " with KD")  max=$MAX  lr=$LR_DEC ######"
        echo "mkdir -p \"$OUTDIR\""
        echo "cd \"$WORKSPACE_DIR\""
        echo "llamafactory-cli train \\"
        echo "  --model_name_or_path \"$M_PATH\" \\"
        echo "  --stage sft \\"
        echo "  --do_train true \\"
        if is_lora; then
          echo "  --finetuning_type lora --lora_rank ${lora_ranks[0]} \\"
        else
          echo "  --finetuning_type full \\"
        fi
        # echo "  --deepspeed examples/deepspeed/ds_z3_config.json \\"
        echo "  --dataset \"$DATASET\" \\"
        echo "  --template \"$template\" \\"
        echo "  --cutoff_len $cutoff_len \\"
        echo "  --max_samples $MAX \\"
        echo "  --output_dir \"$OUTDIR\" \\"
        echo "  --per_device_train_batch_size $per_device_train_batch_size \\"
        echo "  --gradient_accumulation_steps $gradient_accumulation_steps \\"
        echo "  --learning_rate $LR_DEC \\"
        echo "  --num_train_epochs $num_train_epochs \\"
        echo "  --logging_steps $logging_steps \\"
        echo "  --save_steps $save_steps \\"
        echo "  --save_only_model True \\"
        echo "  --plot_loss true \\"
        echo "  --lr_scheduler_type $lr_scheduler_type \\"
        echo "  --warmup_ratio $warmup_ratio \\"
        echo "  --bf16 $bf16 \\"
        echo "  --val_size $val_size \\"
        echo "  --per_device_eval_batch_size $per_device_eval_batch_size \\"
        echo "  --eval_strategy $eval_strategy \\"
        echo "  --eval_steps $eval_steps \\"
        echo "  --trust_remote_code True \\"
        echo "  --flash_attn fa2 \\"
        echo "  --overwrite_cache $overwrite_cache \\"
        
        # Add KD parameters if enabled
        if [[ "$IS_KD" == "true" ]]; then
          echo "  --teacher_model_name_or_path \"$TEACHER_PATH\" \\"
          echo "  --kd_ratio $KD_RATIO \\"
          # echo "  --kd_temperature $KD_TEMPERATURE \\"
        fi
        
        echo "  --use_fast_tokenizer True"
        echo ""
      } >> "$SCRIPT_FILE"

      # evaluation tuple for SFT (I path only) or KD models
      if [[ (! is_lora && "$TAG" == "I") || "$IS_KD" == "true" ]]; then
        add_eval "$OUTDIR" 1
      fi
    done
  }

  # Generate standard training commands
  echo "##### Standard Training #####" >> "$SCRIPT_FILE"
  for LR in "${learning_rates[@]}"; do
    generate_train "$B_MODEL" B "$LR" false
    generate_train "$I_MODEL" I "$LR" false
  done

  # Generate KD training commands if enabled
  if is_kd; then
    echo "" >> "$SCRIPT_FILE"
    echo "##### Knowledge Distillation Training #####" >> "$SCRIPT_FILE"
    for LR in "${learning_rates[@]}"; do
      generate_train "$B_MODEL" B "$LR" true
      generate_train "$I_MODEL" I "$LR" true
    done
  fi

  # ================= delta-merge =================
  if is_lora; then
    echo "##### LoRA delta-merge #####" >> "$SCRIPT_FILE"

    # 参数6：若传入 '#' 则命令以单井号注释；同时 evaluation 用双井号 '##'
    # 参数7：是否为KD训练的适配器
    merge_lora() {
      local SRC=$1 SRC_TAG=$2 TGT=$3 TGT_TAG=$4 LR=$5 COMMENT_PREFIX=${6:-} IS_KD_MERGE=${7:-false}
      local LR_DEC LR_TAG
      LR_DEC="$(to_decimal "$LR")"
      LR_TAG="$(lr_tag_dec "$LR")"
      local SRC_ROOT="$RESULTS_DIR/${MONTHDAY}/result-${MODEL_BASE}-${MONTHDAY}"

      # Add KD suffix to source tag if this is a KD merge
      local SRC_TAG_WITH_SUFFIX="$SRC_TAG"
      if [[ "$IS_KD_MERGE" == "true" ]]; then
        SRC_TAG_WITH_SUFFIX="${SRC_TAG}-kd"
      fi

      for RANK in "${lora_ranks[@]}"; do
        for MAX in "${samples[@]}"; do
          local K; K="$(format_k "$MAX")"
          local ADAP="$SRC_ROOT/${SRC_TAG_WITH_SUFFIX}-${K}-lora-rank${RANK}-${LR_TAG}-${suffix_name}"
          local TAG="${SRC_TAG_WITH_SUFFIX}2${TGT_TAG}"
          # 合并产物目录：放在对应适配器目录下，并添加 merged-<TAG> 后缀（lr 已在父目录体现）
          local MERGED_DIR="$ADAP/merged-${TAG}"

          {
            echo "mkdir -p \"$MERGED_DIR\""
            echo "${COMMENT_PREFIX}python3 $WORKSPACE_DIR/src/shadow/merge_lora.py \\"
            echo "${COMMENT_PREFIX}  --adapter_path \"$ADAP\" \\"
            echo "${COMMENT_PREFIX}  --target_base \"$TGT\" \\"
            echo "${COMMENT_PREFIX}  --merge_tag \"$TAG\" \\"
            echo "${COMMENT_PREFIX}  --template \"$template\""
            echo ""
          } >> "$SCRIPT_FILE"

          # evaluation：启用的（B2I / I2I）用单 '#'; 注释的（I2B / B2B）用双 '##'
          if [[ -z "$COMMENT_PREFIX" ]]; then
            add_eval "$MERGED_DIR" 1
          else
            add_eval "$MERGED_DIR" 2
          fi
        done
      done
    }

    # Standard LoRA merges
    echo "### Standard LoRA Merges ###" >> "$SCRIPT_FILE"
    for LR in "${learning_rates[@]}"; do
      merge_lora "$B_MODEL" B "$I_MODEL" I "$LR"          # B2I（启用，#）
      merge_lora "$I_MODEL" I "$I_MODEL" I "$LR"          # I2I（启用，#）
      merge_lora "$I_MODEL" I "$B_MODEL" B "$LR"  "#"     # I2B（命令 #，evaluation ##）
      merge_lora "$B_MODEL" B "$B_MODEL" B "$LR"   "#"    # B2B（命令 #，evaluation ##）
    done

    # KD LoRA merges if KD is enabled
    if is_kd; then
      echo "" >> "$SCRIPT_FILE"
      echo "### KD LoRA Merges ###" >> "$SCRIPT_FILE"
      for LR in "${learning_rates[@]}"; do
        merge_lora "$B_MODEL" B "$I_MODEL" I "$LR" "" true     # B-kd2I（启用，#）
        merge_lora "$I_MODEL" I "$I_MODEL" I "$LR" "" true     # I-kd2I（启用，#）
        merge_lora "$I_MODEL" I "$B_MODEL" B "$LR" "#" true    # I-kd2B（命令 #，evaluation ##）
        merge_lora "$B_MODEL" B "$B_MODEL" B "$LR" "#" true    # B-kd2B（命令 #，evaluation ##）
      done
    fi
  else
    echo "##### SFT delta-merge #####" >> "$SCRIPT_FILE"
    
    # Standard SFT merge
    echo "### Standard SFT Merge ###" >> "$SCRIPT_FILE"
    LR="${learning_rates[0]}"
    LR_DEC="$(to_decimal "$LR")"
    LR_TAG="$(lr_tag_dec "$LR")"
    K="$(format_k "${samples[0]}")"
    B_DIR="$RESULTS_DIR/${MONTHDAY}/result-${MODEL_BASE}-${MONTHDAY}/B-${K}-sft-${LR_TAG}-${suffix_name}"
    MERGED_DIR="$B_DIR/merged-B2I"
    {
      echo "mkdir -p \"$MERGED_DIR\""
      echo "python3 $WORKSPACE_DIR/src/shadow/apply_diff.py \\"
      echo "  --tuned_model \"$B_DIR\" \\"
      echo "  --target_model \"$I_MODEL\" \\"
      echo "  --base_model \"$B_MODEL\""
      echo ""
    } >> "$SCRIPT_FILE"
    add_eval "$MERGED_DIR" 1

    # KD SFT merge if KD is enabled
    if is_kd; then
      echo "" >> "$SCRIPT_FILE"
      echo "### KD SFT Merge ###" >> "$SCRIPT_FILE"
      B_KD_DIR="$RESULTS_DIR/${MONTHDAY}/result-${MODEL_BASE}-${MONTHDAY}/B-kd-${K}-sft-${LR_TAG}-${suffix_name}"
      MERGED_KD_DIR="$B_KD_DIR/merged-B-kd2I"
      {
        echo "mkdir -p \"$MERGED_KD_DIR\""
        echo "python3 $WORKSPACE_DIR/src/shadow/apply_diff.py \\"
        echo "  --tuned_model \"$B_KD_DIR\" \\"
        echo "  --target_model \"$I_MODEL\" \\"
        echo "  --base_model \"$B_MODEL\""
        echo ""
      } >> "$SCRIPT_FILE"
      add_eval "$MERGED_KD_DIR" 1
    fi
  fi

  # ================= evaluation list =================
  {
    echo "##### Evaluation list #####"
    for line in "${EVAL_LINES[@]}"; do
      echo "$line"
    done
    echo ""
    echo "# please copy this eval_config to opencompass/examples/eval_shadow_202505.py and then run"
    echo ""
    echo "cd ./opencompass"
    echo "python3 ./run.py ./examples/eval_shadow_202505.py"
    echo ""
  } >> "$SCRIPT_FILE"

done

echo "Generation finished. Check $SCRIPT_FILE"