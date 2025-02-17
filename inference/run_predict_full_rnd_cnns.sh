#!/bin/bash

# 定义模型路径和输出目录
MODEL_PATH="/home/z.sun/graph-wsi/pretrained_encoder/resnet-50-best-7.pth"
OUTPUT_DIR="./output"

# 最大并行任务数
MAX_JOBS=2

for i in {01..05}; do
    INPUT_PATH="/home/data_repository/PATH-DT-MSU_dev/WSS2_v2_psi/test/test_${i}.psi"
    OUTPUT_SUBDIR="${OUTPUT_DIR}/test_${i}"
    python inference/predict_full_patched_cnns.py --img_path "$INPUT_PATH" --model_path "$MODEL_PATH" --out_dir "$OUTPUT_SUBDIR" &

    if [[ $(jobs -r -p | wc -l) -ge $MAX_JOBS ]]; then
        wait -n
    fi

    echo "Started processing test_${i}.psi with output in ${OUTPUT_SUBDIR}"
done

# 等待所有后台任务完成
wait
echo "All predictions completed!"