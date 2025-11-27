#!/bin/bash

# 定义模型路径和输出目录
PATCH_ENCODER_PATH="output_dir/graph-hnet-pseudo_with_another_resnet/patch_encoder_best-7.pth"
GNN_PATH="output_dir/graph-hnet-pseudo_with_another_resnet/best-7.pth"
OUTPUT_DIR="./output"

for i in {01..05}; do
    INPUT_PATH="/home/data_repository/PATH-DT-MSU_dev/WSS2/images/test_${i}.psi"
    OUTPUT_SUBDIR="${OUTPUT_DIR}/test_gnn_graph-hnet-pseudo_baseline_${i}"

    python inference/predict_full_patched_gnns.py \
        --img_path "$INPUT_PATH" \
        --patch_encoder_path "$PATCH_ENCODER_PATH" \
        --gnn_path "$GNN_PATH" \
        --out_dir "$OUTPUT_SUBDIR" &

    echo "Started processing test_${i}.psi with output in ${OUTPUT_SUBDIR}"
done

wait
echo "All predictions completed!"