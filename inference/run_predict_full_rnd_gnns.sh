#!/bin/bash

# 定义模型路径和输出目录
PATCH_ENCODER_PATH="/home/z.sun/deephisto/output_dir/graph-hnet-pseudo_baseline/patch_encoder_best-9.pth"
GNN_PATH="/home/z.sun/deephisto/output_dir/graph-hnet-pseudo_baseline/best-9.pth"
OUTPUT_DIR="./output"

# 循环运行 5 个 Python 脚本
for i in {01..05}; do
    INPUT_PATH="/home/data_repository/PATH-DT-MSU_dev/WSS2_v2_psi/test/test_${i}.psi"
    OUTPUT_SUBDIR="${OUTPUT_DIR}/test_gnn_graph-hnet-pseudo_baseline_${i}"

    # 运行 Python 脚本
    python inference/predict_full_patched_gnns.py \
        --img_path "$INPUT_PATH" \
        --patch_encoder_path "$PATCH_ENCODER_PATH" \
        --gnn_path "$GNN_PATH" \
        --out_dir "$OUTPUT_SUBDIR" &

    # 打印运行信息
    echo "Started processing test_${i}.psi with output in ${OUTPUT_SUBDIR}"
done

# 等待所有后台任务完成
wait
echo "All predictions completed!"