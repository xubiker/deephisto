CUDA_VISIBLE_DEVICES=0 python inference/predict_annotated_rnd_gnns.py \
--patch_encoder_weights /home/z.sun/deephisto/output_dir/graph-hnet-pseudo_baseline/patch_encoder_best-9.pth \
--gnn_weights /home/z.sun/deephisto/output_dir/graph-hnet-pseudo_baseline/best-9.pth \
--test_path patches_test