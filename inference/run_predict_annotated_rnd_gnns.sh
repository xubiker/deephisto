CUDA_VISIBLE_DEVICES=0 python inference/predict_annotated_rnd_gnns.py \
--patch_encoder_weights output_dir/graph-hnet-pseudo_k=7_2048/patch_encoder_best-14.pth \
--gnn_weights output_dir/graph-hnet-pseudo_k=7_2048/best-14.pth \
--test_path test_set_saved/patches_test_1.0_10_1.0