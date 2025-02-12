# PAG former
CUDA_VISIBLE_DEVICES=0 python inference/predict_annotated_rnd_gnns.py \
--patch_encoder_weights output_dir/graph-hnet-pseudo_k=7_1024_layer1_512/patch_encoder_best-10.pth \
--gnn_weights output_dir/graph-hnet-pseudo_k=7_1024_layer1_512/best-10.pth \
--test_path test_set_saved/patches_test_1.0_10_1.0

#SG Former 
CUDA_VISIBLE_DEVICES=0 python inference/predict_annotated_rnd_gnns.py \
--patch_encoder_weights output_dir/graph-hnet-pseudo_k=3_1024_layer3_512_nopesudo/patch_encoder_best-0.pth \
--gnn_weights output_dir/graph-hnet-pseudo_k=3_1024_layer3_512_nopesudo/best-0.pth \
--test_path test_set_saved/patches_test_1.0_10_1.0

# GraphSAGE
CUDA_VISIBLE_DEVICES=0 python inference/predict_annotated_rnd_gnns.py \
--patch_encoder_weights output_dir/GraphSAGE/patch_encoder_best-1.pth \
--gnn_weights output_dir/GraphSAGE/best-1.pth \
--test_path test_set_saved/patches_test_1.0_10_1.0

#EdgeConv2d
CUDA_VISIBLE_DEVICES=0 python inference/predict_annotated_rnd_gnns.py \
--patch_encoder_weights output_dir/EdgeConv2d/patch_encoder_best-4.pth \
--gnn_weights output_dir/EdgeConv2d/best-4.pth \
--test_path test_set_saved/patches_test_1.0_10_1.0

#MRConv2d
CUDA_VISIBLE_DEVICES=0 python inference/predict_annotated_rnd_gnns.py \
--patch_encoder_weights /home/z.sun/deephisto/output_dir/MRConv2d/patch_encoder_best-4.pth \
--gnn_weights /home/z.sun/deephisto/output_dir/MRConv2d/best-4.pth \
--test_path test_set_saved/patches_test_1.0_10_1.0


# K = 1
CUDA_VISIBLE_DEVICES=0 python inference/predict_annotated_rnd_gnns.py \
--patch_encoder_weights output_dir/graph-hnet-pseudo_k=1/patch_encoder_best-6.pth \
--gnn_weights output_dir/graph-hnet-pseudo_k=1/best-6.pth \
--test_path test_set_saved/patches_test_1.0_10_1.0

# K = 3
CUDA_VISIBLE_DEVICES=0 python inference/predict_annotated_rnd_gnns.py \
--patch_encoder_weights output_dir/graph-hnet-pseudo_k=3_1024_layer1_512/patch_encoder_best-14.pth \
--gnn_weights output_dir/graph-hnet-pseudo_k=3_1024_layer1_512/best-14.pth \
--test_path test_set_saved/patches_test_1.0_10_1.0

# K = 5
CUDA_VISIBLE_DEVICES=0 python inference/predict_annotated_rnd_gnns.py \
--patch_encoder_weights output_dir/graph-hnet-pseudo_k=5_1024_layer1_512/patch_encoder_best-4.pth \
--gnn_weights output_dir/graph-hnet-pseudo_k=5_1024_layer1_512/best-4.pth \
--test_path test_set_saved/patches_test_1.0_10_1.0

# K = 9
CUDA_VISIBLE_DEVICES=0 python inference/predict_annotated_rnd_gnns.py \
--patch_encoder_weights output_dir/graph-hnet-pseudo_k=9/patch_encoder_best-2.pth \
--gnn_weights output_dir/graph-hnet-pseudo_k=9/best-2.pth \
--test_path test_set_saved/patches_test_1.0_10_1.0