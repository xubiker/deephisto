# PATH-GT 

CUDA_VISIBLE_DEVICES=0 python inference/predict_annotated_rnd_gnns.py \
--patch_encoder_weights output_dir/graph-hnet-pseudo_with_another_resnet/patch_encoder_best-7.pth \
--gnn_weights output_dir/graph-hnet-pseudo_with_another_resnet/best-7.pth \
--test_path test_set_saved/WSS2_layer:2_region_area_influence:0.5_patches_from_one_region:1_region_intersection:0.5

