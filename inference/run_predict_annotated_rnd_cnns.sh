CUDA_VISIBLE_DEVICES=0 python inference/predict_annotated_rnd_cnns.py \
    --model_type resnet50 \
    --model_path output_dir/resnet/patch_encoder_best-44.pth \
    --patches_dir test_set_saved/WSS2_layer:2_region_area_influence:0_patches_from_one_region:1_region_intersection:0.8
