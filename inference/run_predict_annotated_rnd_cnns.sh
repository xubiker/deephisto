CUDA_VISIBLE_DEVICES=0 python inference/predict_annotated_rnd_cnns.py \
    --data_dir /home/data_repository/PATH-DT-MSU_dev/WSS2_v2_psi \
    --model_path /home/z.sun/graph-wsi/pretrained_encoder/resnet-50-best-7.pth \
    --patches_dir /home/z.sun/graph-wsi/Test_data/NEW-test_split_v1_64_layer2_coeff1.0_intersection0.8
