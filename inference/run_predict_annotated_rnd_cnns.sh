CUDA_VISIBLE_DEVICES=0 python inference/predict_annotated_rnd_cnns.py \
    --data_dir /home/data_repository/PATH-DT-MSU_dev/WSS2_v2_psi \
    --model_path /home/z.sun/graph-wsi/pretrained_encoder/resnet-50-best-7.pth \
    --patches_dir test_set_saved/patches_test_1.0_10_1.0

#densenet121
CUDA_VISIBLE_DEVICES=0 python inference/predict_annotated_rnd_cnns.py \
    --data_dir /home/data_repository/PATH-DT-MSU_dev/WSS2_v2_psi \
    --model_path /home/z.sun/wsi_SR_CL/output_dir/v2.5_20x_densenet_bc_0.1_pretrain/best-18.pth \
    --patches_dir test_set_saved/patches_test_1.0_10_1.0 \
    --model_type densenet121

#mobilenet_v2
CUDA_VISIBLE_DEVICES=0 python inference/predict_annotated_rnd_cnns.py \
    --data_dir /home/data_repository/PATH-DT-MSU_dev/WSS2_v2_psi \
    --model_path /home/z.sun/wsi_SR_CL/output_dir/v2.5_20x_mobilenet_bc_0.1/best-12.pth \
    --patches_dir test_set_saved/patches_test_1.0_10_1.0 \
    --model_type mobilenet_v2

#Efficient Net
CUDA_VISIBLE_DEVICES=0 python inference/predict_annotated_rnd_cnns.py \
    --data_dir /home/data_repository/PATH-DT-MSU_dev/WSS2_v2_psi \
    --model_path /home/z.sun/wsi_SR_CL/output_dir/v2.5_20x_efficientnet_v2_s_bc_0.1_pretrain/best-36.pth \
    --patches_dir test_set_saved/patches_test_1.0_10_1.0 \
    --model_type Efficient_Net