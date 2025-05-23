# DeepHisto

Repository, contatining a set of methods, tools and models for histological image analysis developed at [MMIP lab](http://imaging.cs.msu.ru).

This readme is not complete yet. Sorry...

## Installation

(Not ready)

Create env:
```conda env create -n deephisto -f environment.yml```

## Run examples

The examples of using different methods can be found in [examples folder](/examples/).

Use  patch samplers to extract patches from annotated regions. Can be used for training classification models.
`python -m examples.sample_annotated_dense`
`python -m examples.sample_annotated_rnd`
`python -m examples.sample_annotated_rnd --torch`

Use patch samplers to extract patches from whole image. Can be random or dense. Is usefull for predicting for the whole image.
`python -m examples.sample_full_dense`
`python -m examples.sample_full_random`

Train simlpe patch-based model:
`python -m models.patch_cls_simple.train`
`python -m models.patch_cls_simple.train --extract_test`

Make full prediction on WSI image using saved model:
`python -m examples.predict_full_patched`

# PATH-GT 
## step 1, prepare env and data 
### download psimage [text](https://github.com/xubiker/psimage?tab=readme-ov-file)

## step 2, extract data for evaluation and test
Extract Twice , one for evaluation and one for test with different parameters 
python extract_testset.py \
--dataset_path /home/data_repository/PATH-DT-MSU_dev/WSS2 \ 
--region_intersection 0.5 \
--version _v2 \
--patches_from_one_region 1 \
--region_area_influence 0.5 \
--layer 2

## step 3, train baseline model (ResNet50)
CUDA_VISIBLE_DEVICES=0 python main_train.py \
--resnet50 \
--model NOGNN \
--runs_name resnet \
--lr 1e-4 --smoothing 0.1 \
--train_data_path /home/data_repository/PATH-DT-MSU_dev/WSS2 \
--test_output_path test_set_saved/WSS2_layer:2_region_area_influence:0_patches_from_one_region:1_region_intersection:0.8 

## Optional : Inference on Test 
```inference/run_predict_annotated_rnd_cnns.sh```
## step 3, train model PATH-GT 

CUDA_VISIBLE_DEVICES=0 python main_train.py \
--resnet50 --runs_name graph-hnet-pseudo_with_another_resnet \
--model graph-hnet-pseudo \
--finetune /home/z.sun/graph-wsi/pretrained_encoder/resnet-50-best-7.pth \
--lr 1e-5 \
--train_data_path /home/data_repository/PATH-DT-MSU_dev/WSS2 \
--test_output_path test_set_saved/WSS2_layer:2_region_area_influence:0_patches_from_one_region:1_region_intersection:0.8 

## Optional : Inference on Test 
```inference/predict_annotated_rnd_gnns.py```

## step 4, inference for WSI 

Change inference/run_predict_full_rnd_gnns.sh and run it 
```./inference/run_predict_full_rnd_gnns.sh```
