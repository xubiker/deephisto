import os
from pathlib import Path
import argparse
from utils import get_img_ano_paths,extract_and_save_tests_gnns
from copy import deepcopy
import os 

parser = argparse.ArgumentParser(description="Extract patches from dataset.")
parser.add_argument("--dataset_path", type=str, required=True, help="Path to the dataset.")
parser.add_argument("--version", type=str, default="", help="Version of the dataset.")
parser.add_argument("--region_intersection", type=float, default=0.8, help="Region intersection threshold.")
parser.add_argument("--patches_from_one_region", type=int, default=1, help="Number of patches from one region.")
parser.add_argument("--region_area_influence", type=float, default=0.5, help="Region area influence.")
parser.add_argument("--layer", type=int, default=4, help="Layer number.")
parser.add_argument("--out_dir", type=str, default="test_set_saved", help="Output directory for saving patches.")
args = parser.parse_args()


dataset_path = args.dataset_path
version = args.version
region_intersection = args.region_intersection
patches_from_one_region = args.patches_from_one_region
region_area_influence = args.region_area_influence
layer = args.layer
save_subfolder_name = args.dataset_path.split('/')[-1]
out_dir = Path("test_set_saved") / f"{save_subfolder_name}_layer:{layer}_region_area_influence:{region_area_influence}_patches_from_one_region:{patches_from_one_region}_region_intersection:{region_intersection}"

img_anno_paths_test = get_img_ano_paths(
    ds_folder=Path(dataset_path), sample="test", version=version
)

if not os.path.exists(out_dir):
    extract_and_save_tests_gnns(
        img_anno_paths=img_anno_paths_test,
        out_folder=out_dir,
        patch_size=224,
        layer=layer,
        n=40,
        region_intersection=region_intersection,
        patches_from_one_region=patches_from_one_region,
        region_area_influence=region_area_influence,
        graph_size=8,
    )