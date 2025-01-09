"""
Example of preparing patches for test set.
"""

from pathlib import Path
from patch_samplers.region_samplers import extract_and_save_subset
from utils import get_img_ano_paths


if __name__ == "__main__":

    img_anno_paths_test = get_img_ano_paths(
        ds_folder=Path("/mnt/c/dev/data/PATH-DT-MSU.WSS1"), sample="test"
    )

    out_dir = Path("/mnt/c/dev/data/PATH-DT-MSU.WSS1/patches_test")

    extract_and_save_subset(
        img_anno_paths=img_anno_paths_test,
        out_folder=out_dir,
        patch_size=224,
        layer=2,
        patches_per_class=100,
    )
