"""
Example of using AnnoRegionDenseSampler.
"""

import time
from pathlib import Path

import numpy as np

from patch_samplers.region_samplers import AnnoRegionDenseSampler
from utils import get_img_ano_paths


if __name__ == "__main__":

    # setup dataset
    img_anno_paths = get_img_ano_paths(
        Path("/mnt/c/dev/data/PATH-DT-MSU.WSS2"), sample="test"
    )

    # setup params
    b_size = 64  # number of patches per batch

    dataset = AnnoRegionDenseSampler(
        img_anno_paths,
        patch_size=224,
        stride=112,
        layer=1,
        # classes=["AT", "MM"],
    )

    t0 = time.time()
    count = np.zeros([len(dataset.classes)], dtype=np.int32)

    # generate batches with or without torch tensors
    print("Generating batches of structs")
    g = dataset.structs_generator()
    for i, (patch, cls) in enumerate(g):
        # print(f"{i}. {patch.data.shape}, {cls}")
        count[cls] += 1

    t1 = time.time()

    print(f"Total patches: {np.sum(count)}")
    print(f"{np.sum(count) / (t1 - t0)} items/s")
    print(f"patches extracted for classes: {count}")
