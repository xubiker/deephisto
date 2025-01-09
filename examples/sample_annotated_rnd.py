"""
Example of using AnnoRegionRndSampler.
"""

import argparse
import time
from pathlib import Path

import numpy as np

from patch_samplers.region_samplers import AnnoRegionRndSampler
from utils import get_img_ano_paths


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--torch",
        action="store_true",
        help="if set, it will use torch tensor outputs",
    )
    args = parser.parse_args()

    # setup dataset
    img_anno_paths = get_img_ano_paths(
        Path("/mnt/c/dev/data/PATH-DT-MSU.WSS1"), sample="train"
    )

    # setup params
    n = 40  # number of batches to extract
    b_size = 64  # number of patches per batch
    b_per_worker = 2  # number of batches to extract per worker (parallel)

    dataset = AnnoRegionRndSampler(
        img_anno_paths,
        patch_size=224,
        layer=1,
        patches_from_one_region=4,
        one_image_for_batch=True,
    )

    t0 = time.time()
    count = np.zeros([len(dataset.classes)], dtype=np.int32)

    # generate batches with or without torch tensors
    if args.torch:
        print("Generating batches with torch tensors")
        g = dataset.torch_generator(
            batch_size=b_size,
            n_batches=n,
            batches_per_worker=b_per_worker,
        )
        for f, cls, coords in g:
            print(
                f"inputs: {f.shape}, cls: {cls.shape}, crds: {coords.shape}",
                flush=True,
            )
            q = cls.numpy().tolist()
            for cl in cls.numpy().tolist():
                count[int(cl)] += 1
    else:
        print("Generating batches of structs")
        g = dataset.structs_generator(
            batch_size=b_size, n_batches=n, batches_per_worker=b_per_worker
        )
        for batch in g:
            print(f"batch of {len(batch)} patches with coords", flush=True)
            for patch, cls in batch:
                count[cls] += 1

    t1 = time.time()

    print(f"{n * b_size / (t1 - t0)} items/s")
    print(f"patches extracted for classes: {count}")
