"""
Example of using AnnotatedRegionSampler.
"""

from pathlib import Path
import time

import numpy as np

from patch_samplers.region_samplers import AnnotatedRegionSampler
import argparse


def parse_dataset(ds_folder: Path, sample: str) -> list[tuple[Path, Path]]:
    """
    Parse the dataset and return a list of tuples (img_path, anno_path).
    """
    img_paths = [
        p
        for p in (ds_folder / "images" / sample).iterdir()
        if p.is_file() and p.suffix == ".psi"
    ]
    anno_paths = [
        ds_folder / "annotations" / sample / f"{p.stem}.json"
        for p in img_paths
    ]
    return list(zip(img_paths, anno_paths))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--torch",
        action="store_true",
        help="if set, it will use torch tensor outputs",
    )
    args = parser.parse_args()

    # setup dataset
    img_anno_paths = parse_dataset(
        Path("/Users/xubiker/dev/PATH-DT-MSU.WSS1"), sample="train"
    )

    # setup params
    n = 40  # number of batches to extract
    batch_size = 64  # number of patches per batch
    b_per_worker = 2  # number of batches to extract per worker (parallel)

    dataset = AnnotatedRegionSampler(
        img_anno_paths,
        patch_size=224,
        batch_size=batch_size,
        layer=1,
        patches_from_one_region=8,
    )

    t0 = time.time()
    count = np.zeros([len(dataset.classes)], dtype=np.int32)

    # generate batches with or without torch tensors
    if args.torch:
        print("Generating batches with torch tensors")
        g = dataset.generator_torch(n, batches_per_worker=b_per_worker)
        for f, cls, coords in g:
            print(
                f"inputs: {f.shape}, cls: {cls.shape}, crds: {coords.shape}",
                flush=True,
            )
            q = cls.numpy().tolist()
            for cl in cls.numpy().tolist():
                count[int(cl)] += 1
    else:
        print("Generating batches without torch tensors")
        g = dataset.generator(n, batches_per_worker=b_per_worker)
        for batch in g:
            print(f"batch of {len(batch)} patches with coords", flush=True)
            for patch, cls in batch:
                count[cls] += 1

    t1 = time.time()

    print(f"{n * batch_size / (t1 - t0)} it/s")
    print(f"patches extracted for classes: {count}")
