from pathlib import Path
import time

import numpy as np

from patch_samplers.region_samplers import RegionRandomBatchedDataset


if __name__ == "__main__":
    ds_path = Path("/Users/xubiker/dev/PATH-DT-MSU.WSS1")
    sample = "train"
    img_paths = [
        p
        for p in (ds_path / "images" / sample).iterdir()
        if p.is_file() and p.suffix == ".psi"
    ]
    anno_paths = [
        ds_path / "annotations" / sample / f"{p.stem}.json" for p in img_paths
    ]

    img_anno_paths = list(zip(img_paths, anno_paths))

    n = 40
    batch_size = 64

    dataset = RegionRandomBatchedDataset(
        img_anno_paths,
        patch_size=224,
        batch_size=batch_size,
        layer=1,
        patches_from_one_region=8,
    )
    g = dataset.generator_torch(n, factor=2)
    t0 = time.time()
    count = np.zeros([len(dataset.classes)])
    for f, cls, coords in g:
        print(f.shape, cls.shape, coords.shape)
        q = cls.numpy().tolist()
        for cl in cls.numpy().tolist():
            count[int(cl)] += 1

    print(count)
    t1 = time.time()
    print(f"{n * batch_size / (t1 - t0)} it/s")
