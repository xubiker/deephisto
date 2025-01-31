from pathlib import Path
from patch_samplers.region_samplers import AnnoRegionRndSampler
from tqdm import tqdm
import numpy as np 
def get_img_ano_paths(ds_folder: Path, sample: str = "train",version = 'v2.5'):
    img_paths = [
        p
        for p in (ds_folder / sample).iterdir()
        if p.is_file() and p.suffix == ".psi"
    ]
    anno_paths = [
        ds_folder / sample / f"{p.stem}_{version}.json"
        for p in img_paths
    ]
    return list(zip(img_paths, anno_paths))

def extract_and_save_tests_gnns(img_anno_paths: list[tuple[Path, Path]],
    out_folder: Path,
    patch_size: int,
    layer: int,
    n: int,
    intersection=0.95,
    graph_size=8,
):
    dataset = AnnoRegionRndSampler(
        img_anno_paths,
        patch_size=patch_size,
        layer=layer,
        patches_from_one_region=4,
        one_image_for_batch=True,
        normalize_pos=True
    )
    out_folder.mkdir(parents=True, exist_ok=True)
    b_per_worker = 2  # number of batches to extract per worker (parallel)
    patch_sampler = dataset.torch_generator(
            batch_size=graph_size*graph_size,
            n_batches=n,
            batches_per_worker=b_per_worker,
        )
    progress_bar = tqdm(total= n, desc="Predicting", unit="step")
    progress = 0
    for f, cls, coords,img_path in patch_sampler:
        y_true = cls.numpy().tolist()
        features = f.permute(0, 3, 1, 2).contiguous()
        step_file_path = out_folder / f"step_{progress}.npz"
        np.savez(step_file_path, features=features, y_true=y_true, coords=coords,img_path=img_path)
        print(coords)
        progress+=1
        progress_bar.n = round(progress, 2)
        progress_bar.refresh()
        if progress == n:
            break