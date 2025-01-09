from pathlib import Path


def get_img_ano_paths(ds_folder: Path, sample: str = "train"):
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
