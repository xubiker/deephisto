from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterator

import numpy as np
from shapely import Polygon
import torch
from tqdm import tqdm

from psimage import PSImage
from psimage.patches import Patch


@dataclass
class RegionAnnotation:

    file_path: Path
    region_idx: int
    class_: str
    vertices: np.ndarray
    polygon: Polygon = None
    area: float = None

    def __init__(
        self,
        img_path: Path,
        region_idx: int,
        class_: str,
        vertices: np.ndarray,
        layer: int,
        layer_size: tuple[int, int],
        patch_size: int,
        region_intersection: float,
    ):
        self.file_path = img_path
        self.region_idx = region_idx
        self.class_ = class_
        self.vertices = vertices
        self._layer = layer
        self._layer_size = layer_size
        self._patch_size = patch_size
        self._region_intersection = region_intersection

        if len(vertices.shape) != 2 or vertices.shape[1] != 2:
            raise RuntimeError("Invalid region shape. It should be (N, 2).")
        if vertices.dtype != np.float64:
            raise RuntimeError("Invalid region dtype. It should be float64.")
        polygon = Polygon(vertices if layer == 1 else vertices.copy() / layer)
        if not polygon.is_valid:  # try to fix it
            print("invalid polygon found. Fixing...")
            polygon = polygon.buffer(0)
        area = polygon.area
        if area < patch_size * patch_size * region_intersection:
            raise RuntimeError("Region is too small.")
        self.polygon = polygon
        self.area = area

    def __str__(self) -> str:
        return (
            f"Region [{self.file_path.stem}, {self.region_idx}, "
            f"{self.class_}, {self.vertices.shape}, {round(self.area, 0)}]"
        )

    def _extract_patch_coords(
        self, n_patches: int, miss_limit: int = 500
    ) -> list[tuple[int, int]]:
        ps = self._patch_size
        h, w = self._layer_size
        x0, y0, x1, y1 = self.polygon.bounds
        res = []
        for _ in range(n_patches):
            n_miss = 0  # number of random patches not meeting the criteria
            while n_miss < miss_limit:
                x = np.random.randint(x0, min(max(x0 + 1, x1 - ps), w))
                y = np.random.randint(y0, min(max(y0 + 1, y1 - ps), h))
                patch_polygon = Polygon(
                    [
                        (x, y),
                        (x + ps, y),
                        (x + ps, y + ps),
                        (x, y + ps),
                    ]
                )
                ia = self.polygon.intersection(patch_polygon).area
                if ia > ps * ps * self._region_intersection:
                    res.append((y, x))
                    break
                else:
                    n_miss += 1
            if n_miss >= miss_limit:
                raise RuntimeError(
                    "Miss limit reached. Probably region is too small."
                )
        return res


class RegionRandomBatchedDataset:

    def __init__(
        self,
        img_anno_paths: list[tuple[Path, Path]],
        layer: int,
        patch_size: int,
        batch_size: int,
        region_intersection: float = 0.75,
        patches_from_one_region: int = 4,
        region_area_influence: float = 0.5,
        classes: list[str] = None,
    ):
        self.img_anno_paths = img_anno_paths
        self.layer = layer
        self.patch_size = patch_size
        self.batch_size = batch_size
        self.region_intersection = region_intersection
        self.patches_from_one_region = patches_from_one_region
        self.classes = classes
        self._parse_annotations()
        self._calc_region_weights(region_area_influence)
        super().__init__()

    def _parse_annotations(self) -> None:
        self._regions = dict()
        regions_failed = 0
        for psim_path, anno_path in tqdm(
            self.img_anno_paths, "parsing annotations"
        ):
            with PSImage(psim_path) as psim:
                with open(anno_path) as anno_f:
                    for i, a in enumerate(json.load(anno_f)):
                        cls = a["class"]
                        if (
                            self.classes is not None
                            and cls not in self.classes
                        ):
                            continue
                        try:
                            reg = RegionAnnotation(
                                img_path=psim_path,
                                region_idx=i,
                                class_=cls,
                                vertices=np.array(
                                    a["vertices"], dtype=np.float64
                                ),
                                layer=self.layer,
                                layer_size=psim.layer_size(self.layer),
                                patch_size=self.patch_size,
                                region_intersection=self.region_intersection,
                            )
                            if cls not in self._regions:
                                self._regions[cls] = [reg]
                            else:
                                self._regions[cls].append(reg)
                        except Exception:
                            regions_failed += 1
        if regions_failed > 0:
            print(f"Failed to parse {regions_failed} regions.")
        self.classes = sorted(list(self._regions.keys()))
        self._areas_per_cls = {
            cls: sum([i.area for i in regions])
            for cls, regions in self._regions.items()
        }
        print("Total area:")
        for cls in self._areas_per_cls:
            size_gpx = round(self._areas_per_cls[cls] / 1e9, 2)
            size_prc = round(
                self._areas_per_cls[cls]
                / sum(self._areas_per_cls.values())
                * 100,
                2,
            )
            print(f"\t{cls}: {size_gpx} Gpx ({size_prc}%)")

    def _calc_region_weights(self, area_influence: float) -> None:
        assert -1 <= area_influence <= 1
        self._weights = dict()
        for cls, regions in self._regions.items():
            areas = [r.area for r in regions]
            areas_inv = [1 / a for a in areas]
            w_proportional = np.array(areas) / sum(areas)
            w_inv_proportional = np.array(areas_inv) / sum(areas_inv)
            w_default = np.ones(len(regions), dtype=np.float64) / len(regions)

            if area_influence == 0:
                w = w_default
            elif area_influence > 0:
                delta = (w_proportional - w_default) * area_influence
                w = w_default + delta
                w = w / sum(w)
            elif area_influence < 0:
                delta = (w_inv_proportional - w_default) * (-area_influence)
                w = w_default + delta
                w = w / sum(w)

            self._weights[cls] = w

    def _patches_one_region(
        self, region: RegionAnnotation, n: int
    ) -> list[Patch]:
        with PSImage(region.file_path) as psim:
            try:
                coords = region._extract_patch_coords(n)
                return [
                    Patch(
                        self.layer,
                        pos_x=c[1],
                        pos_y=c[0],
                        patch_size=self.patch_size,
                        data=psim.get_region_from_layer(
                            self.layer,
                            c,
                            (
                                c[0] + self.patch_size,
                                c[1] + self.patch_size,
                            ),
                        ),
                    )
                    for c in coords
                ]
            except Exception:
                pass

    def _patches_single_proc(self, n: int) -> list[tuple[Patch, int]]:
        res = []
        while len(res) < n:
            try:
                cls_idx = np.random.randint(len(self.classes))
                cls = self.classes[cls_idx]
                region: RegionAnnotation = np.random.choice(
                    self._regions[cls],
                    p=self._weights[cls],
                )
                k = min(self.patches_from_one_region, n - len(res))
                res.extend(
                    [(p, cls_idx) for p in self._patches_one_region(region, k)]
                )
            except Exception:
                continue
        return res

    def _patches_single_proc_torch(
        self, n_patches: int, k: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        f, l, c = [], [], []
        for _ in range(k):
            r = self._patches_single_proc(n_patches)
            features = torch.Tensor(np.stack([p.data for p, i in r]))
            labels = torch.Tensor([i for p, i in r])
            coords = torch.Tensor(
                np.stack([np.array([p.pos_y, p.pos_x]) for p, i in r])
            )
            f.append(features)
            l.append(labels)
            c.append(coords)
        return f, l, c

    def generator(
        self, n_batches: int, max_workers: int = None
    ) -> Iterator[tuple[Patch, int]]:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(self._patches_single_proc, self.batch_size)
                for _ in range(n_batches)
            ]
            for future in futures:
                r = future.result()
                yield r

    def generator_torch(
        self,
        n_batches: int,
        transforms: callable = None,
        max_workers: int = None,
        factor: int = 2,
    ) -> Iterator[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(
                    self._patches_single_proc_torch, self.batch_size, factor
                )
                for _ in range(n_batches // factor)
            ]
            for future in futures:
                f, l, c = future.result()
                for ff, ll, cc in zip(f, l, c):
                    if transforms is not None:
                        ff = transforms(ff)
                    yield (ff, ll, cc)

    def __len__(self):
        ps = self.patch_size * self.layer
        return int(sum(self._areas_per_cls.values()) / (ps * ps))
