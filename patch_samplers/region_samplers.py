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
    ):
        """
        Create a RegionAnnotation object.

        Args:
            img_path : Path
                Path to the image that contains the region.
            region_idx : int
                Index of the region in the annotation.
            class_ : str
                Class label of the region.
            vertices : np.ndarray
                Vertices of the region.
            layer : int
                Layer of the image where the region is located.
            layer_size : tuple[int, int]
                Size of the layer.
            patch_size : int
                Size of the patches to be extracted from the region.

        Raises:
            RuntimeError: If the region has invalid shape or dtype.
        """
        self.file_path = img_path
        self.region_idx = region_idx
        self.class_ = class_
        self.vertices = vertices
        self._layer = layer
        self._layer_size = layer_size
        self._patch_size = patch_size

        if len(vertices.shape) != 2 or vertices.shape[1] != 2:
            raise RuntimeError("Invalid region shape. It should be (N, 2).")
        if vertices.dtype != np.float64:
            raise RuntimeError("Invalid region dtype. It should be float64.")
        polygon = Polygon(vertices if layer == 1 else vertices.copy() / layer)
        if not polygon.is_valid:  # try to fix it
            print("invalid polygon found. Fixing...")
            polygon = polygon.buffer(0)
        self.polygon = polygon
        self.area = polygon.area

    def __str__(self) -> str:
        """Returns a string representation of RegionAnnotation."""
        return (
            f"Region [{self.file_path.stem}, {self.region_idx}, "
            f"{self.class_}, {self.vertices.shape}, {round(self.area, 0)}]"
        )

    def _extract_patch_coords(
        self,
        n_patches: int,
        region_intersection: float = 0.75,
        miss_limit: int = 500,
    ) -> list[tuple[int, int]]:
        """
        Extracts coordinates for patches within the region.

        This method attempts to extract a specified number of patch
        coordinates within the defined region. Each patch must satisfy
        a minimum intersection area with the region polygon. The method
        employs a random search approach and limits the number of failed
        attempts to find suitable patches.

        Args:
            n_patches (int): Number of patch coordinates to generate.
            region_intersection (float): Minimum intersection area ratio
                required between the patch and the region. Defaults to 0.75.
            miss_limit (int): Maximum number of unsuccessful attempts to
                find a suitable patch. Defaults to 500.

        Returns:
            list[tuple[int, int]]: A list of coordinates (y, x) for each
            extracted patch.

        Raises:
            RuntimeError: If the region is too small or the miss limit is
            reached without finding enough suitable patches.
        """

        ps = self._patch_size
        h, w = self._layer_size
        x0, y0, x1, y1 = self.polygon.bounds
        if self.area < ps * ps * region_intersection:
            raise RuntimeError("Region is too small.")
        coords = []
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
                if ia > ps * ps * region_intersection:
                    coords.append((y, x))
                    break
                else:
                    n_miss += 1
            if n_miss >= miss_limit:
                raise RuntimeError(
                    "Miss limit reached. Probably region is too small."
                )
        return coords


class AnnotatedRegionSampler:

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
        """
        Parse annotation files and extract regions.

        This method parses annotation files and for each one extracts regions
        and stores them in self._regions. It also calculates the total area of
        each class and stores it in self._areas_per_cls.

        Returns: None
        """
        self._regions = dict()
        regions_failed = 0
        for psim_path, anno_path in tqdm(
            self.img_anno_paths, "Parsing annotations"
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
        print("Total area per class:")
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
        """
        Calculate weights for each region based on their area and
        influence factor.

        Args:
            area_influence: Influence of region area on weights. If 0, equal
            weights are assigned to all regions. If > 0, larger regions get
            more weight. If < 0, smaller regions get more weight.

        Returns: None
        """
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
        """
        Extract patches from a region.

        Given a region, extract n patches from it using its internal
        random generator. The patches are extracted in the order of
        their y and x coordinates.

        Args:
            region (RegionAnnotation): Annotation of the region.
            n (int): Number of patches to extract.

        Returns:
            list[Patch]: Patches extracted from the region.
        """
        with PSImage(region.file_path) as psim:
            try:
                coords = region._extract_patch_coords(
                    n, region_intersection=self.region_intersection
                )
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

    def _gen_single_proc(self, n: int) -> list[tuple[Patch, int]]:
        """
        Generate patches from regions for one process.

        Generate n patches from all regions, using the weights calculated
        in _calc_region_weights. The patches are generated one by one, and
        the process is repeated until n patches are generated.

        Args:
            n (int): Number of patches to generate.

        Returns:
            list[tuple[Patch, int]]: List of tuples containing a Patch and its
            class index.
        """
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

    def _gen_single_proc_torch(
        self, n: int
    ) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Generate a list of tensors containing patch data, class indices,
        and coordinates.

        This function generates a specified number of patches using the
        internal `_gen_single_proc` method. Each patch is converted into a
        tuple of tensors containing the patch's normalized data, its class
        index, and its (y, x) coordinates.

        Args:
            n (int): Number of patches to generate.

        Returns:
            list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]: A list of
            tuples, each containing tensors for the patch's features, label,
            and coordinates.
        """

        res = []
        series = self._gen_single_proc(n)
        for patch, idx in series:
            features = torch.tensor(patch.data, dtype=torch.float32) / 255
            labels = torch.tensor(idx, dtype=torch.float32)
            coords = torch.tensor(
                [patch.pos_y, patch.pos_x], dtype=torch.float32
            )
            res.append((features, labels, coords))
        return res

    def _split_chunks(self, n, k):
        """
        Split n items into chunks of size k, with the last chunk potentially
        being smaller.

        Args:
            n (int): Number of items to split.
            k (int): Size of each chunk.

        Returns:
            list[int]: List of chunk sizes.
        """
        q = [k] * (n // k)
        if n % k > 0:
            q.append(n % k)
        return q

    def generator(
        self,
        n_batches: int,
        batches_per_worker: int = 2,
        max_workers: int = None,
    ) -> Iterator[tuple[Patch, int]]:
        """
        Generate batches of patches and labels in parallel.

        This method uses the internal `_gen_single_proc` method to generate
        batches of patches and labels. The required number of batches is split
        between worker processes, which are executed in parallel using a
        `ProcessPoolExecutor`. The results are then yielded in chunks of
        `self.batch_size`.

        Args:
            n_batches (int): Number of batches to generate.
            batches_per_worker (int): Number of batches to generate per worker
                process.
            max_workers (int): Maximum number of worker processes to spawn.
                Defaults to `None`, which means that the number of workers is
                determined by the `ProcessPoolExecutor`.

        Yields:
            tuple[Patch, int]: A tuple containing a list of patches and their
                corresponding labels.
        """
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # split required number of batches between workers
            q = self._split_chunks(n_batches, batches_per_worker)
            futures = [
                executor.submit(self._gen_single_proc, self.batch_size * i)
                for i in q
            ]
            for future in futures:
                lst = future.result()
                for i in range(0, len(lst), self.batch_size):
                    yield lst[i : i + self.batch_size]

    def generator_torch(
        self,
        n_batches: int,
        batches_per_worker: int = 2,
        transforms: callable = None,
        max_workers: int = None,
    ) -> Iterator[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Generate batches of patches and labels in parallel using torch Tensors.

        This method uses the internal `_gen_single_proc_torch` method to
        generate batches of patches and labels. The required number of batches
        is split between worker processes, which are executed in parallel
        using a `ProcessPoolExecutor`. The results are then yielded in chunks
        of `self.batch_size`.

        Args:
            n_batches (int): Number of batches to generate.
            batches_per_worker (int): Number of batches to generate per worker
                process. Defaults to 2.
            transforms (callable): Optional transforms to apply to the patch
                features. Defaults to `None`.
            max_workers (int): Maximum number of worker processes to spawn.
                Defaults to `None`, which means that the number of workers is
                determined by the `ProcessPoolExecutor`.

        Yields:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]: A tuple containing
                tensors for the features, label, and coordinates
                (all stacked in batches).
        """
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # split required number of batches between workers
            q = self._split_chunks(n_batches, batches_per_worker)
            futures = [
                executor.submit(
                    self._gen_single_proc_torch, self.batch_size * i
                )
                for i in q
            ]
            for future in futures:
                lst = future.result()
                for i in range(0, len(lst), self.batch_size):
                    batch_elements = lst[i : i + self.batch_size]
                    features = torch.stack([e[0] for e in batch_elements])
                    labels = torch.stack([e[1] for e in batch_elements])
                    coords = torch.stack([e[2] for e in batch_elements])
                    if transforms is not None:
                        features = transforms(features)
                    yield features, labels, coords

    def __len__(self):
        """
        Returns the approximate number of patches in the dataset.
        """
        ps = self.patch_size * self.layer
        return int(sum(self._areas_per_cls.values()) / (ps * ps))
