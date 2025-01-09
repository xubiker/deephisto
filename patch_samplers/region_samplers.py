from collections import defaultdict
import json
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
from psimage import PSImage
from psimage.patches import Patch
from shapely import Polygon
from tqdm import tqdm

from torch.utils.data import IterableDataset


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

        Raises:
            RuntimeError: If the region has invalid shape or dtype.
        """
        self.file_path = img_path
        self.region_idx = region_idx
        self.class_ = class_
        self.vertices = vertices
        self._layer = layer
        self._layer_size = layer_size

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

    def _extract_patch_coords_rnd(
        self,
        patch_size: int,
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

        ps = patch_size
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

    def _extract_patch_coords_dense(
        self,
        patch_size: int,
        stride: int,
        region_intersection: float = 0.75,
    ) -> list[tuple[int, int]]:
        """
        Extract a list of coordinates for patches of size `patch_size` that
        lie within the region defined by the annotated polygon.

        The `stride` parameter determines the spacing between patch centers.
        The `region_intersection` parameter determines the minimum fraction of
        the patch area that must lie within the region in order for the patch
        to be included in the output.

        The coordinates are returned as a list of (y, x) tuples, where y and x
        are the coordinates of the top-left corner of the patch.

        :param patch_size: The size of the patches to extract.
        :param stride: The spacing between patch centers.
        :param region_intersection: The minimum fraction of the patch area
            that must lie within the region in order for the patch to be
            included in the output.
        :return: A list of (y, x) tuples, where y and x are the coordinates of
            the top-left corner of the patch.
        """
        ps = patch_size
        h, w = self._layer_size
        x0, y0, x1, y1 = self.polygon.bounds
        x0, y0, x1, y1 = round(x0), round(y0), round(x1), round(y1)
        x1 = min(x1, w - patch_size)
        y1 = min(y1, h - patch_size)
        coords = []
        for y in range(y0, y1, stride):
            for x in range(x0, x1, stride):
                patch_polygon = Polygon(
                    [
                        (x, y),
                        (x + ps, y),
                        (x + ps, y + ps),
                        (x, y + ps),
                    ]
                )
                ia = self.polygon.intersection(patch_polygon).area
                if ia > patch_size * patch_size * region_intersection:
                    coords.append((y, x))
        return coords


def _parse_annotations(
    img_anno_paths: list[tuple[Path, Path]],
    layer: int,
    classes: list[str] = None,
) -> dict[str, list[RegionAnnotation]]:
    """
    Parse annotations from the given image-annotation path pairs.

    Returns:
        dict[str, list[RegionAnnotation]]: A dictionary where each key is a
            class label and the value is a list of RegionAnnotation objects
            associated with that label.

    Raises:
        ValueError: If the region is too small.
    """
    regions_all = defaultdict(list)
    regions_per_image = [defaultdict(list) for _ in img_anno_paths]
    regions_failed = 0
    for j, (psim_path, anno_path) in enumerate(
        tqdm(img_anno_paths, "Parsing annotations")
    ):
        with PSImage(psim_path) as psim:
            with open(anno_path) as anno_f:
                for i, a in enumerate(json.load(anno_f)):
                    cls = a["class"]
                    if classes is not None and cls not in classes:
                        continue
                    try:
                        reg = RegionAnnotation(
                            img_path=psim_path,
                            region_idx=i,
                            class_=cls,
                            vertices=np.array(a["vertices"], dtype=np.float64),
                            layer=layer,
                            layer_size=psim.layer_size(layer),
                        )
                        # add region to image dictionary
                        regions_per_image[j][cls].append(reg)
                        # add region to united dictionary
                        regions_all[cls].append(reg)
                    except Exception:
                        regions_failed += 1

    if regions_failed > 0:
        print(f"Failed to parse {regions_failed} regions.")

    l = {cls: len(r) for cls, r in regions_all.items()}
    print(f"regions all: {l}")

    print("regions per image:")
    for i, rpi in enumerate(regions_per_image):
        l = {cls: len(r) for cls, r in rpi.items()}
        print(f"\timage {i}: {l}")

    return regions_all, regions_per_image


class AnnoRegionRndSampler:

    def __init__(
        self,
        img_anno_paths: list[tuple[Path, Path]],
        layer: int,
        patch_size: int,
        region_intersection: float = 0.75,
        patches_from_one_region: int = 4,
        region_area_influence: float = 0.5,
        classes: list[str] = None,
        one_image_for_batch: bool = False,
    ):
        """
        Initialize an AnnoRegionRndSampler object.

        Args:
            img_anno_paths: list[tuple[Path, Path]]
                A list of tuples containing the paths to the images and their
                corresponding annotations.
            layer: int
                The layer of image to extract patches from.
            patch_size: int
                The size of the patches to be extracted.
            region_intersection: float, optional
                The minimum fraction of the patch area that must lie within the
                region in order for the patch to be included in the output.
                Defaults to 0.75.
            patches_from_one_region: int, optional
                The number of patches to extract from each region. Defaults to 4.
            region_area_influence: float, optional
                The influence of the region area on the weights. If 0, equal
                weights are assigned to all regions. If > 0, larger regions get
                more weight. If < 0, smaller regions get more weight. Defaults to
                0.5.
            classes: list[str], optional
                A list of class labels to be used. If None, all classes are
                used. Defaults to None.
            one_image_for_batch: bool, optional
                If True, each batch will contain patches from only one image.
                Defaults to False.

        """
        self.img_anno_paths = img_anno_paths
        self.layer = layer
        self.patch_size = patch_size
        self.region_intersection = region_intersection
        self.patches_from_one_region = patches_from_one_region
        self.region_area_influence = region_area_influence
        self.classes = classes
        self.one_image_for_batch = one_image_for_batch
        self.regions, self.regions_per_image = _parse_annotations(
            img_anno_paths, layer=layer, classes=classes
        )
        self.classes = sorted(list(self.regions.keys()))
        self._print_anno_stats(self.regions)
        self._reg_w_all, self._reg_w_per_img, self._img_w, self._img_w_all = (
            self._calc_weights(self.regions, self.regions_per_image)
        )
        # self._print_weights(self._reg_w_all, self._reg_w_per_img, self._img_w)
        self._set_mp()

    def _set_mp(self):
        """
        ! Used in order to avoid conflicts with pytorch !
        Use the spawn or forkserver start method instead of the default fork.
        These method reinitialize the child process in a clean state.
        Should be called before creating ProcessPoolExecutor.
        """
        import multiprocessing as mp

        mp.set_start_method("spawn", force=True)

    def _print_anno_stats(self, regions: dict[str, list[RegionAnnotation]]):
        areas_per_cls = {
            cls: sum([i.area for i in regs]) for cls, regs in regions.items()
        }
        print("Total area per class:")
        for cls in areas_per_cls:
            size_gpx = round(areas_per_cls[cls] / 1e9, 2)
            size_prc = round(
                areas_per_cls[cls] / sum(areas_per_cls.values()) * 100,
                2,
            )
            print(f"\t{cls}: {size_gpx} Gpx ({size_prc}%)")
        print(f"Approximate number of patches in dataset: {len(self)}")

    def _calc_area_weights(
        self, areas: list[float], area_influence: float
    ) -> list[np.ndarray]:
        """
        Calculate the weights for each region based on their areas.

        The influence of the region area on the weights is controlled by the
        `area_influence` parameter. If 0, equal weights are assigned to all
        regions. If > 0, larger regions get more weight. If < 0, smaller regions
        get more weight.

        Parameters
        ----------
        areas : list[float]
            A list of region areas.
        area_influence : float
            The influence of the region area on the weights.

        Returns
        -------
        list[np.ndarray]
            A list of weights for regions.
        """
        assert -1 <= area_influence <= 1
        areas_inv = [1 / a for a in areas]
        w_proportional = np.array(areas) / sum(areas)
        w_inv_proportional = np.array(areas_inv) / sum(areas_inv)
        w_default = np.ones(len(areas), dtype=np.float64) / len(areas)

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
        return w

    def _print_weights(
        self, reg_weights_all, reg_weights_per_img, img_weights
    ):
        print("reg_w_all")
        for cls, weigts in reg_weights_all.items():
            print(f"\t{cls}: {len(weigts)} items")
        print("reg_w_per_img")
        for i in range(len(self.img_anno_paths)):
            print(f"\t image {i}:")
            for cls, weigts in reg_weights_per_img[i].items():
                print(f"\t\t{cls}: {weigts}")
        print("img_w")
        for cls, weights in img_weights.items():
            print(f"\t{cls}: {weights}")

    def _calc_weights(
        self,
        regions: dict[str, list[RegionAnnotation]],
        regions_per_image: list[dict[str, list[RegionAnnotation]]],
    ) -> tuple:
        """
        Calculate the weights for each region and each image based on their areas.

        Parameters
        ----------
        regions : dict[str, list[RegionAnnotation]]
            A dictionary where each key is a class label and the value is a list
            of RegionAnnotation objects associated with that label.
        regions_per_image : list[dict[str, list[RegionAnnotation]]]
            {A list of dictionaries where each key is a class label and the value
            is a list of RegionAnnotation objects associated with that label} for
            each image.

        Returns
        -------
        tuple
            A tuple of three dictionaries:

            reg_weights_all : dict[str, list[float]]
                A dictionary where each key is a class label and the value is a
                list of weights for regions of that class.
            reg_weights_per_img : list[dict[str, list[float]]]
                {A list of dictionaries where each key is a class label and the
                value is a list of weights for regions of that class} for each
                image.
            img_weights : dict[str, np.ndarray]
                A dictionary where each key is a class label and the value is a
                numpy array of weights for images.
            img_weights_all: np.ndarray
                A numpy array of weights for all images corresponding to the
                area of annotated regions on each image.
        """

        # calculate weights of regions corresponding to class
        # cls -> [weights] * n_regions_for_class
        reg_weights_all = {
            cls: self._calc_area_weights(
                [r.area for r in reg], self.region_area_influence
            )
            for cls, reg in regions.items()
        }

        # calculate weights of regions corresponding to class for each image
        # [cls -> [weights] * n_regions_for_class_on_image] * n_images
        reg_weights_per_img = []
        for regions in regions_per_image:
            w = {
                cls: self._calc_area_weights(
                    [r.area for r in reg], self.region_area_influence
                )
                for cls, reg in regions.items()
            }
            reg_weights_per_img.append(w)

        # calculate weights of images corresponding to class
        # cls -> [weights] * n_images
        img_weights = dict()
        for cls in self.classes:
            # for all images extract only regions of the desired class
            regs_per_image = [
                regions[cls] if cls in regions else []
                for regions in regions_per_image
            ]
            a = np.array(
                [sum([r.area for r in regs]) for regs in regs_per_image]
            )
            img_weights[cls] = a / np.sum(a)

        # calculate weights of images corresponding to the area of annotated regions
        all_regs_areas_per_image = [
            sum([sum([j.area for j in i]) for i in r.values()])
            for r in regions_per_image
        ]
        img_weights_all = self._calc_area_weights(
            all_regs_areas_per_image, self.region_area_influence
        )

        return (
            reg_weights_all,
            reg_weights_per_img,
            img_weights,
            img_weights_all,
        )

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
            coords = region._extract_patch_coords_rnd(
                n_patches=n,
                patch_size=self.patch_size,
                region_intersection=self.region_intersection,
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

    def _gen_single_proc(
        self, n: int, cls_idx: int = None
    ) -> list[tuple[Patch, int]]:
        """
        Generate patches from regions for one process.

        Generate n patches from all regions, using the weights calculated
        in _calc_region_weights. The patches are generated one by one, and
        the process is repeated until n patches are generated.

        Args:
            n (int): Number of patches to generate.
            cls_idx (int): Index of the class to generate patches for. If not
            provided, patches from all classes are generated.

        Returns:
            list[tuple[Patch, int]]: List of tuples containing a Patch and its
            class index.
        """
        res = []
        if self.one_image_for_batch:
            img_idx = np.random.choice(
                len(self.img_anno_paths), p=self._img_w_all
            )
            # get classes for selected img (image annotation may not have all classes)
            classes_for_img = self._reg_w_per_img[img_idx].keys()
            classes_idx = [self.classes.index(cls) for cls in classes_for_img]
            while len(res) < n:
                try:
                    # select class
                    c_idx = cls_idx or np.random.choice(classes_idx)
                    cls = self.classes[c_idx]
                    if cls not in classes_for_img:
                        raise Exception(f"Class {cls} not found in image")
                    # select region
                    region: RegionAnnotation = np.random.choice(
                        self.regions_per_image[img_idx][cls],
                        p=self._reg_w_per_img[img_idx][cls],
                    )
                    k = min(self.patches_from_one_region, n - len(res))
                    res.extend(
                        [
                            (p, c_idx)
                            for p in self._patches_one_region(region, k)
                        ]
                    )
                except Exception:
                    continue
        else:
            while len(res) < n:
                try:
                    c_idx = cls_idx or np.random.randint(len(self.classes))
                    cls = self.classes[c_idx]
                    region: RegionAnnotation = np.random.choice(
                        self.regions[cls],
                        p=self._reg_w_all[cls],
                    )
                    k = min(self.patches_from_one_region, n - len(res))
                    res.extend(
                        [
                            (p, c_idx)
                            for p in self._patches_one_region(region, k)
                        ]
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
        for patch, idx in self._gen_single_proc(n):
            features = torch.tensor(patch.data, dtype=torch.float32) / 255
            labels = torch.tensor(idx, dtype=torch.int64)
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

    def structs_generator(
        self,
        batch_size: int,
        n_batches: int,
        batches_per_worker: int = 2,
        max_workers: int = None,
        cls_idx: int = None,
    ) -> Iterator[tuple[Patch, int]]:
        """
        Generate batches of patches and labels in parallel.

        This method uses the internal `_gen_single_proc` method to generate
        batches of patches and labels. The required number of batches is split
        between worker processes, which are executed in parallel using a
        `ProcessPoolExecutor`. The results are then yielded in chunks of
        `batch_size`.

        Args:
            batch_size (int): Number of patches per batch.
            n_batches (int): Number of batches to generate.
            batches_per_worker (int): Number of batches to generate per worker
                process.
            max_workers (int): Maximum number of worker processes to spawn.
                Defaults to `None`, which means that the number of workers is
                determined by the `ProcessPoolExecutor`.
            cls_idx (int): Index of the class to generate patches for. If not
                provided, patches from all classes are generated.

        Yields:
            tuple[Patch, int]: A tuple containing a list of patches and their
                corresponding labels.
        """
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # split required number of batches between workers
            q = self._split_chunks(n_batches, batches_per_worker)
            futures = [
                executor.submit(self._gen_single_proc, batch_size * i, cls_idx)
                for i in q
            ]
            for future in futures:
                lst = future.result()
                for i in range(0, len(lst), batch_size):
                    yield lst[i : i + batch_size]

    def torch_generator(
        self,
        batch_size: int,
        n_batches: int,
        batches_per_worker: int = 2,
        transforms: callable = None,
        max_workers: int = None,
        cls_idx: int = None,
    ) -> Iterator[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Generate batches of patches and labels in parallel using torch Tensors.

        This method uses the internal `_gen_single_proc_torch` method to
        generate batches of patches and labels. The required number of batches
        is split between worker processes, which are executed in parallel
        using a `ProcessPoolExecutor`. The results are then yielded in chunks
        of `batch_size`.

        Args:
            batch_size (int): Number of patches per batch.
            n_batches (int): Number of batches to generate.
            batches_per_worker (int): Number of batches to generate per worker
                process. Defaults to 2.
            transforms (callable): Optional transforms to apply to the patch
                features. Defaults to `None`.
            max_workers (int): Maximum number of worker processes to spawn.
                Defaults to `None`, which means that the number of workers is
                determined by the `ProcessPoolExecutor`.
            cls_idx (int): Index of the class to generate patches for. If not
                provided, patches from all classes are generated.

        Yields:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]: A tuple containing
                tensors for the features, label, and coordinates
                (all stacked in batches).
        """
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # split required number of batches between workers
            q = self._split_chunks(n_batches, batches_per_worker)

            futures = [
                executor.submit(self._gen_single_proc_torch, batch_size * i)
                for i in q
            ]
            for future in futures:
                lst = future.result()
                for i in range(0, len(lst), batch_size):
                    batch_elements = lst[i : i + batch_size]
                    features = torch.stack([e[0] for e in batch_elements])
                    labels = torch.stack([e[1] for e in batch_elements])
                    coords = torch.stack([e[2] for e in batch_elements])
                    if transforms is not None:
                        features = transforms(features)
                    yield features, labels, coords

    def torch_iterable_dataset(self) -> IterableDataset:
        """
        Creates a custom pytorch IterableDataset for generating patches. It can be
        used with a DataLoader, but may be slower compared to the implemented
        generator_torch().

        This function returns an instance of a custom IterableDataset that yields
        features, labels, and coordinates as torch Tensors. It uses an internal
        generator function `_g` to continuously produce batches of data by randomly
        sampling regions and extracting patches from them. The dataset is infinite,
        yielding data indefinitely until the iteration is stopped.

        Returns:
            IterableDataset: An instance of a custom IterableDataset yielding tuples
            of features, labels, and coordinates as torch Tensors.
        """

        def _g():
            while True:
                try:
                    cls_idx = np.random.randint(len(self.classes))
                    cls = self.classes[cls_idx]
                    region: RegionAnnotation = np.random.choice(
                        self.regions[cls],
                        p=self._reg_w_all[cls],
                    )
                    for p in self._patches_one_region(
                        region, self.patches_from_one_region
                    ):
                        f = torch.tensor(p.data, dtype=torch.float32) / 255
                        l = torch.tensor(cls_idx, dtype=torch.int64)
                        c = torch.tensor(
                            [p.pos_y, p.pos_y], dtype=torch.float32
                        )
                        yield f, l, c
                except Exception:
                    continue

        class CustomIterableDataset(IterableDataset):
            def __init__(self):
                pass

            def __iter__(self):
                for features, cls, coords in _g():
                    yield features, cls, coords

        return CustomIterableDataset()

    def __len__(self):
        """
        Returns the approximate number of patches in the dataset.
        """
        ps = self.patch_size * self.layer
        return int(
            sum([sum([r.area for r in lst]) for lst in self.regions.values()])
            / (ps * ps)
        )


class AnnoRegionDenseSampler:

    def __init__(
        self,
        img_anno_paths: list[tuple[Path, Path]],
        layer: int,
        patch_size: int,
        stride: int,
        region_intersection: float = 0.75,
        classes: list[str] = None,
    ):
        """
        Initialize an AnnoRegionDenseSampler object.

        Args:
            img_anno_paths: list[tuple[Path, Path]]
                A list of tuples containing the paths to the images and their
                corresponding annotations.
            layer: int
                The layer of the image to extract patches from.
            patch_size: int
                The size of the patches to be extracted.
            stride: int
                The spacing between patches.
            region_intersection: float, optional
                The minimum fraction of the patch area that must lie within the
                region in order for the patch to be included in the output.
                Defaults to 0.75.
            classes: list[str], optional
                A list of class labels to be used. If None, all classes are
                used. Defaults to None.
        """
        self.img_anno_paths = img_anno_paths
        self.layer = layer
        self.patch_size = patch_size
        self.stride = stride
        self.region_intersection = region_intersection
        self.regions, _ = _parse_annotations(
            img_anno_paths, layer=layer, classes=classes
        )
        self.classes = sorted(list(self.regions.keys()))

    def _patches_one_region(self, region: RegionAnnotation) -> list[Patch]:
        with PSImage(region.file_path) as psim:
            coords = region._extract_patch_coords_dense(
                patch_size=self.patch_size,
                stride=self.stride,
                region_intersection=self.region_intersection,
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

    def structs_generator(self) -> Iterator[tuple[Patch, int]]:
        for cls_idx, cls in enumerate(self.classes):
            regions = self.regions[cls]
            for region in regions:
                for p in self._patches_one_region(region):
                    yield p, cls_idx


def extract_and_save_subset(
    img_anno_paths: list[tuple[Path, Path]],
    out_folder: Path,
    patch_size: int,
    layer: int,
    patches_per_class: int,
    intersection=0.95,
):
    from patch_samplers.region_samplers import AnnoRegionRndSampler
    from PIL import Image

    sampler = AnnoRegionRndSampler(
        img_anno_paths=img_anno_paths,
        layer=layer,
        patch_size=patch_size,
        region_intersection=intersection,
        region_area_influence=0,  # equal weights for all regions
        patches_from_one_region=1,  # only one patch per region
    )

    batch_size = 4
    for cls_idx, cls in enumerate(sampler.classes):
        (out_folder / str(cls_idx)).mkdir(parents=True, exist_ok=True)
        n = patches_per_class // batch_size
        g = sampler.structs_generator(
            batch_size=batch_size,
            n_batches=n,
            cls_idx=cls_idx,
        )
        count = 0
        for batch in tqdm(g, total=n, desc=f"extracting class {cls}"):
            for patch, cls in batch:
                Image.fromarray(patch.data).save(
                    out_folder / str(cls_idx) / f"{count}.jpg"
                )
                count += 1
