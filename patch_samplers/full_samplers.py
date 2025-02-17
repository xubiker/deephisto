from enum import Enum
import time
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import shared_memory
from pathlib import Path
from typing import Iterable, Iterator

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from psimage.core.image import PSImage
from psimage.core.patches import Patch
import torch


class SamplerExecutionMode(Enum):
    INMEMORY_SINGLEPROC = 1
    ONDISK_MULTIPROC = 2


class FullImageRndSampler:

    def __init__(
        self,
        psimage_path: Path,
        layer: int,
        patch_size: int,
        batch_size: int,
        mode: SamplerExecutionMode,
        dense_level: int = 2,
        speedup: int = 16,
    ):
        self.mode = mode
        self._psim_path = psimage_path
        with PSImage(psimage_path) as psim:
            self.layer = layer
            psim._assert_layer(layer)
            self.h, self.w = psim.layer_size(self.layer)
            if self.mode == SamplerExecutionMode.INMEMORY_SINGLEPROC:
                self.data = self._read_image(psim)
            self.dh = self.h // speedup
            self.dw = self.w // speedup
        print(
            f"Image {self.h} x {self.w} at {speedup}x -> {self.dh} x {self.dw}"
        )
        self.patch_size = patch_size
        self.batch_size = batch_size
        self._downscale = speedup
        self.dense_level = dense_level
        self._filled_ratio = []
        super().__init__()

    def _read_image(self, psim: PSImage) -> np.ndarray:
        print("Loading image into memory...")
        return psim.get_region_from_layer(self.layer, (0, 0), (self.h, self.w))

    def _init_accum_mp(self):
        self._accum_shm = shared_memory.SharedMemory(
            create=True, size=self.dh * self.dw * np.dtype(np.float32).itemsize
        )

    def _init_accum_sp(self):
        self._accum = np.zeros([self.dh, self.dw], dtype=np.float32)

    def plot_empty_area_history(self, filename: str):
        plt.plot(self._filled_ratio)
        plt.title("Empty area")
        plt.xlabel("iteration")
        plt.ylabel("empty area percentage")
        plt.savefig(filename, format="jpg", dpi=300)

    def _update_accum_mp(self, patches: list[Patch]) -> float:
        shm = shared_memory.SharedMemory(name=self._accum_shm.name)
        shared_accum = np.ndarray(
            (self.dh, self.dw), dtype=np.float32, buffer=shm.buf
        )
        filled_ratio = self._update_accum_sp(shared_accum, patches)
        shm.close()
        return filled_ratio

    def _update_accum_sp(
        self, accum: np.ndarray, patches: list[Patch]
    ) -> float:
        d = self._downscale
        s = self.patch_size
        for patch in patches:
            y = patch.pos_y
            x = patch.pos_x
            accum[
                y // d : (y + s) // d,
                x // d : (x + s) // d,
            ] += 1
        filled_ratio = np.count_nonzero(accum) / accum.size
        return filled_ratio

    def _calc_probmap_mp(self):
        shm = shared_memory.SharedMemory(name=self._accum_shm.name)
        shared_accum = np.ndarray(
            (self.dh, self.dw), dtype=np.float32, buffer=shm.buf
        )
        p = self._calc_probmap_sp(shared_accum)
        shm.close()
        return p

    def _calc_probmap_sp(self, accum: np.ndarray):
        p = np.where(accum >= self.dense_level, 0, 1)
        if np.count_nonzero(p) < self.batch_size:
            while np.count_nonzero(p) < self.batch_size:
                p[
                    np.random.randint(0, p.shape[0], size=1),
                    np.random.randint(0, p.shape[1], size=1),
                ] = 1
        p = p / np.sum(p)
        return p

    def _cleanup_mp(self):
        """Clean up shared memory resources."""
        shm = shared_memory.SharedMemory(name=self._accum_shm.name)
        self._accum = np.ndarray(
            (self.dh, self.dw), dtype=np.float32, buffer=shm.buf
        ).copy()
        self._accum_shm.close()
        self._accum_shm.unlink()

    def _prepare_indices(
        self, probmap: np.ndarray = None
    ) -> list[tuple[int, int]]:
        def clamp(y, x):
            return (
                max(min(y, self.h - self.patch_size), 0),
                max(min(x, self.w - self.patch_size), 0),
            )

        # 1. calculate indices
        if probmap is not None:
            indices = list(
                np.random.choice(
                    self.dh * self.dw,
                    size=self.batch_size,
                    replace=False,
                    p=probmap.flatten(),
                )
            )
            pd2 = self.patch_size // self._downscale // 2
            indices = [
                clamp(
                    (ind // self.dw - pd2) * self._downscale
                    + np.random.randint(self._downscale),
                    (ind % self.dw - pd2) * self._downscale
                    + np.random.randint(self._downscale),
                )
                for ind in indices
            ]
        else:
            indices = [
                (
                    np.random.randint(self.h - self.patch_size),
                    np.random.randint(self.w - self.patch_size),
                )
                for _ in range(self.batch_size)
            ]
        return indices

    def _extract_patches_psim(
        self,
        indices: list[tuple[int, int]],
        psim: PSImage,
    ) -> list[Patch]:
        patches = [
            Patch(
                layer=self.layer,
                pos_x=x,
                pos_y=y,
                patch_size=self.patch_size,
                data=np.array(
                    psim.get_region_from_layer(
                        self.layer,
                        (y, x),
                        (y + self.patch_size, x + self.patch_size),
                    ),
                ),
            )
            for y, x in indices
        ]
        return patches

    def _extract_patches_np(
        self,
        indices: list[tuple[int, int]],
        data: np.ndarray,
    ) -> list[Patch]:
        patches = [
            Patch(
                layer=self.layer,
                pos_x=x,
                pos_y=y,
                patch_size=self.patch_size,
                data=data[y : y + self.patch_size, x : x + self.patch_size, :],
            )
            for y, x in indices
        ]
        return patches

    def _generate_batch(self):
        if self.mode == SamplerExecutionMode.INMEMORY_SINGLEPROC:
            pm = self._calc_probmap_sp(self._accum)
            indices = self._prepare_indices(pm)
            patches = self._extract_patches_np(indices=indices, data=self.data)
            filled_ratio = self._update_accum_sp(self._accum, patches)
            return filled_ratio, patches
        if self.mode == SamplerExecutionMode.ONDISK_MULTIPROC:
            pm = self._calc_probmap_mp()
            indices = self._prepare_indices(pm)
            patches = []
            with PSImage(self._psim_path) as psim:
                patches = self._extract_patches_psim(
                    indices=indices,
                    psim=psim,
                )
            filled_ratio = self._update_accum_mp(patches)
            return filled_ratio, patches

    def __iter__(self) -> Iterator[tuple[list[Patch], float]]:
        if self.mode == SamplerExecutionMode.INMEMORY_SINGLEPROC:
            return self._generator_sp()
        if self.mode == SamplerExecutionMode.ONDISK_MULTIPROC:
            return self._generator_mp()

    def _generator_mp(self) -> Iterator[tuple[list[Patch], float]]:
        # init accum
        self._init_accum_mp()
        filled_ratio = 0

        # create pool and process tasks
        with ProcessPoolExecutor() as executor:
            futures = []
            while filled_ratio < 1:
                futures.append(executor.submit(self._generate_batch))
                # Check completed futures
                for f in futures[:]:
                    if f.done():
                        try:
                            filled_ratio, patches = f.result()
                            self._filled_ratio.append(filled_ratio)
                            yield patches, filled_ratio
                            if filled_ratio >= 1:
                                break
                        except Exception as e:
                            print(f"Task raised an exception: {e}")
                        futures.remove(f)
                # Optional: Add a small delay to prevent
                # overwhelming the executor
                time.sleep(0.01)

                if filled_ratio >= 1:
                    # Cancel remaining futures and break
                    for pending in futures:
                        pending.cancel()
                    break  # Break out of the generator entirely
        # cleanup
        self._cleanup_mp()

    def _generator_sp(self) -> Iterator[tuple[list[Patch], float]]:
        # init accum
        self._init_accum_sp()
        filled_ratio = 0

        while filled_ratio < 1:
            filled_ratio, patches = self._generate_batch()
            # Check completed futures
            self._filled_ratio.append(filled_ratio)
            yield patches, filled_ratio
            if filled_ratio >= 1:
                break

    def generator(self) -> Iterator[tuple[list[Patch], float]]:
        if self.mode == SamplerExecutionMode.INMEMORY_SINGLEPROC:
            return self._generator_sp()
        if self.mode == SamplerExecutionMode.ONDISK_MULTIPROC:
            return self._generator_mp()

    def generator_torch(
        self,
    ) -> Iterator[tuple[torch.Tensor, torch.Tensor, float]]:
        for patches, filled_ratio in self.generator():
            features = torch.Tensor(np.stack([p.data for p in patches]))
            coords = torch.Tensor(
                np.stack([np.array([p.pos_y, p.pos_x]) for p in patches])
            )
            yield features, coords, filled_ratio

    def visualize_heatmap(self, name: str):
        if self._accum is not None:
            a = (self._accum / np.max(self._accum) * 255).astype(np.uint8)
            im = Image.fromarray(a)
            im.save(name)
            a = np.where(a > 0, 255, 0).astype(np.uint8)
            im = Image.fromarray(a)
            im.save("_" + name, quality=98)


class FullImageDenseSampler:

    def __init__(
        self,
        psimage_path: Path,
        layer: int,
        patch_size: int,
        batch_size: int,
        mode: SamplerExecutionMode,
        stride: int = None,
    ):
        self._psim_path = psimage_path
        self.mode = mode
        with PSImage(psimage_path) as psim:
            self.layer = layer
            psim._assert_layer(layer)
            self.h, self.w = psim.layer_size(self.layer)
            if self.mode == SamplerExecutionMode.INMEMORY_SINGLEPROC:
                self.data = self._read_image(psim)

        self.patch_size = patch_size
        self.batch_size = batch_size
        self.stride = stride
        print(f"Image {self.h} x {self.w}")
        super().__init__()

    def _read_image(self, psim: PSImage) -> np.ndarray:
        print("Loading image into memory...")
        return psim.get_region_from_layer(self.layer, (0, 0), (self.h, self.w))

    def _generate_batch_psim(self, coords):
        with PSImage(self._psim_path) as psim:
            # Extract patches from given coordinates
            patches = [
                Patch(
                    layer=self.layer,
                    pos_x=x,
                    pos_y=y,
                    patch_size=self.patch_size,
                    data=np.array(
                        psim.get_region_from_layer(
                            self.layer,
                            (y, x),
                            (y + self.patch_size, x + self.patch_size),
                        ),
                    ),
                )
                for y, x in coords
            ]
        return patches

    def _generate_batch_memory(self, coords):
        # Extract patches from given coordinates
        patches = [
            Patch(
                layer=self.layer,
                pos_x=x,
                pos_y=y,
                patch_size=self.patch_size,
                data=self.data[
                    y : y + self.patch_size,
                    x : x + self.patch_size,
                    :,
                ],
            )
            for y, x in coords
        ]
        return patches

    def __iter__(self) -> Iterable[tuple[list[Patch], float]]:
        return self.generator()

    def _create_batched_coords(self):
        def chunk_list(lst, n):
            # Split list into chunks of size n
            return [lst[i : i + n] for i in range(0, len(lst), n)]

        # Calculate all coordinates for patches based on stride
        coords = [
            (y, x)
            for y in range(0, self.h - self.patch_size, self.stride)
            for x in range(0, self.w - self.patch_size, self.stride)
        ]
        # Add last column patches
        coords += [
            (y, self.w - self.patch_size)
            for y in range(0, self.h - self.patch_size, self.stride)
        ]
        # Add last row patches
        coords += [
            (self.h - self.patch_size, x)
            for x in range(0, self.w - self.patch_size, self.stride)
        ]

        # Include the last patch at the bottom-right corner
        coords.append((self.h - self.patch_size, self.w - self.patch_size))

        # Group coordinates into batches
        coords_batched = chunk_list(coords, self.batch_size)
        while len(coords_batched[-1]) < self.batch_size:
            coords_batched[-1].append(coords[-1])

        return coords_batched

    def _generator_mp(self) -> Iterable[tuple[list[Patch], float]]:

        coords_batched = self._create_batched_coords()

        # Create a pool of processes to generate batches
        with ProcessPoolExecutor() as executor:
            futures = [
                executor.submit(self._generate_batch_psim, coords)
                for coords in coords_batched
            ]

            # Yield the patches and progress as they are completed
            for i, f in enumerate(futures):
                try:
                    patches = f.result()
                    yield patches, i / len(futures)
                except Exception as e:
                    print(f"Task raised an exception: {e}")

    def _generator_sp(self) -> Iterable[tuple[list[Patch], float]]:
        coords_batched = self._create_batched_coords()
        for i, coords in enumerate(coords_batched):
            patches = self._generate_batch_memory(coords)
            yield patches, i / len(coords_batched)

    def generator(self) -> Iterable[tuple[list[Patch], float]]:
        if self.mode == SamplerExecutionMode.INMEMORY_SINGLEPROC:
            return self._generator_sp()
        if self.mode == SamplerExecutionMode.ONDISK_MULTIPROC:
            return self._generator_mp()

    def generator_torch(
        self,
    ) -> Iterator[tuple[torch.Tensor, torch.Tensor, float]]:
        for patches, filled_ratio in self.generator():
            features = torch.tensor(
                np.stack([p.data for p in patches]).astype(np.float32) / 255
            )
            coords = torch.tensor(
                np.stack(
                    [
                        np.array([p.pos_y, p.pos_x], dtype=np.float32)
                        for p in patches
                    ]
                )
            )
            yield features, coords, filled_ratio
