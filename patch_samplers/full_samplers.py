import time
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import shared_memory
from pathlib import Path
from typing import Iterable, Iterator

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from psimage.image import PSImage
from psimage.patches import Patch
import torch


class FullImageRndSampler:

    def __init__(
        self,
        psimage_path: Path,
        layer: int,
        patch_size: int,
        batch_size: int,
        dense_level: int = 2,
        speedup: int = 16,
        # bg_image: np.ndarray = None,
    ):
        self._psim_path = psimage_path
        with PSImage(psimage_path) as psim:
            self.layer = layer
            psim._assert_layer(layer)
            self.h, self.w = psim.layer_size(self.layer)
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

    def _init_shared(self):
        self._accum = shared_memory.SharedMemory(
            create=True, size=self.dh * self.dw * np.dtype(np.float32).itemsize
        )

    def plot_empty_area_history(self, filename: str):
        plt.plot(self._filled_ratio)
        plt.title("Empty area")
        plt.xlabel("iteration")
        plt.ylabel("empty area percentage")
        plt.savefig(filename, format="jpg", dpi=300)

    def _update_shared_accum(self, patches: list[Patch]) -> float:
        shm = shared_memory.SharedMemory(name=self._accum.name)
        shared_accum = np.ndarray(
            (self.dh, self.dw), dtype=np.float32, buffer=shm.buf
        )
        d = self._downscale
        s = self.patch_size
        for patch in patches:
            y = patch.pos_y
            x = patch.pos_x
            shared_accum[
                y // d : (y + s) // d,
                x // d : (x + s) // d,
            ] += 1
        filled_ratio = np.count_nonzero(shared_accum) / (self.dh * self.dw)
        shm.close()
        return filled_ratio

    def _calc_prob_map(self):
        shm = shared_memory.SharedMemory(name=self._accum.name)
        shared_accum = np.ndarray(
            (self.dh, self.dw), dtype=np.float32, buffer=shm.buf
        )
        p = np.where(shared_accum >= self.dense_level, 0, 1)
        shm.close()
        if np.count_nonzero(p) < self.batch_size:
            while np.count_nonzero(p) < self.batch_size:
                p[
                    np.random.randint(0, p.shape[0], size=1),
                    np.random.randint(0, p.shape[1], size=1),
                ] = 1
        p = p / np.sum(p)
        return p

    def _cleanup(self):
        """Clean up shared memory resources."""
        shm = shared_memory.SharedMemory(name=self._accum.name)
        self._accum_copy = np.ndarray(
            (self.dh, self.dw), dtype=np.float32, buffer=shm.buf
        ).copy()
        self._accum.close()
        self._accum.unlink()

    def _extract_batch(
        self,
        psim: PSImage,
        probmap: np.ndarray = None,
    ) -> list[Patch]:

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

        # 2. extract patches in corresponding positions
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

    def _generate_batch_mp(self):
        pm = self._calc_prob_map()
        patches = []
        with PSImage(self._psim_path) as psim:
            patches = self._extract_batch(
                psim,
                probmap=pm,
            )
        filled_ratio = self._update_shared_accum(patches)
        return filled_ratio, patches

    def __iter__(self) -> Iterator[tuple[list[Patch], float]]:
        return self.generator()

    def generator(self) -> Iterator[tuple[list[Patch], float]]:
        # init accum
        self._init_shared()
        filled_ratio = 0

        # create pool and process tasks
        with ProcessPoolExecutor() as executor:
            futures = []
            while filled_ratio < 1:
                futures.append(executor.submit(self._generate_batch_mp))
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
        self._cleanup()

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
        if self._accum_copy is not None:
            a = (self._accum_copy / np.max(self._accum_copy) * 255).astype(
                np.uint8
            )
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
        stride: int = None,
    ):
        self._psim_path = psimage_path
        with PSImage(psimage_path) as psim:
            self.layer = layer
            psim._assert_layer(layer)
            self.h, self.w = psim.layer_size(self.layer)
        self.patch_size = patch_size
        self.batch_size = batch_size
        self.stride = stride
        print(f"Image {self.h} x {self.w}")
        super().__init__()

    def _generate_batch_mp(self, coords):
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

    def __iter__(self) -> Iterable[tuple[list[Patch], float]]:
        return self.__next__()

    def generator(self) -> Iterable[tuple[list[Patch], float]]:

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

        # Create a pool of processes to generate batches
        with ProcessPoolExecutor() as executor:
            futures = [
                executor.submit(self._generate_batch_mp, coords)
                for coords in coords_batched
            ]

            # Yield the patches and progress as they are completed
            for i, f in enumerate(futures):
                try:
                    patches = f.result()
                    yield patches, i / len(futures)
                except Exception as e:
                    print(f"Task raised an exception: {e}")

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
