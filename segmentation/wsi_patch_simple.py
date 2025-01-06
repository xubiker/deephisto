from pathlib import Path

from typing import Callable

import numpy as np
from PIL import Image
from psimage.image import PSImage
from psimage.patches import Patch
from tqdm import tqdm

from anno.utils import AnnoDescription
from patch_samplers.wsi_samplers import DenseSamplerBatched


class ImagePredictorPatched:
    def __init__(
        self,
        psim: PSImage,
        patch_sampler,
        batch_predictor: Callable[[list[Patch]], list[np.ndarray]],
        anno: AnnoDescription,
        layer: int,
        downscale: int = 4,
    ):
        self.psim = psim
        self.patch_sampler = patch_sampler
        self.batch_predictor = batch_predictor
        self.anno = anno
        self.layer = layer
        self.downscale = downscale

    def predict_mock(self, patches: list[Patch]) -> list[np.ndarray]:
        def rand_pred(n):
            pred = np.random.rand(n)
            pred /= np.sum(pred)
            return pred

        n = len(self.anno.anno_classes)
        preds = [rand_pred(n) for p in patches]
        return preds

    def process(self):
        d = self.downscale
        h, w = self.psim.layer_size(self.layer)
        dh, dw = h // self.downscale, w // self.downscale
        n = len(self.anno.anno_classes)
        prediction = np.zeros([dh, dw, n], dtype=np.float32)
        count = np.zeros([dh, dw], dtype=np.uint8)
        progress_bar = tqdm(total=100, desc="Predicting", unit="step")
        for patches, progress in self.patch_sampler:
            patch_preds = self.predict_mock(patches)
            for i, p in enumerate(patches):
                # print(p)
                # print(patch_preds[i])
                prediction[
                    p.pos_y // d : (p.pos_y + p.patch_size) // d,
                    p.pos_x // d : (p.pos_x + p.patch_size) // d,
                    :,
                ] += patch_preds[i]
                count[
                    p.pos_y // d : (p.pos_y + p.patch_size) // d,
                    p.pos_x // d : (p.pos_x + p.patch_size) // d,
                ] += 1
            progress_bar.n = round(progress * 100, 2)
            progress_bar.refresh()
        # prediction[:,:,:] /= count[:,:,None]
        prediction = np.argmax(prediction, axis=2)
        print(prediction.shape)
        Image.fromarray(
            (prediction.astype(np.float32) / np.max(prediction) * 255).astype(
                np.uint8
            )
        ).show()

    def save_visualization(self, path: Path):
        pass


if __name__ == "__main__":
    img_path = Path(
        "/Users/xubiker/dev/PATH-DT-MSU.WSS1/images/test/test_01.psi"
    )

    # anno_dsc = AnnoDescription.with_known_colors(
    #     {
    #         "AT": (245, 119, 34),  # AT (orange)
    #         "BG": (153, 255, 255),  # BG (cyan)
    #         "LP": (64, 170, 72),  # LP (green)
    #         "MM": (255, 0, 0),  # MM (red)
    #         "TUM": (33, 67, 156),  # TUM (blue)
    #         # "DYS": (128, 0, 255),  # TUM (violet)
    #     }
    # )

    patch_sampler = DenseSamplerBatched(
        img_path, layer=2, patch_size=224, batch_size=16, stride=112
    )

    for inputs, coords, filled_ratio in patch_sampler.generator_torch():
        print(inputs.shape, coords.shape, filled_ratio)

    # with PSImage(img_path) as psim:
    #     predictor = ImagePredictorPatched(
    #         psim,
    #         patch_sampler=patch_sampler,
    #         batch_predictor=None,
    #         anno=anno_dsc,
    #         layer=2,
    #         downscale=4,
    #     )
    #     predictor.process()
