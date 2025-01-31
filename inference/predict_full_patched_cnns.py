from pathlib import Path

from typing import Callable
import argparse
import numpy as np
from PIL import Image
from psimage.image import PSImage
from psimage.patches import Patch
import torch
from tqdm import tqdm
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.graph_hnet_pseudo import Graph_HNet
from torchvision import datasets, transforms,models 
from anno.utils import AnnoDescription
from einops import rearrange

from patch_samplers.full_samplers import (
    FullImageDenseSampler,
    FullImageRndSampler,
)

def get_device():
    import torch


    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    return device

class ImagePredictorPatched:
    def __init__(
        self,
        psim_path: Path,
        patch_sampler,
        model,
        device,
        anno: AnnoDescription,
        layer: int,
        downscale: int = 4,
    ):
        self.patch_sampler = patch_sampler
        self.patch_encoder = model
        self.anno = anno
        self.layer = layer
        self.device = device
        self.downscale = downscale
        with PSImage(psim_path) as psim:
            self.h, self.w = psim.layer_size(self.layer)

    def process(self) -> np.ndarray:
        d = self.downscale
        dh, dw = self.h // self.downscale, self.w // self.downscale
        n = len(self.anno.anno_classes)
        prediction = np.zeros([dh, dw, n], dtype=np.float32)
        # count = np.zeros([dh, dw], dtype=np.uint8)
        progress_bar = tqdm(total=100, desc="Predicting", unit="step")
        for patches, progress in self.patch_sampler:
            patch_preds = self.batch_predictor(patches,self.patch_encoder,self.device)
            for i, p in enumerate(patches):
                prediction[
                    p.pos_y // d : (p.pos_y + p.patch_size) // d,
                    p.pos_x // d : (p.pos_x + p.patch_size) // d,
                    :,
                ] += patch_preds[i]
                # count[
                #     p.pos_y // d : (p.pos_y + p.patch_size) // d,
                #     p.pos_x // d : (p.pos_x + p.patch_size) // d,
                # ] += 1
            progress_bar.n = round(progress * 100, 2)
            progress_bar.refresh()
        # prediction[:,:,:] /= count[:,:,None]
        prediction = np.argmax(prediction, axis=2)
        return prediction


    def batch_predictor(self,patches: list[Patch], patch_encoder, device) -> np.ndarray:
        features = torch.tensor(
            np.stack([patch.data for patch in patches]) / 255,
            dtype=torch.float32,
        ).to(device)
        features = features.permute(0, 3, 1, 2).contiguous()
        coords = torch.tensor(
            np.stack([np.array(( p.pos_x / self.w,p.pos_y / self.h)) for p in patches]),
            dtype=torch.float32,).to(device)
        with torch.no_grad():
            batch_size = 1
            logits,latents = patch_encoder(features)
            
            predictions = logits.detach().cpu().numpy()
        return predictions


def perform_and_save_visualizations(
    img_path: Path,
    anno_dsc: AnnoDescription,
    pred: np.ndarray,
    out_dir: Path = Path("."),
):
    out_dir.mkdir(exist_ok=True, parents=True)

    # colorize
    h, w = pred.shape[:2]
    colored_image = np.zeros((h, w, 3), dtype=np.uint8)

    # Map each index in `pred` to its corresponding color
    for anno in anno_dsc.anno_classes:
        colored_image[pred == anno.id] = anno.color

    # save colorized mask
    Image.fromarray(colored_image).save(
        out_dir / f"{img_path.stem}_mask.jpg", quality=95
    )

    # save original image (downscaled)
    psim = PSImage(img_path)
    img = psim.get_region((0, 0), (psim.height, psim.width), target_hw=(h, w))
    psim.close()
    Image.fromarray(img).save(out_dir / f"{img_path.stem}.jpg", quality=95)

    # create and save overlay
    alpha = 0.6
    v = (img * alpha + colored_image * (1 - alpha)).astype(np.uint8)
    Image.fromarray(v).save(
        out_dir / f"{img_path.stem}_overlay.jpg", quality=95
    )


def load_model_cnn(patch_encoder_weights_path, device) -> torch.nn.Module:
    patch_encoder  = models.resnet50(pretrained=True)
    num_ftrs = patch_encoder.fc.in_features
    patch_encoder.fc = torch.nn.Sequential(
                torch.nn.Dropout(0),
                torch.nn.Linear(num_ftrs,5)
            )
    loadnet = torch.load(patch_encoder_weights_path, map_location=torch.device('cpu'))
    keyname = 'model'
    print(patch_encoder.load_state_dict(loadnet[keyname], strict=True))
    
    patch_encoder.to(device).eval()
    return patch_encoder

def parse_args():
    parser = argparse.ArgumentParser(description="WSI Prediction Script")
    parser.add_argument("--img_path", type=str, required=True, help="Path to the input WSI file (.psi)")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the pretrained model weights")
    parser.add_argument("--out_dir", type=str, default="./output/", help="Output directory for saving results")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    # --- load model ---
    device = get_device()
    patch_encoder = load_model_cnn(args.model_path, device)

    # --- setup all params ---
    anno_dsc = AnnoDescription.with_known_colors(
        {
            "AT": (245, 119, 34),  # AT (orange)
            "BG": (153, 255, 255),  # BG (cyan)
            "LP": (64, 170, 72),  # LP (green)
            "MM": (255, 0, 0),  # MM (red)
            "TUM": (33, 67, 156),  # TUM (blue)
        }
    )
    layer = 2
    downscale_vis = 8
    random_sampler = True

    # --- make WSI prediction ---
    patch_sampler = None
    if random_sampler:
        patch_sampler = FullImageRndSampler(
            Path(args.img_path), layer=layer, patch_size=224, batch_size=64
        )
    else:
        patch_sampler = FullImageDenseSampler(
            Path(args.img_path), layer=layer, patch_size=224, batch_size=64, stride=112
        )
    predictor = ImagePredictorPatched(
        Path(args.img_path),
        patch_sampler=patch_sampler.generator(),
        model=patch_encoder,
        device=device,
        anno=anno_dsc,
        layer=layer,
        downscale=downscale_vis,
    )
    pred = predictor.process()

    # --- save visualizations ---
    perform_and_save_visualizations(
        Path(args.img_path), anno_dsc, pred, out_dir=Path(args.out_dir)
    )
