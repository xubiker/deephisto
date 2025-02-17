from pathlib import Path
from torch.utils.data import DataLoader
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
import json
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import get_img_ano_paths
from models.graph_hnet_pseudo import Graph_HNet
from torchvision import datasets, transforms,models 
from anno.utils import AnnoDescription
from einops import rearrange
from patch_samplers.region_samplers import AnnoRegionRndSampler
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score, classification_report
from wsiDataset import PsiGraphDataset_Test
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
        patch_sampler,
        model,
        device,
        anno: AnnoDescription,
        layer: int,
        downscale: int = 4,
        rnd_sampler = True
    ):
        self.patch_sampler = patch_sampler
        self.patch_encoder,self.model = model
        self.anno = anno
        self.layer = layer
        self.device = device
        self.downscale = downscale
        self.rnd_sampler = rnd_sampler


    def process(self) -> np.ndarray:
        y_true = []
        y_pred = []
        progress_bar = tqdm(total=len(self.patch_sampler), desc="Predicting", unit="step")
        progress = 0
        for batch in self.patch_sampler:
            if len(batch) == 4:
                f, cls, coords,_ = batch 
            else:
                f, cls, coords = batch
            patch_preds = self.batch_predictor(f, coords, self.patch_encoder, self.model, self.device)
            prediction = np.argmax(patch_preds, axis=1)
            q = cls.numpy().tolist()
            y_true.append(q)
            y_pred.append(prediction)
            progress+=1
            progress_bar.n = round(progress, 2)
            progress_bar.refresh()
        # 将 y_true 和 y_pred 转换为一维数组
        y_true_flat = np.concatenate(y_true).ravel()
        y_pred_flat = np.concatenate(y_pred).ravel()

        # 生成分类报告
        report = classification_report(y_true_flat, y_pred_flat)
        print(report)
            
        # d = self.downscale
        # dh, dw = self.h // self.downscale, self.w // self.downscale
        # n = len(self.anno.anno_classes)
        # prediction = np.zeros([dh, dw, n], dtype=np.float32)
        # # count = np.zeros([dh, dw], dtype=np.uint8)
        # progress_bar = tqdm(total=100, desc="Predicting", unit="step")
        # for patches, progress in self.patch_sampler:
        #     patch_preds = self.batch_predictor(patches,self.patch_encoder,self.model,self.device)
        #     for i, p in enumerate(patches):
        #         prediction[
        #             p.pos_y // d : (p.pos_y + p.patch_size) // d,
        #             p.pos_x // d : (p.pos_x + p.patch_size) // d,
        #             :,
        #         ] += patch_preds[i]
        #         # count[
        #         #     p.pos_y // d : (p.pos_y + p.patch_size) // d,
        #         #     p.pos_x // d : (p.pos_x + p.patch_size) // d,
        #         # ] += 1
        #     progress_bar.n = round(progress * 100, 2)
        #     progress_bar.refresh()
        # # prediction[:,:,:] /= count[:,:,None]
        return report


    def batch_predictor(self,features: list[Patch],coords, patch_encoder,model, device) -> np.ndarray:
        graph_size = 8
        if self.rnd_sampler:
            features = features.permute(0, 3, 1, 2).contiguous().to(device)
            coords = coords.to(device)
            with torch.no_grad():
                batch_size = 1
                logits,latents = patch_encoder(features)
                
                latents = rearrange(latents,'(b h w) d -> b d h w', b=batch_size, h=graph_size, w=graph_size)
                _, predicted = torch.max(logits, 1)
                predictions = model(latents,coords,predicted).detach().cpu().numpy()
        else:
            features = features.to(device)
            coords = coords.to(device)
            with torch.no_grad():
                batch_size = coords.shape[0]
                
                features = rearrange(features,'b n c h w -> (b n) c h w', b=batch_size, c=3,h=224, w=224)
                coords = rearrange(coords,'b n d -> (b n) d', b=batch_size, n =graph_size*graph_size , d=2 )
                logits,latents = patch_encoder(features)
                latents = rearrange(latents,'(b h w) d -> b d h w', b=batch_size, h=graph_size, w=graph_size)
                _, predicted = torch.max(logits, 1)
                predictions = model(latents,coords,predicted).detach().cpu().numpy()
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


def load_model_gnn(patch_encoder_weights_path,gnn_weights_path, device) -> torch.nn.Module:
    patch_encoder  = models.resnet50(pretrained=True)
    num_ftrs = patch_encoder.fc.in_features
    patch_encoder.fc = torch.nn.Sequential(
                torch.nn.Dropout(0),
                torch.nn.Linear(num_ftrs,5)
            )
    parent_dir = os.path.dirname(gnn_weights_path)

    # 构造 config.json 的路径
    config_path = os.path.join(parent_dir, "config.json")
    with open(config_path, 'r') as f:
            config = json.load(f)
    model = Graph_HNet(num_ftrs,config)
    loadnet = torch.load(patch_encoder_weights_path, map_location=torch.device('cpu'))
    keyname = 'model'
    print(patch_encoder.load_state_dict(loadnet[keyname], strict=True))
    loadnet = torch.load(gnn_weights_path, map_location=torch.device('cpu'))
    print(model.load_state_dict(loadnet[keyname], strict=True))
    
    patch_encoder.to(device).eval()
    model.to(device).eval()
    return patch_encoder,model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Image Prediction with Command Line Arguments')
    parser.add_argument('--img_anno_path', type=str, default="/home/data_repository/PATH-DT-MSU_dev/WSS2_v2_psi",
                        help='Path to the image and annotation data')
    parser.add_argument('--patch_encoder_weights', type=str,
                        default="/home/z.sun/deephisto/saves/Mask-Newgraph-hnet-pseudo_k=5_block-2-attn-0/patch_encoder_best-11.pth",
                        help='Path to the patch encoder weights')
    parser.add_argument('--gnn_weights', type=str,
                        default="/home/z.sun/deephisto/saves/Mask-Newgraph-hnet-pseudo_k=5_block-2-attn-0/best-11.pth",
                        help='Path to the GNN weights')
    parser.add_argument('--rnd_sampler',default=False,action='store_true')
    parser.add_argument('--test_path',default="patches_test",type=str)
    args = parser.parse_args()

    img_anno_paths = get_img_ano_paths(
        Path(args.img_anno_path), sample="test"
    )

    # --- load model ---
    device = get_device()
    patch_encoder, model = load_model_gnn(args.patch_encoder_weights, args.gnn_weights, device)

    # --- setup all params ---
    anno_dsc = AnnoDescription.with_known_colors(
        {
            "AT": (245, 119, 34),  # AT (orange)
            "BG": (153, 255, 255),  # BG (cyan)
            "LP": (64, 170, 72),  # LP (green)
            "MM": (255, 0, 0),  # MM (red)
            "TUM": (33, 67, 156),  # TUM (blue)
            # "DYS": (128, 0, 255),  # TUM (violet)
        }
    )
    layer = 2
    downscale_vis = 8
    random_sampler = True

    # --- make WSI prediction ---
    patch_sampler = None
    if args.rnd_sampler:
        dataset = AnnoRegionRndSampler(
            img_anno_paths,
            patch_size=224,
            layer=layer,
            patches_from_one_region=4,
            one_image_for_batch=True,
            normalize_pos=True
        )
            # setup params
        n = 100  # number of batches to extract
        b_size = 64  # number of patches per batch
        b_per_worker = 2  # number of batches to extract per worker (parallel)
        patch_sampler = dataset.torch_generator(
            batch_size=b_size,
            n_batches=n,
            batches_per_worker=b_per_worker,
        )
    else:
        dataset = PsiGraphDataset_Test(Path(args.test_path))
        patch_sampler = DataLoader(dataset, batch_size=1, shuffle=True)
    predictor = ImagePredictorPatched(
        patch_sampler=patch_sampler,
        model=(patch_encoder, model),
        device=device,
        anno=anno_dsc,
        layer=layer,
        downscale=downscale_vis,
        rnd_sampler=args.rnd_sampler
    )
    pred = predictor.process()

    
