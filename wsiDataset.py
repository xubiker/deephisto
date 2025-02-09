import os
from typing import Any
import torch
from torch.utils.data import DataLoader,Dataset

import torchvision
import albumentations as A 
from albumentations.pytorch.transforms import ToTensorV2
from patch_samplers.region_samplers import AnnoRegionRndSampler
import numpy as np
import cv2
from torchvision import datasets, transforms
from pathlib import Path
from utils import get_img_ano_paths
class PsiGraphDataset_Test(Dataset):
    def __init__(self, npz_folder: Path):
        """
        初始化数据集。

        参数:
            npz_folder (Path): 包含 .npz 文件的文件夹路径。
        """
        self.npz_folder = npz_folder
        self.npz_files = sorted(list(npz_folder.glob("*.npz")))  # 获取所有 .npz 文件并排序
        self.if_npy = False
        # 检查是否有文件
        if not self.npz_files:
            self.npz_files = sorted(list(npz_folder.glob("*.npy")))
            self.if_npy = True
        if not self.npz_files:
            raise ValueError(f"在文件夹 {npz_folder} 中未找到 .npz And .npy 文件。")

    def __len__(self):
        """
        返回数据集的大小（即 .npz 文件的数量）。
        """
        return len(self.npz_files)

    def __getitem__(self, idx):
        """
        根据索引返回数据。

        参数:
            idx (int): 数据索引。

        返回:
            features (torch.Tensor): 特征数据。
            y_true (torch.Tensor): 标签数据。
            coords (torch.Tensor): 坐标数据。
        """
        # 加载 .npz 文件
        npz_file = self.npz_files[idx]
        if not self.if_npy:
            data = np.load(npz_file)        
            features = torch.from_numpy(data["features"]).float()  # 转换为 float32 张量
            y_true = torch.from_numpy(data["y_true"]).long()       # 转换为 int64 张量
            coords = torch.from_numpy(data["coords"]).float()      # 转换为 float32 张量
        else:
            data = np.load(npz_file, allow_pickle=True).item()
            features = torch.from_numpy(data["imgs"]).float()  # 转换为 float32 张量
            y_true = torch.from_numpy(data["label"]).long()       # 转换为 int64 张量
            coords = torch.from_numpy(data["pos"]).float()    

        return features, y_true, coords
    
def build_transform(is_train, args):
    # mean = IMAGENET_DEFAULT_MEAN
    # std = IMAGENET_DEFAULT_STD
    # train transform
    if is_train:
        transform = transforms.Compose(
        [
            transforms.Lambda(
                lambda x: (
                    torch.from_numpy(x[0]).permute(2, 0, 1),
                    (torch.where(torch.from_numpy(x[1]) == 1 )[0]).squeeze(),
                    torch.tensor([x[2],x[3]])
                    # (torch.where(torch.from_numpy(x[1]) == 1 )[0]).squeeze(),
                    # torch.from_numpy(x[1])
                )
            ),
            transforms.Lambda(
                lambda x:(

                    torchvision.transforms.Compose([
                    torchvision.transforms.RandomHorizontalFlip(),
                    torchvision.transforms.RandomVerticalFlip(),
                    torchvision.transforms.RandomRotation(45),
                    torchvision.transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                    # torchvision.transforms.RandomCrop(args.patch_size),
                    ])(x[0]),
                    x[1],x[2]
                        )
            )

        ]
    )
    else :
        transform = torchvision.transforms.Compose([

                transforms.Lambda(
                    lambda x: (
                        torch.from_numpy(x[0]).permute(2, 0, 1),
                        (torch.where(torch.from_numpy(x[1]) == 1 )[0]).squeeze(),
                        torch.tensor([x[2],x[3]])
                    )
                ),
            ])

    return transform

def build_dataset(is_train, args):
    if is_train:
        inputs_transform = transforms.Compose(
        [transforms.Lambda(lambda x: x.permute(0, 3, 1, 2).contiguous())]
        )
        data_augmentations = transforms.Compose(
            [
                inputs_transform,
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.RandomVerticalFlip(),
                torchvision.transforms.RandomRotation(45),
                torchvision.transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)
            ]
        )
        img_anno_paths_train = get_img_ano_paths(
        ds_folder=Path(args.train_data_path), sample="train")
        train_val_dataset = AnnoRegionRndSampler(
        img_anno_paths_train,
        patch_size=args.patch_size,
        layer=args.layer,
        patches_from_one_region=4,
        one_image_for_batch=True,
        normalize_pos=True
        )
        return train_val_dataset,data_augmentations
    else:
        return PsiGraphDataset_Test(Path(args.test_output_path))


