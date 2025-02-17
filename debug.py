# %%
import os 
os.environ['CUDA_VISIBLE_DEVICES'] = "1"
import torch
import matplotlib.pylab as plt 
import numpy as np
from models.graph_hnet_pseudo import Graph_HNet
from pathlib import Path
from patch_samplers.full_samplers import (
    FullImageDenseSampler,
    FullImageRndSampler,
)
from typing import Callable

import numpy as np
from PIL import Image
from psimage.image import PSImage
from psimage.patches import Patch
import torch
from tqdm import tqdm

from anno.utils import AnnoDescription

# %%
img_path = Path(
        "/home/data_repository/PATH-DT-MSU_dev/WSS2_v2_psi/test/test_01.psi"
    )
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
patch_sampler = None
if random_sampler:
    patch_sampler = FullImageRndSampler(
        img_path, layer=layer, patch_size=224, batch_size=64
    )
else:
    patch_sampler = FullImageDenseSampler(
        img_path, layer=layer, patch_size=224, batch_size=64, stride=224
    )

# %%
for inputs, coords, filled_ratio in patch_sampler.generator_torch():
        print(inputs.shape, coords.shape, filled_ratio)

# %%



