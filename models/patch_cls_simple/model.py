import torch.nn as nn
from torchvision import models


def get_model(n_classes: int):
    # Load Pretrained ResNet
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(
        model.fc.in_features, n_classes
    )  # Adjust the final layer
    return model
