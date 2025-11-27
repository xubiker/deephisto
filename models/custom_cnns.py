import torch
import torch.nn as nn
from torchvision.models import resnet50

class CustomResNet(nn.Module):
    def __init__(self, original_resnet, drop_out=0.5, nb_classes=10):
        super(CustomResNet, self).__init__()
        # 保留原始 ResNet 的所有层
        self.features = nn.Sequential(
            original_resnet.conv1,
            original_resnet.bn1,
            original_resnet.relu,
            original_resnet.maxpool,
            original_resnet.layer1,
            original_resnet.layer2,
            original_resnet.layer3,
            original_resnet.layer4,
            original_resnet.avgpool,
        )
        # 替换原始的 fc 层
        self.num_ftrs = original_resnet.fc.in_features
        self.fc = nn.Sequential(
            nn.Dropout(drop_out),
            nn.Linear(self.num_ftrs, nb_classes),
        )

    def forward(self, x):
        # 提取特征
        x = self.features(x)
        x = torch.flatten(x, 1)
        # 保存 fc 层之前的输出
        latents = x
        # 通过新的 fc 层
        logits = self.fc(x)
        return logits, latents