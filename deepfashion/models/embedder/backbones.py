import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18
from transformers import AutoModel

class ResNet18(nn.Module):
    def __init__(
            self,
            out_features: int = 64,
            freeze_backbone: bool = False
            ):
        super().__init__()
        self.model = resnet18(pretrained=True)
        self.model.fc = nn.Linear(self.model.fc.in_features, out_features, bias=True)
        
    def forward(self, inputs):
        embed = self.model(inputs)
        return embed