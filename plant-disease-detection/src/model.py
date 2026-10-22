import torch.nn as nn
from torchvision import models


def build_model(name: str, num_classes: int, pretrained: bool = True):
    w = "DEFAULT" if pretrained else None
    if name == "mobilenet_v2":
        m = models.mobilenet_v2(weights=w)
        m.classifier[1] = nn.Linear(m.last_channel, num_classes)
    elif name == "efficientnet_b0":
        m = models.efficientnet_b0(weights=w)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)
    elif name == "resnet50":
        m = models.resnet50(weights=w)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
    else:
        raise ValueError(f"Unknown model: {name}")
    return m
