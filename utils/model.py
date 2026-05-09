import torch
import torch.nn as nn
from torchvision.models import efficientnet_b3, EfficientNet_B3_Weights


def build_efficientnetb3(num_classes=2, freeze_backbone=True):
    model = efficientnet_b3(weights=EfficientNet_B3_Weights.IMAGENET1K_V1)

    if freeze_backbone:
        for param in model.features.parameters():
            param.requires_grad = False

    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.Dropout(p=0.3),
        nn.Linear(256, num_classes),
    )

    return model


def unfreeze_backbone(model, depth=None):
    for param in model.features.parameters():
        param.requires_grad = True
    if depth is not None:
        total_layers = len(list(model.features.children()))
        for i, child in enumerate(model.features.children()):
            if i < total_layers - depth:
                for param in child.parameters():
                    param.requires_grad = False
