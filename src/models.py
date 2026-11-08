"""Models: our own counting CNN and standard torchvision backbones.

Every model takes a batch of 1-channel images, already scaled to [0, 1]
(shape N x 1 x H x W), and returns the predicted cell count (shape N).
Input normalisation lives inside each model so the training and evaluation
code is identical for all of them.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

TRAIN_COUNT_MEAN = 50.5  # mean of the training targets (src/baseline.py)


def conv_bn_relu(cin, cout, k=3):
    return nn.Sequential(
        nn.Conv2d(cin, cout, k, padding=k // 2, bias=False),  # BN supplies the bias
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
    )


class OwnCountCNN(nn.Module):
    """Small fully convolutional counter.

    Feature extractor (3x3 convs, max-pool /8) -> 1x1 conv -> softplus gives a
    non-negative "density" per output location; the count is the SUM of the
    density map. Counting is additive over image regions, so summing local
    evidence matches the task better than global average pooling + FC
    (that alternative is kept as `pool="avg"` for the ablation).

    widths: channels of c1..c5 (c4 and c5 share the last width).
    """

    def __init__(self, in_hw=(96, 128), widths=(16, 32, 64), pool="sum",
                 pixel_mean=0.0, pixel_std=1.0):
        super().__init__()
        # Train-split pixel statistics; buffers so they are saved with the weights.
        self.register_buffer("pixel_mean", torch.tensor(float(pixel_mean)))
        self.register_buffer("pixel_std", torch.tensor(float(pixel_std)))
        w1, w2, w3 = widths
        self.pool = pool
        self.features = nn.Sequential(
            conv_bn_relu(1, w1),    # c1  H   x W
            nn.MaxPool2d(2),        # p1  H/2 x W/2
            conv_bn_relu(w1, w2),   # c2
            nn.MaxPool2d(2),        # p2  H/4 x W/4
            conv_bn_relu(w2, w3),   # c3
            conv_bn_relu(w3, w3),   # c4
            nn.MaxPool2d(2),        # p3  H/8 x W/8
            conv_bn_relu(w3, w3),   # c5
        )
        n_pos = (in_hw[0] // 8) * (in_hw[1] // 8)
        if pool == "sum":
            self.head = nn.Conv2d(w3, 1, 1)
            # Start with every location predicting mean/n_pos, so the initial
            # count is the training mean (softplus^-1(v) = log(exp(v) - 1)).
            v = TRAIN_COUNT_MEAN / n_pos
            nn.init.zeros_(self.head.weight)
            nn.init.constant_(self.head.bias, math.log(math.expm1(v)))
        elif pool == "avg":
            self.head = nn.Linear(w3, 1)
            nn.init.zeros_(self.head.weight)
            nn.init.constant_(self.head.bias, TRAIN_COUNT_MEAN)
        else:
            raise ValueError(pool)

    def density(self, x):
        x = (x - self.pixel_mean) / self.pixel_std
        return F.softplus(self.head(self.features(x)))  # N x 1 x H/8 x W/8

    def forward(self, x):
        if self.pool == "sum":
            return self.density(x).sum(dim=(1, 2, 3))
        x = (x - self.pixel_mean) / self.pixel_std
        f = self.features(x).mean(dim=(2, 3))
        return self.head(f).squeeze(1)


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class TorchvisionRegressor(nn.Module):
    """Standard backbone with its classifier replaced by a 1-output linear head.

    The gray image is repeated to 3 channels and normalised with ImageNet
    statistics (needed for the pretrained weights; kept for scratch training
    too, so the only difference between the two is the initial weights).
    """

    def __init__(self, arch: str, pretrained: bool):
        super().__init__()
        weights = "DEFAULT" if pretrained else None
        if arch == "resnet18":
            net = torchvision.models.resnet18(weights=weights)
            net.fc = nn.Linear(net.fc.in_features, 1)
            head = net.fc
        elif arch == "mobilenet_v3_small":
            net = torchvision.models.mobilenet_v3_small(weights=weights)
            net.classifier[-1] = nn.Linear(net.classifier[-1].in_features, 1)
            head = net.classifier[-1]
        elif arch == "mobilenet_v3_large":
            net = torchvision.models.mobilenet_v3_large(weights=weights)
            net.classifier[-1] = nn.Linear(net.classifier[-1].in_features, 1)
            head = net.classifier[-1]
        elif arch == "efficientnet_b0":
            net = torchvision.models.efficientnet_b0(weights=weights)
            net.classifier[-1] = nn.Linear(net.classifier[-1].in_features, 1)
            head = net.classifier[-1]
        else:
            raise ValueError(arch)
        nn.init.normal_(head.weight, std=1e-3)
        nn.init.constant_(head.bias, TRAIN_COUNT_MEAN)
        self.net = net
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))

    def forward(self, x):
        x = (x.expand(-1, 3, -1, -1) - self.mean) / self.std
        return self.net(x).squeeze(1)


def build_model(name: str, in_hw, pretrained=False, **kw) -> nn.Module:
    if name == "own":
        return OwnCountCNN(in_hw=in_hw, **kw)
    return TorchvisionRegressor(name, pretrained)
