# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath
from timm.models.registry import register_model

class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_first (default) or channels_last.
    The channels_first corresponds to inputs with shape (batch_size, channels, height, width),
    while channels_last corresponds to inputs with shape (batch_size, height, width, channels) .
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_first"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation Block proposed in SENet (https://arxiv.org/abs/1709.01507)
    We assume the inputs to this layer are channels_first (batch_size, channels, height, width).
    """

    def __init__(self, in_channel, hidden_dim):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Conv2d(in_channel, hidden_dim, (1, 1), bias=True)
        self.fc2 = nn.Conv2d(hidden_dim, in_channel, (1, 1), bias=True)
        self.act1 = nn.ReLU(inplace=True)
        self.act2 = nn.Sigmoid()

    def forward(self, inputs):
        x = self.avg_pool(inputs)
        x = self.act1(self.fc1(x))
        x = self.act2(self.fc2(x))
        output = x * inputs
        return output


class SCBlock(nn.Module):
    r""" stcnet Block. There are two equivalent implementations:
    (1) SwConv -> LayerNorm (channels_first) -> ReLu -> 1x1 Conv -> LayerNorm (channels_first) -> SE -> 1x1 Conv -> LayerNorm (channels_first); all in (N, C, H, W)
    (2) SwConv -> Permute to (N, H, W, C)-> LayerNorm (channels_last) -> Linear -> LayerNorm (channels_first) -> SE -> Linear -> LayerNorm (channels_first) -> Permute back
    We use (2) as we find it slightly faster in PyTorch

    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """

    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.scn_spatial = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1),
            LayerNorm(dim, eps=1e-6),
            nn.GELU())

        self.scn_se = SEBlock(dim, dim)

        self.scn_channel = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1, padding=0),
            LayerNorm(dim, eps=1e-6),
            nn.GELU())

        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones(dim),requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.scn_spatial(x)
        x = self.scn_se(x)
        x = self.scn_channel(x)
        if self.gamma is not None:
            x = self.gamma.view(1, -1, 1, 1) * x

        x = input + self.drop_path(x)

        return x


class STCNet(nn.Module):
    r""" stcnet
        A PyTorch impl of : `A ConvNet for the 2020s`  -
          https://arxiv.org/pdf/2201.03545.pdf

    Args:
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 9, 3]
        dims (int): Feature dimension at each stage. Default: [96, 192, 384, 768]
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
        head_init_scale (float): Init scaling value for classifier weights and biases. Default: 1.
    """

    def __init__(self, in_chans=3, num_classes=1000,
                 depths=[3, 3, 3, 3], dims=[64, 128, 256, 512], drop_path_rate=0.,
                 layer_scale_init_value=1e-6, head_init_scale=1.,
                 ):
        super().__init__()

        self.stage_stem = nn.Sequential(
            nn.Conv2d(in_chans*4, dims[0], kernel_size=3, padding=1),  # spatial-wise conv
            LayerNorm(dims[0], eps=1e-6),
            nn.GELU(),
            nn.Conv2d(dims[0], dims[0], kernel_size=1, padding=0),
            LayerNorm(dims[0], eps=1e-6),
            nn.GELU())

        cur = 0
        dim_in = 4*dims[0]
        self.stages_head = nn.ModuleList()  # 4 feature resolution stages, each consisting of multiple residual blocks
        self.stages_body = nn.ModuleList()  # 4 feature resolution stages, each consisting of multiple residual blocks
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        for i in range(4):
            stage_head = nn.Sequential(
                nn.Conv2d(dim_in, dims[i], kernel_size=3, padding=1),  # spatial-wise conv
                LayerNorm(dims[i], eps=1e-6),
                nn.GELU(),
                nn.Conv2d(dims[i], dims[i], kernel_size=1, padding=0),
                LayerNorm(dims[i], eps=1e-6),
                nn.GELU())
            self.stages_head.append(stage_head)

            stage_body = nn.Sequential(
                *[SCBlock(dim=dims[i], drop_path=dp_rates[cur + j],layer_scale_init_value=layer_scale_init_value) for j in range(1, depths[i])]
            )
            self.stages_body.append(stage_body)
            dim_in = 4 * dims[i]
            cur += depths[i]

        out_channels = 1024
        self.stage_out = nn.Sequential(
            nn.Conv2d(dims[-1], out_channels, kernel_size=1, padding=0),  # spatial-wise conv
            LayerNorm(out_channels, eps=1e-6),
            nn.GELU(),
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(p=0.75, inplace=True)
        self.classifier = nn.Linear(out_channels, num_classes)

        self.apply(self._init_weights)
        self.classifier.weight.data.mul_(head_init_scale)
        self.classifier.bias.data.mul_(head_init_scale)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = F.pixel_unshuffle(x, 2)
        x = self.stage_stem(x)
        for i in range(4):
            x = F.pixel_unshuffle(x, 2)
            x = self.stages_head[i](x)
            x = self.stages_body[i](x)
        x = self.stage_out(x)
        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.dropout(x)
        x = self.classifier(x)
        return x


model_urls = {
    "stcnet_tiny_1k": "https://dl.fbaipublicfiles.com/stcnet/stcnet_tiny_1k_224_ema.pth",
    "stcnet_small_1k": "https://dl.fbaipublicfiles.com/stcnet/stcnet_small_1k_224_ema.pth",
    "stcnet_base_1k": "https://dl.fbaipublicfiles.com/stcnet/stcnet_base_1k_224_ema.pth",
    "stcnet_large_1k": "https://dl.fbaipublicfiles.com/stcnet/stcnet_large_1k_224_ema.pth",
    "stcnet_tiny_22k": "https://dl.fbaipublicfiles.com/stcnet/stcnet_tiny_22k_224.pth",
    "stcnet_small_22k": "https://dl.fbaipublicfiles.com/stcnet/stcnet_small_22k_224.pth",
    "stcnet_base_22k": "https://dl.fbaipublicfiles.com/stcnet/stcnet_base_22k_224.pth",
    "stcnet_large_22k": "https://dl.fbaipublicfiles.com/stcnet/stcnet_large_22k_224.pth",
    "stcnet_xlarge_22k": "https://dl.fbaipublicfiles.com/stcnet/stcnet_xlarge_22k_224.pth",
}


@register_model
def stcnet_tiny(pretrained=False, in_22k=False, **kwargs):
    model = STCNet(depths=[3, 3, 6, 3], dims=[64, 128, 256, 512], **kwargs)
    if pretrained:
        url = model_urls['stcnet_tiny_22k'] if in_22k else model_urls['stcnet_tiny_1k']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu", check_hash=True)
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def stcnet_small(pretrained=False, in_22k=False, **kwargs):
    model = STCNet(depths=[3, 3, 27, 3], dims=[96, 192, 384, 768], **kwargs)
    if pretrained:
        url = model_urls['stcnet_small_22k'] if in_22k else model_urls['stcnet_small_1k']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def stcnet_base(pretrained=False, in_22k=False, **kwargs):
    model = STCNet(depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024], **kwargs)
    if pretrained:
        url = model_urls['stcnet_base_22k'] if in_22k else model_urls['stcnet_base_1k']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def stcnet_large(pretrained=False, in_22k=False, **kwargs):
    model = STCNet(depths=[3, 3, 27, 3], dims=[192, 384, 768, 1536], **kwargs)
    if pretrained:
        url = model_urls['stcnet_large_22k'] if in_22k else model_urls['stcnet_large_1k']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def stcnet_xlarge(pretrained=False, in_22k=False, **kwargs):
    model = STCNet(depths=[3, 3, 27, 3], dims=[256, 512, 1024, 2048], **kwargs)
    if pretrained:
        assert in_22k, "only ImageNet-22K pre-trained stcnet-XL is available; please set in_22k=True"
        url = model_urls['stcnet_xlarge_22k']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
    return model
