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
    while channels_last corresponds to inputs with shape (batch_size, height, width, channels).
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
    r""" SC Block. There are two equivalent implementations:
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
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim//32),
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


class SConvNet(nn.Module):
    r""" SConvNet
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
                 depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], drop_path_rate=0.,
                 layer_scale_init_value=1e-6, head_init_scale=1.,
                 ):
        super().__init__()

        self.stem_stacked_size = 2
        dim_in = in_chans * (self.stem_stacked_size ** 2)
        dim_out = dims[0]
        self.stem = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=3, padding=1, groups=1),
            LayerNorm(dim_out, eps=1e-6),
            nn.GELU(),
            nn.Conv2d(dim_out, dim_out, kernel_size=1, padding=0),
            LayerNorm(dim_out, eps=1e-6),
            nn.GELU())

        self.stacked_size = [2, 2, 2, 2]
        dim_in = dim_out* (self.stacked_size[0] ** 2)
        self.downsample_layers = nn.ModuleList()  # stem and 3 intermediate downsampling conv layers
        for i in range(4):
            downsample_layer = nn.Sequential(
                nn.Conv2d(dim_in, dim_in, kernel_size=3, padding=1, groups=2**i),
                LayerNorm(dim_in, eps=1e-6),
                nn.GELU(),
                nn.Conv2d(dim_in, dims[i], kernel_size=1, padding=0),
                LayerNorm(dims[i], eps=1e-6),
                nn.GELU())
            self.downsample_layers.append(downsample_layer)
            dim_in = dims[i]*(self.stacked_size[i]**2)

        self.stages = nn.ModuleList()  # 4 feature resolution stages, each consisting of multiple residual blocks
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[SCBlock(dim=dims[i], drop_path=dp_rates[cur + j],
                        layer_scale_init_value=layer_scale_init_value) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        out_channels = 1024
        self.conv_out = nn.Sequential(
            nn.Conv2d(dims[-1], out_channels, kernel_size=1, padding=0),  # spatial-wise conv
            LayerNorm(out_channels, eps=1e-6, data_format="channels_first"),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(p=0.75, inplace=True)
        )

        # self.norm = nn.LayerNorm(out_channels, eps=1e-6)  # final norm layer
        self.head = nn.Linear(out_channels, num_classes)

        self.apply(self._init_weights)
        self.head.weight.data.mul_(head_init_scale)
        self.head.bias.data.mul_(head_init_scale)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    def forward_features(self, x):
        x = F.pixel_unshuffle(x, self.stem_stacked_size)
        x = self.stem(x)
        for i in range(4):
            x = F.pixel_unshuffle(x, self.stacked_size[i])
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        return self.conv_out(x)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x


model_urls = {
    "sconvnet_tiny_1k": "https://dl.fbaipublicfiles.com/sconvnet/sconvnet_tiny_1k_224_ema.pth",
    "sconvnet_small_1k": "https://dl.fbaipublicfiles.com/sconvnet/sconvnet_small_1k_224_ema.pth",
    "sconvnet_base_1k": "https://dl.fbaipublicfiles.com/sconvnet/sconvnet_base_1k_224_ema.pth",
    "sconvnet_large_1k": "https://dl.fbaipublicfiles.com/sconvnet/sconvnet_large_1k_224_ema.pth",
    "sconvnet_tiny_22k": "https://dl.fbaipublicfiles.com/sconvnet/sconvnet_tiny_22k_224.pth",
    "sconvnet_small_22k": "https://dl.fbaipublicfiles.com/sconvnet/sconvnet_small_22k_224.pth",
    "sconvnet_base_22k": "https://dl.fbaipublicfiles.com/sconvnet/sconvnet_base_22k_224.pth",
    "sconvnet_large_22k": "https://dl.fbaipublicfiles.com/sconvnet/sconvnet_large_22k_224.pth",
    "sconvnet_xlarge_22k": "https://dl.fbaipublicfiles.com/sconvnet/sconvnet_xlarge_22k_224.pth",
}


@register_model
def sconvnet_tiny(pretrained=False, in_22k=False, **kwargs):
    model = SConvNet(depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], **kwargs)
    if pretrained:
        url = model_urls['sconvnet_tiny_22k'] if in_22k else model_urls['sconvnet_tiny_1k']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu", check_hash=True)
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def sconvnet_small(pretrained=False, in_22k=False, **kwargs):
    model = SConvNet(depths=[3, 3, 27, 3], dims=[96, 192, 384, 768], **kwargs)
    if pretrained:
        url = model_urls['sconvnet_small_22k'] if in_22k else model_urls['sconvnet_small_1k']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def sconvnet_base(pretrained=False, in_22k=False, **kwargs):
    model = SConvNet(depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024], **kwargs)
    if pretrained:
        url = model_urls['sconvnet_base_22k'] if in_22k else model_urls['sconvnet_base_1k']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def sconvnet_large(pretrained=False, in_22k=False, **kwargs):
    model = SConvNet(depths=[3, 3, 27, 3], dims=[192, 384, 768, 1536], **kwargs)
    if pretrained:
        url = model_urls['sconvnet_large_22k'] if in_22k else model_urls['sconvnet_large_1k']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def sconvnet_xlarge(pretrained=False, in_22k=False, **kwargs):
    model = SConvNet(depths=[3, 3, 27, 3], dims=[256, 512, 1024, 2048], **kwargs)
    if pretrained:
        assert in_22k, "only ImageNet-22K pre-trained sconvnet-XL is available; please set in_22k=True"
        url = model_urls['sconvnet_xlarge_22k']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
    return model