"""Standalone CSPNeXt backbone (P5 architecture) compatible with mmpose/mmdet weights.

Layer naming and structure exactly match mmdetection's CSPNeXt so that pretrained
state_dict keys (prefixed with ``backbone.``) can be loaded directly.

RTMPose-M config: arch='P5', deepen_factor=0.67, widen_factor=0.75, channel_attention=True
-> output channels = int(1024 * 0.75) = 768 at stride 32 (8×8 for 256px input)
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class ConvModule(nn.Module):
    """Conv2d + BatchNorm2d + SiLU, matching mmcv ConvModule state_dict layout."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        groups: int = 1,
        bias: bool = False,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, groups=groups, bias=bias,
        )
        self.bn = nn.BatchNorm2d(out_channels, momentum=0.03, eps=0.001)
        self.activate = nn.SiLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        return self.activate(self.bn(self.conv(x)))


class DepthwiseSeparableConvModule(nn.Module):
    """Depthwise separable conv matching mmcv's state_dict layout.

    Keys: ``depthwise_conv.{conv,bn}.*``, ``pointwise_conv.{conv,bn}.*``
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
    ) -> None:
        super().__init__()
        self.depthwise_conv = ConvModule(
            in_channels, in_channels, kernel_size,
            stride=stride, padding=padding, groups=in_channels,
        )
        self.pointwise_conv = ConvModule(in_channels, out_channels, 1)

    def forward(self, x: Tensor) -> Tensor:
        return self.pointwise_conv(self.depthwise_conv(x))


class ChannelAttention(nn.Module):
    """Channel attention (squeeze-excite) matching mmdet's ChannelAttention."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(channels, channels, 1, 1, 0, bias=True)
        self.act = nn.Hardsigmoid(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        with torch.cuda.amp.autocast(enabled=False):
            out = self.global_avgpool(x)
        out = self.fc(out)
        out = self.act(out)
        return x * out


class CSPNeXtBlock(nn.Module):
    """Single bottleneck block used inside CSPLayer (CSPNeXt variant).

    conv1: 3×3 regular conv  (or depthwise-sep if use_depthwise)
    conv2: 5×5 depthwise-separable conv
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        expansion: float = 1.0,
        add_identity: bool = True,
        use_depthwise: bool = False,
    ) -> None:
        super().__init__()
        hidden = int(out_channels * expansion)
        if use_depthwise:
            self.conv1 = DepthwiseSeparableConvModule(in_channels, hidden, 3, padding=1)
        else:
            self.conv1 = ConvModule(in_channels, hidden, 3, padding=1)
        self.conv2 = DepthwiseSeparableConvModule(hidden, out_channels, 5, padding=2)
        self.add_identity = add_identity and in_channels == out_channels

    def forward(self, x: Tensor) -> Tensor:
        out = self.conv2(self.conv1(x))
        return out + x if self.add_identity else out


class CSPLayer(nn.Module):
    """Cross Stage Partial layer with optional channel attention."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        expand_ratio: float = 0.5,
        num_blocks: int = 1,
        add_identity: bool = True,
        use_depthwise: bool = False,
        channel_attention: bool = False,
    ) -> None:
        super().__init__()
        mid = int(out_channels * expand_ratio)
        self.channel_attention = channel_attention
        self.main_conv = ConvModule(in_channels, mid, 1)
        self.short_conv = ConvModule(in_channels, mid, 1)
        self.final_conv = ConvModule(2 * mid, out_channels, 1)
        self.blocks = nn.Sequential(*(
            CSPNeXtBlock(mid, mid, 1.0, add_identity, use_depthwise)
            for _ in range(num_blocks)
        ))
        if channel_attention:
            self.attention = ChannelAttention(2 * mid)

    def forward(self, x: Tensor) -> Tensor:
        x_short = self.short_conv(x)
        x_main = self.blocks(self.main_conv(x))
        x_final = torch.cat((x_main, x_short), dim=1)
        if self.channel_attention:
            x_final = self.attention(x_final)
        return self.final_conv(x_final)


class SPPBottleneck(nn.Module):
    """Spatial Pyramid Pooling (SPP) bottleneck."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_sizes: tuple[int, ...] = (5, 9, 13),
    ) -> None:
        super().__init__()
        mid = in_channels // 2
        self.conv1 = ConvModule(in_channels, mid, 1)
        self.poolings = nn.ModuleList(
            nn.MaxPool2d(k, stride=1, padding=k // 2) for k in kernel_sizes
        )
        self.conv2 = ConvModule(mid * (len(kernel_sizes) + 1), out_channels, 1)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        with torch.cuda.amp.autocast(enabled=False):
            x = torch.cat([x] + [p(x) for p in self.poolings], dim=1)
        return self.conv2(x)


# P5 architecture: (in_ch, out_ch, num_blocks, add_identity, use_spp)
_P5_ARCH = [
    (64, 128, 3, True, False),
    (128, 256, 6, True, False),
    (256, 512, 6, True, False),
    (512, 1024, 3, False, True),
]


class CSPNeXt(nn.Module):
    """CSPNeXt backbone (P5) for RTMPose.

    Args:
        widen_factor: Width multiplier (0.75 for RTMPose-M).
        deepen_factor: Depth multiplier (0.67 for RTMPose-M).
        out_indices: Which stages to return (default: only stage4 for pose).
        channel_attention: Whether to use channel attention in CSPLayers.
    """

    def __init__(
        self,
        widen_factor: float = 0.75,
        deepen_factor: float = 0.67,
        out_indices: tuple[int, ...] = (4,),
        channel_attention: bool = True,
    ) -> None:
        super().__init__()
        self.out_indices = out_indices

        base_ch = int(_P5_ARCH[0][0] * widen_factor)
        half_ch = base_ch // 2
        self.stem = nn.Sequential(
            ConvModule(3, half_ch, 3, stride=2, padding=1),
            ConvModule(half_ch, half_ch, 3, stride=1, padding=1),
            ConvModule(half_ch, base_ch, 3, stride=1, padding=1),
        )

        self.layers = ["stem"]
        for i, (in_ch, out_ch, n_blocks, add_id, use_spp) in enumerate(_P5_ARCH):
            in_ch = int(in_ch * widen_factor)
            out_ch = int(out_ch * widen_factor)
            n_blocks = max(round(n_blocks * deepen_factor), 1)

            stage = []
            stage.append(ConvModule(in_ch, out_ch, 3, stride=2, padding=1))
            if use_spp:
                stage.append(SPPBottleneck(out_ch, out_ch))
            stage.append(CSPLayer(
                out_ch, out_ch,
                num_blocks=n_blocks,
                add_identity=add_id,
                channel_attention=channel_attention,
            ))
            name = f"stage{i + 1}"
            self.add_module(name, nn.Sequential(*stage))
            self.layers.append(name)

        last_out = int(_P5_ARCH[-1][1] * widen_factor)
        self.out_channels = last_out

    def forward(self, x: Tensor) -> Tensor:
        """Returns the feature map from the last requested stage."""
        outs = []
        for i, name in enumerate(self.layers):
            layer = getattr(self, name)
            x = layer(x)
            if i in self.out_indices:
                outs.append(x)
        if len(outs) == 1:
            return outs[0]
        return tuple(outs)

    def init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5), mode="fan_in")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)


if __name__ == "__main__":
    net = CSPNeXt(widen_factor=0.75, deepen_factor=0.67, channel_attention=True)
    x = torch.randn(1, 3, 256, 256)
    out = net(x)
    print(f"CSPNeXt-M output: {out.shape}")
    print(f"Out channels: {net.out_channels}")
    n = sum(p.numel() for p in net.parameters())
    print(f"Params: {n:,}")
