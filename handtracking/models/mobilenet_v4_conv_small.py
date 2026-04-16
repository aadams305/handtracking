"""
MobileNetV4 Conv-Small style backbone (conv-only, no attention).
Simplified Universal Inverted Bottleneck stack for 160x160 RGB input.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def _make_divisible(v: int, divisor: int = 8) -> int:
    return max(divisor, int(v + divisor / 2) // divisor * divisor)


class ConvBNAct(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel: int = 3,
        stride: int = 1,
        groups: int = 1,
    ) -> None:
        super().__init__()
        pad = kernel // 2
        self.conv = nn.Conv2d(
            in_ch, out_ch, kernel, stride, pad, groups=groups, bias=False
        )
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU6(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class UIBInverted(nn.Module):
    """Universal inverted bottleneck (conv-only): expand -> optional extra dw -> project."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        stride: int,
        expand: int = 4,
        extra_dw: bool = True,
    ) -> None:
        super().__init__()
        hidden = _make_divisible(in_ch * expand)
        self.use_res = stride == 1 and in_ch == out_ch
        layers = [
            ConvBNAct(in_ch, hidden, 1, 1),
        ]
        if extra_dw:
            layers.append(ConvBNAct(hidden, hidden, 3, stride, groups=hidden))
        else:
            layers.append(ConvBNAct(hidden, hidden, 3, stride, groups=hidden))
        layers.append(
            nn.Sequential(
                nn.Conv2d(hidden, out_ch, 1, 1, 0, bias=False),
                nn.BatchNorm2d(out_ch),
            )
        )
        self.conv = nn.Sequential(*layers)
        self.out_act = nn.ReLU6(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv(x)
        if self.use_res:
            y = y + x
        return self.out_act(y)


class MobileNetV4ConvSmall(nn.Module):
    """Outputs spatial feature map before GAP (we expose features for SimCC)."""

    def __init__(self, in_ch: int = 3, width_mult: float = 0.5) -> None:
        super().__init__()
        w = lambda c: _make_divisible(int(c * width_mult))

        self.stem = ConvBNAct(in_ch, w(32), 3, 2)
        c1, c2, c3, c4 = w(24), w(40), w(80), w(128)
        self.stage1 = nn.Sequential(
            UIBInverted(w(32), c1, stride=2, expand=2, extra_dw=True),
            UIBInverted(c1, c1, stride=1, expand=2, extra_dw=True),
        )
        self.stage2 = nn.Sequential(
            UIBInverted(c1, c2, stride=2, expand=4, extra_dw=True),
            UIBInverted(c2, c2, stride=1, expand=4, extra_dw=True),
        )
        self.stage3 = nn.Sequential(
            UIBInverted(c2, c3, stride=2, expand=4, extra_dw=True),
            UIBInverted(c3, c3, stride=1, expand=4, extra_dw=True),
        )
        self.stage4 = nn.Sequential(
            UIBInverted(c3, c4, stride=2, expand=4, extra_dw=True),
            UIBInverted(c4, c4, stride=1, expand=4, extra_dw=True),
        )
        self._out_channels = c4

    @property
    def out_channels(self) -> int:
        return self._out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        return x
