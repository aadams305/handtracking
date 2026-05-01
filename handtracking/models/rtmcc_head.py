"""RTMCCHead: Gated Attention Unit (GAU) based SimCC head for RTMPose.

State-dict key layout matches mmpose's RTMCCHead exactly so pre-trained
weights from ``mmpose/models/heads/coord_cls_heads/rtmcc_head.py`` load
directly (after stripping the ``head.`` prefix).

Key parameters for RTMPose-M hand:
    in_channels=768, out_channels=21 (joints), input_size=(256,256),
    in_featuremap_size=(8,8), simcc_split_ratio=2.0 -> 512 bins per axis,
    gau: hidden_dims=256, s=128
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class ScaleNorm(nn.Module):
    """Scale-normalisation layer (L2 norm + learnable scale)."""

    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.g = nn.Parameter(torch.tensor(dim**0.5))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        norm = torch.norm(x, dim=-1, keepdim=True).clamp(min=self.eps)
        return x / norm * self.g


class Scale(nn.Module):
    """Learnable per-element scale."""

    def __init__(self, dim: int, init_value: float = 1.0) -> None:
        super().__init__()
        self.scale = nn.Parameter(init_value * torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        return x * self.scale


class RTMCCBlock(nn.Module):
    """Gated Attention Unit (self-attention variant) used in RTMPose.

    Matches ``mmpose.models.utils.rtmcc_block.RTMCCBlock`` state dict keys.
    """

    def __init__(
        self,
        num_token: int,
        in_token_dims: int,
        out_token_dims: int,
        expansion_factor: int = 2,
        s: int = 128,
        eps: float = 1e-5,
        dropout_rate: float = 0.0,
        drop_path: float = 0.0,
        act_fn: str = "SiLU",
        use_rel_bias: bool = True,
        pos_enc: bool = False,
    ) -> None:
        super().__init__()
        self.s = s
        self.num_token = num_token
        self.use_rel_bias = use_rel_bias
        self.pos_enc = pos_enc

        self.e = int(in_token_dims * expansion_factor)

        if use_rel_bias:
            self.w = nn.Parameter(torch.rand(2 * num_token - 1))

        self.o = nn.Linear(self.e, out_token_dims, bias=False)
        self.uv = nn.Linear(in_token_dims, 2 * self.e + s, bias=False)
        self.gamma = nn.Parameter(torch.rand(2, s))
        self.beta = nn.Parameter(torch.rand(2, s))

        self.ln = ScaleNorm(in_token_dims, eps=eps)

        nn.init.xavier_uniform_(self.uv.weight)

        if act_fn in ("SiLU", "silu"):
            self.act_fn = nn.SiLU(True)
        elif act_fn in ("ReLU", "relu"):
            self.act_fn = nn.ReLU(True)
        else:
            raise ValueError(f"Unknown activation: {act_fn}")

        self.shortcut = in_token_dims == out_token_dims
        if self.shortcut:
            self.res_scale = Scale(in_token_dims)

        self.sqrt_s = math.sqrt(s)
        self.dropout_rate = dropout_rate
        if dropout_rate > 0.0:
            self.dropout = nn.Dropout(dropout_rate)

    def rel_pos_bias(self, seq_len: int) -> Tensor:
        t = F.pad(self.w[:2 * seq_len - 1], [0, seq_len]).repeat(seq_len)
        t = t[..., :-seq_len].reshape(-1, seq_len, 3 * seq_len - 2)
        r = (2 * seq_len - 1) // 2
        return t[..., r:-r]

    def _rope(self, x: Tensor, dim: int = 1) -> Tensor:
        shape = x.shape
        seq_len = shape[dim]
        half = shape[-1] // 2
        pos = torch.arange(seq_len, device=x.device, dtype=torch.float)
        freq = 10000.0 ** (-torch.arange(half, device=x.device, dtype=torch.float) / half)
        sinusoid = pos[:, None] * freq[None, :]
        sin, cos = sinusoid.sin(), sinusoid.cos()
        while sin.dim() < x.dim():
            sin = sin.unsqueeze(0)
            cos = cos.unsqueeze(0)
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)

    def _forward(self, x: Tensor) -> Tensor:
        x = self.ln(x)
        uv = self.act_fn(self.uv(x))

        u, v, base = torch.split(uv, [self.e, self.e, self.s], dim=2)
        base = base.unsqueeze(2) * self.gamma[None, None, :] + self.beta
        if self.pos_enc:
            base = self._rope(base, dim=1)
        q, k = torch.unbind(base, dim=2)

        qk = torch.bmm(q, k.transpose(1, 2))
        if self.use_rel_bias:
            bias = self.rel_pos_bias(q.size(1))
            qk = qk + bias[:, :q.size(1), :k.size(1)]

        kernel = torch.square(F.relu(qk / self.sqrt_s))
        if self.dropout_rate > 0.0:
            kernel = self.dropout(kernel)
        x = u * torch.bmm(kernel, v)
        return self.o(x)

    def forward(self, x: Tensor) -> Tensor:
        if self.shortcut:
            return self.res_scale(x) + self._forward(x)
        return self._forward(x)


class RTMCCHead(nn.Module):
    """RTMPose SimCC head with GAU attention.

    Args:
        in_channels: Backbone output channels (768 for RTMPose-M).
        out_channels: Number of keypoints (21 for hand).
        input_size: Spatial input resolution (256, 256).
        in_featuremap_size: Feature map spatial size (8, 8) at stride 32.
        simcc_split_ratio: Sub-pixel oversampling factor (2.0 -> 512 bins).
        gau_cfg: Dict configuring the GAU block.
    """

    def __init__(
        self,
        in_channels: int = 768,
        out_channels: int = 21,
        input_size: tuple[int, int] = (256, 256),
        in_featuremap_size: tuple[int, int] = (8, 8),
        simcc_split_ratio: float = 2.0,
        final_layer_kernel_size: int = 7,
        gau_cfg: dict | None = None,
    ) -> None:
        super().__init__()
        if gau_cfg is None:
            gau_cfg = dict(
                hidden_dims=256, s=128, expansion_factor=2,
                dropout_rate=0.0, drop_path=0.0, act_fn="ReLU",
                use_rel_bias=False, pos_enc=False,
            )

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.input_size = input_size
        self.simcc_split_ratio = simcc_split_ratio

        flatten_dims = in_featuremap_size[0] * in_featuremap_size[1]

        self.final_layer = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=final_layer_kernel_size,
            stride=1, padding=final_layer_kernel_size // 2,
        )
        self.mlp = nn.Sequential(
            ScaleNorm(flatten_dims),
            nn.Linear(flatten_dims, gau_cfg["hidden_dims"], bias=False),
        )

        W = int(input_size[0] * simcc_split_ratio)
        H = int(input_size[1] * simcc_split_ratio)
        self.num_bins_x = W
        self.num_bins_y = H

        self.gau = RTMCCBlock(
            out_channels,
            gau_cfg["hidden_dims"],
            gau_cfg["hidden_dims"],
            s=gau_cfg["s"],
            expansion_factor=gau_cfg["expansion_factor"],
            dropout_rate=gau_cfg["dropout_rate"],
            drop_path=gau_cfg["drop_path"],
            act_fn=gau_cfg["act_fn"],
            use_rel_bias=gau_cfg["use_rel_bias"],
            pos_enc=gau_cfg["pos_enc"],
        )

        self.cls_x = nn.Linear(gau_cfg["hidden_dims"], W, bias=False)
        self.cls_y = nn.Linear(gau_cfg["hidden_dims"], H, bias=False)

    def forward(self, feats: Tensor) -> tuple[Tensor, Tensor]:
        """Forward pass.

        Args:
            feats: Feature map [B, C, H, W] from backbone.

        Returns:
            pred_x: [B, K, W] SimCC x-axis logits.
            pred_y: [B, K, H] SimCC y-axis logits.
        """
        feats = self.final_layer(feats)   # [B, K, H, W]
        feats = torch.flatten(feats, 2)   # [B, K, H*W]
        feats = self.mlp(feats)           # [B, K, hidden]
        feats = self.gau(feats)           # [B, K, hidden]
        return self.cls_x(feats), self.cls_y(feats)

    def init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)


if __name__ == "__main__":
    head = RTMCCHead(in_channels=768, out_channels=21)
    x = torch.randn(2, 768, 8, 8)
    px, py = head(x)
    print(f"RTMCCHead: pred_x={px.shape}, pred_y={py.shape}")
    n = sum(p.numel() for p in head.parameters())
    print(f"Params: {n:,}")
