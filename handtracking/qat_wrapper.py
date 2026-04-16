"""QuantStub/DeQuantStub wrapper for QAT (INT8 targets)."""

from __future__ import annotations

import torch
import torch.nn as nn


class QATSimCCWrapper(nn.Module):
    def __init__(self, inner: nn.Module) -> None:
        super().__init__()
        self.quant = torch.ao.quantization.QuantStub()
        self.inner = inner
        self.dequant_x = torch.ao.quantization.DeQuantStub()
        self.dequant_y = torch.ao.quantization.DeQuantStub()

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.quant(x)
        lx, ly = self.inner(x)
        return self.dequant_x(lx), self.dequant_y(ly)


def apply_qat_prepare(model: nn.Module) -> nn.Module:
    model.qconfig = torch.ao.quantization.get_default_qat_qconfig("qnnpack")
    torch.ao.quantization.prepare_qat(model, inplace=True)
    return model
