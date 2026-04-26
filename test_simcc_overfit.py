"""Tiny overfit check: HandSimCCNet + Gaussian soft-CE on one fixed batch."""

import torch
from handtracking.losses import SimCCGaussianSoftCELoss
from handtracking.models.hand_simcc import INPUT_SIZE, HandSimCCNet

device = torch.device("cpu")
model = HandSimCCNet(width_mult=0.5).to(device)
opt = torch.optim.Adam(model.parameters(), lr=3e-3)
loss_fn = SimCCGaussianSoftCELoss().to(device)

x = torch.randn(1, 3, INPUT_SIZE, INPUT_SIZE, device=device)
tgt = torch.rand(1, 21, 2, device=device) * (INPUT_SIZE - 1)

for i in range(3):
    model.train()
    lx, ly = model(x)
    loss = loss_fn(lx, ly, tgt)
    opt.zero_grad()
    loss.backward()
    opt.step()
    if i == 2:
        print("Final loss:", float(loss.detach()))
