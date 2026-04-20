import torch
import torch.nn as nn
from handtracking.models.mobilenet_v4_conv_small import MobileNetV4ConvSmall

class TrueSimCC(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = MobileNetV4ConvSmall(width_mult=0.5)
        self.feat_conv = nn.Conv2d(64, 10, 1)
        self.x_proj = nn.Linear(5, 320)
        self.y_proj = nn.Linear(5, 320)
        
    def forward(self, x):
        f = self.backbone(x)
        f = self.feat_conv(f)
        hx = f.sum(dim=2)
        hy = f.sum(dim=3)
        return self.x_proj(hx), self.y_proj(hy)

model = TrueSimCC()
opt = torch.optim.Adam(model.parameters(), lr=1e-2)
x = torch.randn(1, 3, 160, 160)
tx = torch.randint(0, 320, (1, 10))
ty = torch.randint(0, 320, (1, 10))
ce = nn.CrossEntropyLoss()

for i in range(50):
    lx, ly = model(x)
    loss = ce(lx.view(10, 320), tx.view(10)) + ce(ly.view(10, 320), ty.view(10))
    opt.zero_grad()
    loss.backward()
    opt.step()
    if i == 49: print("Final loss:", loss.item())

