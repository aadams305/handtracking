import cv2
import numpy as np
import onnxruntime as ort
import torch

from handtracking.dataset import normalize_bgr_tensor
from handtracking.geometry import letterbox_image
from handtracking.models.hand_simcc import HandSimCCNet, INPUT_SIZE, NUM_BINS

ckpt = torch.load("checkpoints/hand_simcc.pt", map_location="cpu", weights_only=False)
net = HandSimCCNet(width_mult=0.5)
net.load_state_dict(ckpt["model"], strict=True)
net.eval()

img = cv2.imread("data/demo_images/img_1.jpg")
lb_img, _ = letterbox_image(img, INPUT_SIZE)
inpt = normalize_bgr_tensor(lb_img).unsqueeze(0)

lx, ly = net(inpt)

sess = ort.InferenceSession("models/hand_simcc.onnx")
ox, oy = sess.run(None, {"input": inpt.numpy()})


def decode(lx, ly):
    px = np.exp(lx - np.max(lx, axis=-1, keepdims=True))
    px /= px.sum(axis=-1, keepdims=True)
    py = np.exp(ly - np.max(ly, axis=-1, keepdims=True))
    py /= py.sum(axis=-1, keepdims=True)
    xs = np.arange(NUM_BINS) * (INPUT_SIZE / float(NUM_BINS))
    x = (px * xs).sum(axis=-1)
    y = (py * xs).sum(axis=-1)
    return x, y


print("PyTorch Decode:")
x, y = decode(lx.detach().numpy(), ly.detach().numpy())
for i in range(21):
    print(f"[{x[0, i]:.1f}, {y[0, i]:.1f}]", end=" ")
print("\nONNX Decode:")
x, y = decode(ox, oy)
for i in range(21):
    print(f"[{x[0, i]:.1f}, {y[0, i]:.1f}]", end=" ")
print()
