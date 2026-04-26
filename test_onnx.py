import numpy as np
import onnxruntime as ort

from handtracking.models.hand_simcc import INPUT_SIZE, NUM_BINS

sess = ort.InferenceSession("models/hand_simcc.onnx")
h = INPUT_SIZE
img1 = np.zeros((1, 3, h, h), dtype=np.float32)
img2 = np.random.randn(1, 3, h, h).astype(np.float32)

lx1, ly1 = sess.run(None, {"input": img1})
lx2, ly2 = sess.run(None, {"input": img2})


def decode(lx, ly):
    px = np.exp(lx - np.max(lx, axis=-1, keepdims=True))
    px /= px.sum(axis=-1, keepdims=True)
    py = np.exp(ly - np.max(ly, axis=-1, keepdims=True))
    py /= py.sum(axis=-1, keepdims=True)
    xs = np.arange(NUM_BINS) * (INPUT_SIZE / float(NUM_BINS))
    x = (px * xs).sum(axis=-1)
    y = (py * xs).sum(axis=-1)
    return x, y


print(decode(lx1, ly1))
print(decode(lx2, ly2))
