import onnxruntime as ort
import numpy as np
sess = ort.InferenceSession("models/hand_simcc.onnx", providers=["CPUExecutionProvider"])
in_name = sess.get_inputs()[0].name
for i in range(3):
    img = np.random.randn(1,3,160,160).astype(np.float32) * (i+1)
    outx, outy = sess.run(None, {in_name: img})
    print(i, outx[0,0,:5]) # print first 5 bins of joint 0
