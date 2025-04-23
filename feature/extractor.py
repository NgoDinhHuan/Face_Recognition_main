
import onnxruntime as ort
import numpy as np
import cv2
import config

session = None

def load_model():
    global session
    if session is None:
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        session = ort.InferenceSession(config.MODEL_PATH, providers=providers)
    return session

def extract_feature(image: np.ndarray) -> np.ndarray:
    global session
    if session is None:
        raise RuntimeError("Model not loaded. Call load_model() first.")

    # Resize đúng kích thước model yêu cầu
    img = cv2.resize(image, config.IMAGE_SIZE)
    img = img.astype(np.float32) / 255.0
    img = (img - 0.5) / 0.5  # normalize to [-1, 1]
    img = np.transpose(img, (2, 0, 1))  # HWC → CHW
    img = np.expand_dims(img, axis=0)  # (1, C, H, W)

    input_name = session.get_inputs()[0].name
    output = session.run(None, {input_name: img})
    return output[0][0]  # vector 128D
