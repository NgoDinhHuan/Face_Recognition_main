
import onnxruntime as ort
import numpy as np
import cv2
import config

session = None
input_dtype = None

def load_model():
    global session, input_dtype
    if session is None:
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        session = ort.InferenceSession(config.MODEL_PATH, providers=providers)

        # Tự xác định kiểu input model yêu cầu
        input_dtype_str = session.get_inputs()[0].type
        if "float16" in input_dtype_str:
            input_dtype = np.float16
        else:
            input_dtype = np.float32

    return session

def extract_feature(image: np.ndarray) -> np.ndarray:
    global session, input_dtype
    if session is None or input_dtype is None:
        raise RuntimeError("Model not loaded. Call load_model() first.")

    img = cv2.resize(image, config.IMAGE_SIZE)
    img = img.astype(np.float32) / 255.0
    img = (img - 0.5) / 0.5
    img = np.transpose(img, (2, 0, 1))  # HWC → CHW
    img = np.expand_dims(img, axis=0)  # (1, 3, H, W)
    img = img.astype(input_dtype)      # tự động chuyển float16 hoặc float32

    input_name = session.get_inputs()[0].name
    output = session.run(None, {input_name: img})
    return output[0][0]
