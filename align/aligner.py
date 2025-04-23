
import cv2
import numpy as np
from PIL import Image
from face_alignment.mtcnn import MTCNN
import config

# Khởi tạo detector 1 lần duy nhất
mtcnn_detector = MTCNN(device='cpu', crop_size=config.IMAGE_SIZE)

def align_face(image: np.ndarray) -> np.ndarray:
    """
    Trả về ảnh đã align (np.ndarray, kích thước chuẩn 112x112)
    hoặc None nếu không detect được
    """
    img_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    try:
        _, faces = mtcnn_detector.align_multi(img_pil, limit=1)
        if not faces or len(faces) == 0:
            return None
        aligned = np.array(faces[0])
        return aligned
    except Exception as e:
        print("Align error:", e)
        return None
