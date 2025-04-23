import os
import cv2
from align.aligner import align_face
from feature.extractor import extract_feature
from utils.faiss_index import add_to_index
import config

def enroll_face(image, name: str, filename: str = None) -> dict:
    aligned = align_face(image)
    if aligned is None:
        return {"success": False, "message": "No face detected"}

    # Trích xuất embedding và lưu
    vector = extract_feature(aligned)
    add_to_index(name, vector)

    # Chuẩn bị tên file để lưu
    if filename:
        safe_name = os.path.basename(filename)
    else:
        safe_name = f"{name}.jpg"

    # Lưu ảnh gốc API gửi lên
    os.makedirs(config.ORIGINAL_IMAGE_DIR, exist_ok=True)
    original_path = os.path.join(config.ORIGINAL_IMAGE_DIR, safe_name)
    cv2.imwrite(original_path, image)

    # Lưu ảnh đã align (đã resize/crop)
    os.makedirs(config.ENROLL_IMAGE_DIR, exist_ok=True)
    aligned_path = os.path.join(config.ENROLL_IMAGE_DIR, safe_name)
    cv2.imwrite(aligned_path, aligned)

    return {
        "success": True,
        "name": name,
        "saved_original": original_path,
        "saved_aligned": aligned_path
    }
