import os
import json
import cv2
import numpy as np
from api_interface.response_utils import build_response
import time
from datetime import datetime

from feature.extractor import load_model, extract_feature
from align.aligner import align_face
from utils.faiss_index import add_to_index, search_index
import config
from config import ID_MAP_PATH


class FaceRecognizer:
    def __init__(self):
        print(" Loading model...")
        load_model()
        print(" Model ready.")

        if os.path.exists(ID_MAP_PATH):
            with open(ID_MAP_PATH, "r", encoding="utf-8") as f:
                self.id_map = json.load(f)
        else:
            self.id_map = {}

    def save_id_map(self):
        with open(ID_MAP_PATH, "w", encoding="utf-8") as f:
            json.dump(self.id_map, f, indent=2, ensure_ascii=False)

    def enroll_from_folder(self, folder_path: str, folder_name: str) -> dict:
        vectors = []
        image_files = [f for f in os.listdir(folder_path) if f.lower().endswith((".jpg", ".png"))]

        if not image_files:
            return {"success": False, "message": f"No image in folder {folder_name}"}

        aligned_saved = False
        aligned_path = None

        for img_file in image_files:
            img_path = os.path.join(folder_path, img_file)
            image = cv2.imread(img_path)
            if image is None:
                continue

            aligned = align_face(image)
            if aligned is None:
                continue

            vector = extract_feature(aligned)
            vectors.append(vector)

            # Lưu ảnh align đầu tiên
            if not aligned_saved:
                os.makedirs(config.ENROLL_IMAGE_DIR, exist_ok=True)
                aligned_path = os.path.join(config.ENROLL_IMAGE_DIR, f"{folder_name}.jpg")
                cv2.imwrite(aligned_path, aligned)
                aligned_saved = True

        if not vectors:
            return {"success": False, "message": f"No face detected in {folder_name}"}

        # Trung bình vector
        avg_vector = np.mean(vectors, axis=0)
        add_to_index(folder_name, avg_vector)

        # Dùng chính vector trung bình để tính score nhận diện
        result = search_index(avg_vector)
        score = float(round(result["score"], 4))

        # Gán ID nếu chưa có
        if folder_name not in self.id_map:
            new_id = f"{len(self.id_map) + 1:03}"
            self.id_map[folder_name] = {
                "id": new_id,
                "name": folder_name,
                "confidence": score,
                "enrolled_at": datetime.utcnow().isoformat() + "Z"
            }
            self.save_id_map()

        return {
            "success": True,
            "id": self.id_map[folder_name]["id"],
            "name": self.id_map[folder_name]["name"],
            "score": score
        }

    def recognize(self, image: np.ndarray) -> dict:
        start_time = time.time()

        try:
            aligned = align_face(image)
            if aligned is None:
                return build_response(
                    success=False,
                    matched=False,
                    person_id="",
                    person_name="",
                    confidence=0,
                    message="No face detected"
                )

            vector = extract_feature(aligned)
            result = search_index(vector)

            folder_name = result["name"]
            score = float(round(result["score"], 4))

            if score >= config.THRESHOLD and folder_name in self.id_map:
                person_id = self.id_map[folder_name]["id"]
                person_name = self.id_map[folder_name]["name"]

                response = build_response(
                    success=True,
                    matched=True,
                    person_id=person_id,
                    person_name=person_name,
                    confidence=score
                )
            else:
                # Gán ID và tên cho người lạ
                unknown_id = f"{len(self.id_map) + 1:03}"
                unknown_name = f"unknown_{unknown_id}"

                # Ghi vào id_map.json
                self.id_map[unknown_name] = {
                    "id": unknown_id,
                    "name": unknown_name,
                    "confidence": score,
                    "enrolled_at": datetime.utcnow().isoformat() + "Z"
                }
                self.save_id_map()

                response = build_response(
                    success=True,
                    matched=False,
                    person_id=unknown_id,
                    person_name=unknown_name,
                    confidence=score,
                    message="Unknown face - new ID assigned"
                )


        except Exception:
            response = build_response(
                success=False,
                matched=False,
                person_id="",
                person_name="",
                confidence=0,
                message="Error in face recognition process"
            )

        response["processing_time_ms"] = int((time.time() - start_time) * 1000)
        return response
