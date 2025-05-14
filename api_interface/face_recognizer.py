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
from utils.milvus_client import insert_embedding, search_embedding
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
        image_files = [f for f in os.listdir(folder_path) if f.lower().endswith((".jpg", ".png"))]

        if not image_files:
            return {"success": False, "message": f"No image in folder {folder_name}"}

        success_count = 0

        #  Tạo thư mục riêng cho từng người
        person_image_dir = os.path.join(config.ENROLL_IMAGE_DIR, folder_name)
        person_embedding_dir = os.path.join(config.EMBEDDING_DIR, folder_name)
        os.makedirs(person_image_dir, exist_ok=True)
        os.makedirs(person_embedding_dir, exist_ok=True)

        for img_file in image_files:
            img_path = os.path.join(folder_path, img_file)
            image = cv2.imread(img_path)
            if image is None:
                continue

            aligned = align_face(image)
            if aligned is None:
                continue

            vector = extract_feature(aligned)
            if vector is None:
                continue

            #  Lưu aligned theo người đó
            aligned_filename = os.path.splitext(img_file)[0] + ".jpg"
            aligned_path = os.path.join(person_image_dir, aligned_filename)
            align_rgb = cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB)
            cv2.imwrite(aligned_path, align_rgb)

            #   embedding theo người đó
            # embedding_filename = os.path.splitext(img_file)[0] + ".npy"
            # embedding_path = os.path.join(person_embedding_dir, embedding_filename)
            # np.save(embedding_path, vector)

            #  Add vào FAISS
            unique_name = f"{folder_name}_{os.path.splitext(img_file)[0]}"
            add_to_index(unique_name, vector)
            # Thêm vào Milvus
            insert_embedding(vector.tolist(), unique_name)

            success_count += 1

        if success_count == 0:
            return {"success": False, "message": f"No face detected in {folder_name}"}

        # Gán ID
        if folder_name not in self.id_map:
            new_id = f"{len(self.id_map) + 1:03}"
            self.id_map[folder_name] = {
                "id": new_id,
                "name": folder_name,
                "confidence": 1.0,
                "enrolled_at": datetime.utcnow().isoformat() + "Z"
            }
            self.save_id_map()

        return {
            "success": True,
            "id": self.id_map[folder_name]["id"],
            "name": self.id_map[folder_name]["name"],
            "images_enrolled": success_count
        }

    def recognize(self, image: np.ndarray) -> dict:
        start_time = time.time()
        detailed = {}  # để lưu breakdown timings

        try:
            # 1. Phát hiện và căn chỉnh mặt
            t0 = time.time()
            aligned = align_face(image)
            detailed['detection_ms'] = int((time.time() - t0) * 1000)

            if aligned is None:
                return build_response(
                    success=False,
                    matched=False,
                    person_id="",
                    person_name="",
                    confidence=0,
                    message="No face detected"
                )

            # 2. Trích xuất đặc trưng
            t1 = time.time()
            vector = extract_feature(aligned)
            detailed['embedding_ms'] = int((time.time() - t1) * 1000)

            # 3. Tìm kiếm trong Milvus
            t2 = time.time()
            milvus_result = search_embedding(vector.tolist(), top_k=1)
            detailed['search_ms'] = int((time.time() - t2) * 1000)

            t3 = time.time()

            if milvus_result and milvus_result["name"] is not None:
                full_name = milvus_result["name"]
                score = float(round(milvus_result["score"], 4))
                score = np.clip(score, -1.0, 1.0)
                folder_name = full_name.split("_")[0]
                if score >= config.THRESHOLD and folder_name in self.id_map:
                    person_id   = self.id_map[folder_name]["id"]
                    person_name = self.id_map[folder_name]["name"]
                    response = build_response(
                        success=True,
                        matched=True,
                        person_id=person_id,
                        person_name=person_name,
                        confidence=score
                    )
                else:
                    unknown_id   = f"{len(self.id_map) + 1:03}"
                    unknown_name = f"unknown_{unknown_id}"
                    self.id_map[unknown_name] = {
                        "id": unknown_id,
                        "name": unknown_name,
                        "confidence": 0,
                        "enrolled_at": datetime.utcnow().isoformat() + "Z"
                    }
                    self.save_id_map()
                    response = build_response(
                        success=True,
                        matched=False,
                        person_id=unknown_id,
                        person_name=unknown_name,
                        confidence=0,
                        message="Unknown face - new ID assigned"
                    )
                detailed['postprocess_ms'] = int((time.time() - t3) * 1000)
            else:
                return build_response(
                    success=False,
                    matched=False,
                    person_id="",
                    person_name="",
                    confidence=0,
                    message="No face matched"
                )

        except Exception as e:
            print(f" Error during recognition: {e}")
            response = build_response(
                success=False,
                matched=False,
                person_id="",
                person_name="",
                confidence=0,
                message="Error in face recognition process"
            )

        # --- gán tổng thời gian và detailed breakdown ---
        total_ms = int((time.time() - start_time) * 1000)
        detailed['total_ms']            = total_ms
        response["detailed_timings_ms"] = detailed

        return response


