import os
import cv2
import argparse
import json
from api_interface.face_recognizer import FaceRecognizer
import config

recognizer = FaceRecognizer()

def enroll_from_images():
    image_dir = config.ORIGINAL_IMAGE_DIR
    person_dirs = [d for d in os.listdir(image_dir) if os.path.isdir(os.path.join(image_dir, d))]

    print(f" Tổng số người cần enroll: {len(person_dirs)}")

    for person in person_dirs:
        person_embedding_folder = os.path.join(config.EMBEDDING_DIR, person)

        #  Nếu người này đã có folder embeddings --> bỏ qua
        if os.path.exists(person_embedding_folder) and os.listdir(person_embedding_folder):
            continue

        person_path = os.path.join(image_dir, person)
        images = [f for f in os.listdir(person_path) if f.lower().endswith((".jpg", ".png"))]

        if not images:
            print(f" Không tìm thấy ảnh trong: {person_path}")
            continue

        result = recognizer.enroll_from_folder(folder_path=person_path, folder_name=person)

def recognize_from_test():
    test_dir = config.TEST_IMAGE_DIR
    files = [f for f in os.listdir(test_dir) if f.lower().endswith((".jpg", ".png"))]

    for file in files:
        image_path = os.path.join(test_dir, file)
        image = cv2.imread(image_path)
        if image is None:
            print(f" Không đọc được ảnh: {file}")
            continue

        result = recognizer.recognize(image)
        print(f"\n  Test: {file}")
        print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["enroll", "recognize", "both"],
        default="both", 
        help="Chọn chế độ: enroll / recognize / both (mặc định: both)"
    )
    args = parser.parse_args()

    if args.mode == "enroll":
        enroll_from_images()

    elif args.mode == "recognize":
        enroll_from_images() 
        recognize_from_test()

    elif args.mode == "both":
        enroll_from_images()
        recognize_from_test()
