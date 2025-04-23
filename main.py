# main.py

import os
import cv2
from feature.extractor import load_model
from enrollment import enroll_face
from recognition import recognize_face
import config

def test_image(image_path):
    if not os.path.exists(image_path):
        print(" File không tồn tại:", image_path)
        return

    # Lấy tên người từ tên file
    filename = os.path.basename(image_path)
    name = os.path.splitext(filename)[0]

    # Đọc ảnh & load model
    image = cv2.imread(image_path)
    load_model()

    # Kiểm tra xem đã enroll chưa
    embedding_path = os.path.join(config.EMBEDDING_DIR, f"{name}.npy")
    if os.path.exists(embedding_path):
        print(f"Người '{name}' đã được enroll → Bỏ qua bước enroll.")
    else:
        print(f"\n Enrolling '{name}'...")
        enroll_result = enroll_face(image, name, filename)
        print(" Enroll result:", enroll_result)

    # Recognize
    print(f"\n🔍 Recognizing '{filename}'...")
    recog_result = recognize_face(image)

    if recog_result["success"]:
        print(f"Nhận diện thành công: {recog_result['name']} (score = {recog_result['score']:.4f})")
    else:
        print(f" Không nhận diện được. Score = {recog_result.get('score', '-')}")


if __name__ == "__main__":
    # Duyệt tất cả ảnh trong image_test/
    test_folder = config.TEST_IMAGE_DIR
    test_files = [f for f in os.listdir(test_folder) if f.lower().endswith(('.jpg', '.png'))]

    if not test_files:
        print(" Không có ảnh nào trong database/image_test/")
    else:
        for file in test_files:
            print("\n==============================")
            image_path = os.path.join(test_folder, file)
            test_image(image_path)
