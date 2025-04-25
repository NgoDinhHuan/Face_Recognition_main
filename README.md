 Face Recognition API Module
Dự án nhận diện khuôn mặt sử dụng ONNX model + MTCNN aligner + FAISS + Python class FaceRecognizer.

✅ Thiết kế tách biệt để API có thể dễ dàng import và sử dụng.

📁 Cấu trúc thư mục

Face_Recognition_main/
├── api_interface/
│   └── face_recognizer.py      ← Class duy nhất API cần dùng
│   └── response_utils.py       ← Format JSON chuẩn hóa
├── align/                      ← Căn chỉnh khuôn mặt bằng MTCNN
├── feature/                    ← Trích xuất đặc trưng từ ảnh (ONNX model)
├── utils/                      ← FAISS index và xử lý vector
├── database/
│   ├── images/                 ← Ảnh gốc mỗi người (theo folder)
│   ├── image_enroll/           ← Ảnh đã align
│   ├── image_test/             ← Ảnh dùng để test recognize
│   ├── embeddings/             ← Các vector `.npy`
│   └── id_map.json             ← Lưu id, tên, confidence, thời điểm
├── models/
│   └── edgeface_fp16.onnx      ← Model nhận diện
├── main.py                     ← Script CLI để test
├── config.py                   ← Cấu hình
└── requirements.txt
 Cách sử dụng với API ( FastAPI...)
✅ 1. Import class

from api_interface.face_recognizer import FaceRecognizer

✅ 2. Khởi tạo một lần khi server khởi động

recognizer = FaceRecognizer()

✅ 3. Nhận diện (recognize)

def recognize_api(image_np: np.ndarray):
    result = recognizer.recognize(image_np)
    return result  # JSON chuẩn hóa
image_np phải là ảnh dạng numpy array (shape (H, W, 3), dtype uint8, BGR).

✅ 4. (enroll) – nhiều ảnh cùng một người

def enroll_from_folder(folder_path: str, folder_name: str):
    result = recognizer.enroll_from_folder(folder_path, folder_name)
    return result 
folder_path là thư mục chứa nhiều ảnh .jpg hoặc .png

- Đầu ra recognize() – JSON chuẩn
+ Trường hợp match:

{
  "request_id": "d6e12...",
  "timestamp": "2025-04-23T15:16:00Z",
  "success": true,
  "message": "",
  "result": {
    "matched": true,
    "person_id": "001",
    "person_name": "congphuong",
    "confidence": 0.812
  },
  "processing_time_ms": 142
}
+ Trường hợp không match → tự gán unknown_id:

{
  "request_id": "...",
  "timestamp": "...",
  "success": true,
  "message": "Unknown face - new ID assigned",
  "result": {
    "matched": false,
    "person_id": "006",
    "person_name": "unknown_006",
    "confidence": 0.0
  },
  "processing_time_ms": 129
}
❌ Lỗi xử lý (không detect được mặt):

{
  "success": false,
  "message": "No face detected",
  ...
}
- File id_map.json sau khi enroll

{
  "congphuong": {
    "id": "001",
    "name": "congphuong",
    "confidence": 0.843,
    "enrolled_at": "2025-04-23T15:00:12Z"
  },
  "unknown_006": {
    "id": "006",
    "name": "unknown_006",
    "confidence": 0.0,
    "enrolled_at": "2025-04-23T15:15:30Z"
  }
}
⚙️ Cài đặt thư viện
bash
Sao chép
Chỉnh sửa
pip install -r requirements.txt
Yêu cầu:

onnxruntime (CPU) hoặc onnxruntime-gpu

numpy, opencv-python, faiss-cpu, torch...

