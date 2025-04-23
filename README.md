 Face Recognition 
Hệ thống này giúp bạn:

✅ Nhận diện khuôn mặt từ ảnh gửi lên

✅ Enroll người mới từ nhiều ảnh (1 người có nhiều ảnh)

✅ Gán ID tự động, lưu vector nhúng (embedding) trung bình

✅ Trả về thông tin: id, name, score

🧩 Công nghệ sử dụng

Thành phần	Mô tả
Align	MTCNN (định vị khuôn mặt trước khi trích đặc trưng)
Model	EdgeFace (ONNX – float16)
Vector nhúng	512 chiều (128 cũng hỗ trợ tùy model)
So sánh	Cosine Similarity qua FAISS
⚙️ Class API cần sử dụng
"from api_interface.face_recognizer import FaceRecognizer"
Khởi tạo:
recognizer = FaceRecognizer()

📁 Cấu trúc dự án
Face_Recognition_Main/
├── api_interface/                            #  Class duy nhất API sử dụng
│   └── face_recognizer.py                    #  Gói toàn bộ logic enroll + recognize
│
├── align/                                    #  MTCNN face alignment
│   └── aligner.py                            #  Hàm align_face(image) → 112x112
│
├── feature/                                  #  Load model + trích xuất embedding
│   └── extractor.py                          #  extract_feature(aligned) → vector (512D)
│
├── utils/                                    #  FAISS index add/search
│   ├── faiss_index.py                        #    FAISS index wrapper
│                         
│
├── database/                                 #  Tất cả dữ liệu lưu trữ
│   ├── images/                               #    Ảnh gốc của từng người (mỗi người 1 folder)
│   │   ├── van/                              #    → chứa: 1.jpg, 2.jpg, ...
│   │   └── huan/
│   ├── image_enroll/                         #    Ảnh align đầu tiên sau khi enroll
│   │   └── van.jpg
│   ├── embeddings/                           #    Vector `.npy` trung bình của từng người
│   │   └── van.npy
│   ├── image_test/                           #    Ảnh test nhận diện từ local
│   │   └── unknown1.jpg
│   └── id_map.json                           #    Ánh xạ name → id (VD: "van": {"id": "001"})
│
├── models/                                   #  Chứa model ONNX đã convert
│   └── edgeface_fp16.onnx
│
├── main.py                                   # File CLI test: enroll & recognize từ local
├── app.py                                    #  (Tuỳ chọn) FastAPI app: chạy API server
├── config.py                                 # Đường dẫn, threshold, id_map, model_path
├── requirements.txt                          #  Các thư viện cần cài đặt (cv2, numpy, faiss, ...)
├── README.md                                 #  Hướng dẫn tổng quan hệ thống                         

🚀 Các API endpoint cần xây duwng
1. Nhận diện 1 ảnh – POST /recognize
@app.post("/recognize")
async def recognize_face(file: UploadFile = File(...)):
    contents = await file.read()
    image = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)
    result = recognizer.recognize(image)
    return result
✅ Input:
file: ảnh khuôn mặt (jpg/png)

Kiểu multipart/form-data

✅ Output:
json
{
  "success": true,
  "id": "002",
  "name": "Van Nguyen",
  "score": 0.9812
}
Hoặc:

json
{
  "success": false,
  "message": "Unknown face",
  "score": 0.32
}
2. Enroll người mới – POST /enroll
@app.post("/enroll")
async def enroll_person(files: List[UploadFile] = File(...), name: str = Form(...)):
    images = []
    for file in files:
        contents = await file.read()
        img = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)
        if img is not None:
            images.append(img)

    result = recognizer.enroll_from_images(images, folder_name=name)
    return result
✅ Input:
files: danh sách ảnh (jpg/png) của cùng một người

name: tên hoặc định danh thư mục người đó

Kiểu: multipart/form-data

✅ Output:
json
{
  "success": true,
  "id": "003",
  "name": "ngoc",
  "score": 0.9947
}
 Luồng xử lý API
✅ Enroll
API gửi nhiều ảnh + tên → recognizer.enroll_from_images(...)

Hệ thống align → extract → tính vector trung bình

Gán ID nếu chưa có

Lưu ảnh align đầu tiên và vector .npy

Trả về: {success, id, name, score}

✅ Recognize
API gửi 1 ảnh

Align → extract → so sánh với tất cả vector đã lưu (*.npy)

Nếu score > ngưỡng (THRESHOLD) → trả về ID người đó
✅ Yêu cầu tích hợp

Hệ thống cần từ API
Giải mã ảnh UploadFile → np.ndarray 
Duy trì một recognizer = FaceRecognizer() duy nhất
Gửi đúng multipart/form-data với tên và ảnh


👨‍💻 Liên hệ
Dự án được phát triển bởi Ngô Đình Huân ☀️

---^-^---
