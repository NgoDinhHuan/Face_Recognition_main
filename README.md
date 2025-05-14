# Face Recognition API

## Giới thiệu
API nhận diện khuôn mặt sử dụng ONNX model + MTCNN aligner + Milvus vector database. Hệ thống cho phép đăng ký và nhận diện khuôn mặt với độ chính xác cao.

✅ Thiết kế tách biệt để API có thể dễ dàng import và sử dụng.

## Tính năng chính
- Đăng ký khuôn mặt mới (enroll)
- Nhận diện khuôn mặt (recognize)
- Lưu trữ vector đặc trưng trong Milvus
- API RESTful với FastAPI
- Hỗ trợ Docker deployment

## Yêu cầu hệ thống
- Python 3.8+
- Docker và Docker Compose
- Milvus 2.3.4+
- CUDA (tùy chọn, cho GPU acceleration)

## Cài đặt

### 1. Clone repository
```bash
git clone <repository_url>
cd face-recognition-api
```

### 2. Cài đặt dependencies
```bash
pip install -r requirements.txt
```

Yêu cầu thư viện chính:
- onnxruntime (CPU) hoặc onnxruntime-gpu
- numpy, opencv-python, faiss-cpu, torch
- fastapi, uvicorn
- pymilvus

### 3. Cấu trúc thư mục
```
├── api.py                 # FastAPI endpoints
├── main.py               # Script chạy thử nghiệm
├── config.py             # Cấu hình hệ thống
├── api_interface/        # Module nhận diện khuôn mặt
├── feature/             # Trích xuất đặc trưng
├── align/               # Căn chỉnh khuôn mặt
├── utils/               # Tiện ích (FAISS, Milvus)
├── database/            # Lưu trữ dữ liệu
└── models/              # Model ONNX
```

## Sử dụng

### 1. Sử dụng như một module Python
```python
from api_interface.face_recognizer import FaceRecognizer

# Khởi tạo một lần khi server khởi động
recognizer = FaceRecognizer()

# Nhận diện (recognize)
def recognize_api(image_np: np.ndarray):
    result = recognizer.recognize(image_np)
    return result  # JSON chuẩn hóa
# image_np phải là ảnh dạng numpy array (shape (H, W, 3), dtype uint8, BGR)

# Đăng ký (enroll) – nhiều ảnh cùng một người
def enroll_from_folder(folder_path: str, folder_name: str):
    result = recognizer.enroll_from_folder(folder_path, folder_name)
    return result 
# folder_path là thư mục chứa nhiều ảnh .jpg hoặc .png
```

### 2. Chạy với Docker

#### 2.1. Khởi động Milvus
```bash
docker-compose -f docker-compose-milvus.yml up -d
```

#### 2.2. Khởi động API
```bash
docker-compose up -d
```

### 3. Chạy trực tiếp

#### 3.1. Khởi động API
```bash
python api.py
```
hoặc
```bash
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```

### 4. API Endpoints

#### 4.1. Nhận diện khuôn mặt
```http
POST /recognize
```
- Content-Type: multipart/form-data
- Body: files (ảnh jpg/png)

Response khi match:
```json
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
```

Response khi không match:
```json
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
```

#### 4.2. Đăng ký khuôn mặt
```http
POST /enroll
```
- Content-Type: multipart/form-data
- Body: files (ảnh) + person_name

Response:
```json
{
  "success": true,
  "id": "002",
  "name": "jane_doe",
  "images_enrolled": 3
}
```

File id_map.json sau khi enroll:
```json
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
```

#### 4.3. Lấy danh sách người đã đăng ký
```http
GET /database
```

#### 4.4. Kiểm tra trạng thái
```http
GET /health
```

## Cấu hình

### 1. Biến môi trường
```bash
MILVUS_HOST=standalone
MILVUS_PORT=19530
```

### 2. Cấu hình hệ thống (config.py)
```python
IMAGE_SIZE = (112, 112)
THRESHOLD = 0.4
```

## Ví dụ sử dụng

### 1. Python
```python
import requests

# Nhận diện
files = {'files': open('image.jpg', 'rb')}
response = requests.post('http://localhost:8000/recognize', files=files)
print(response.json())

# Đăng ký
files = [('files', open('image1.jpg', 'rb')), ('files', open('image2.jpg', 'rb'))]
data = {'person_name': 'john_doe'}
response = requests.post('http://localhost:8000/enroll', files=files, data=data)
print(response.json())
```

### 2. cURL
```bash
# Nhận diện
curl -X POST "http://localhost:8000/recognize" -H "accept: application/json" -H "Content-Type: multipart/form-data" -F "files=@image.jpg"

# Đăng ký
curl -X POST "http://localhost:8000/enroll" -H "accept: application/json" -H "Content-Type: multipart/form-data" -F "files=@image1.jpg" -F "files=@image2.jpg" -F "person_name=john_doe"
```

## Xử lý lỗi

### 1. Các mã lỗi phổ biến
- 400: Bad Request
- 500: Internal Server Error
- 503: Service Unavailable

### 2. Thông báo lỗi
```json
{
  "success": false,
  "message": "No face detected"
}
```

## Monitoring

### 1. Health Check
```http
GET /health
```
Response:
```json
{
  "status": "healthy",
  "services": {
    "api": "up",
    "milvus": "up"
  }
}
```

### 2. Logs
```bash
# API logs
docker-compose logs -f api

# Milvus logs
docker-compose -f docker-compose-milvus.yml logs -f
```

## Backup và Restore

### 1. Backup
```bash
# Backup Milvus data
docker run --rm -v $(pwd)/volumes/milvus:/milvus_data -v $(pwd)/backup:/backup alpine tar czf /backup/milvus_backup.tar.gz -C /milvus_data .
```

### 2. Restore
```bash
# Restore Milvus data
docker run --rm -v $(pwd)/volumes/milvus:/milvus_data -v $(pwd)/backup:/backup alpine sh -c "rm -rf /milvus_data/* && tar xzf /backup/milvus_backup.tar.gz -C /milvus_data"
```

## Best Practices
1. Luôn kiểm tra health check trước khi sử dụng
2. Backup dữ liệu định kỳ
3. Monitor logs để phát hiện lỗi sớm
4. Sử dụng try-catch khi gọi API
5. Kiểm tra định dạng file trước khi gửi

## Contributing
1. Fork repository
2. Tạo branch mới
3. Commit changes
4. Push lên branch
5. Tạo Pull Request

## License
MIT License

## Contact
- Email: [your-email]
- GitHub: [your-github]

## Acknowledgments
- ONNX Runtime
- MTCNN
- Milvus
- FastAPI

