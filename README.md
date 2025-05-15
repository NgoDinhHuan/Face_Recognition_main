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

### 4. Public API ra Internet với Ngrok

Bạn có thể sử dụng Ngrok để public API ra ngoài internet (dùng cho demo, test nhanh, không khuyến nghị cho production).

#### 4.1. Cài đặt pyngrok
```bash
pip install pyngrok
```

#### 4.2. Chạy API với Ngrok
```bash
python run_with_ngrok.py
```
Sau khi chạy, terminal sẽ hiển thị một đường dẫn public (dạng https://xxxx.ngrok.io). Bạn có thể truy cập API và API docs qua link này.

- Đường dẫn API: `https://xxxx.ngrok.io`
- Đường dẫn docs: `https://xxxx.ngrok.io/docs`

#### 4.3. Ý nghĩa file run_with_ngrok.py
- Tự động khởi động ngrok tunnel cho cổng 8000
- In ra public URL
- Khởi động FastAPI server
- Khi dừng server, ngrok cũng tự dừng

**Lưu ý bảo mật:**
- Không dùng ngrok cho môi trường production hoặc dữ liệu nhạy cảm
- Chỉ dùng cho mục đích demo, thử nghiệm, chia sẻ nhanh

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

## Troubleshooting (Khắc phục sự cố)

### 1. Lỗi cài đặt thư viện
- **Lỗi thiếu thư viện:**
  - Chạy lại: `pip install -r requirements.txt`
- **Lỗi version không tương thích:**
  - Kiểm tra Python >= 3.8, upgrade pip: `pip install --upgrade pip`

### 2. Lỗi khi chạy Docker/Milvus
- **Milvus không khởi động:**
  - Kiểm tra RAM >= 4GB, chạy lại: `docker-compose -f docker-compose-milvus.yml up -d`
  - Xem log: `docker-compose -f docker-compose-milvus.yml logs -f`
- **API không kết nối được Milvus:**
  - Kiểm tra biến môi trường `MILVUS_HOST`, `MILVUS_PORT`
  - Đảm bảo Milvus đã chạy trước khi start API

### 3. Lỗi CUDA/GPU
- **Lỗi CUDA/cuDNN:**
  - Đảm bảo đã cài đúng driver GPU, CUDA, cuDNN
  - Nếu không dùng GPU, cài onnxruntime bản CPU: `pip install onnxruntime`

### 4. Lỗi file lớn khi push lên GitHub
- **Cảnh báo file > 50MB:**
  - Xóa file lớn khỏi git: `git rm --cached <file>`
  - Dùng [Git LFS](https://git-lfs.github.com/) cho file lớn

### 5. Lỗi khi sử dụng Ngrok
- **Không tạo được public URL:**
  - Kiểm tra kết nối internet
  - Đảm bảo chưa có process nào chiếm port 8000
- **API không truy cập được từ ngoài:**
  - Kiểm tra firewall, thử lại với mạng khác

### 6. Lỗi nhận diện/đăng ký khuôn mặt
- **Không detect được mặt:**
  - Kiểm tra ảnh đầu vào (đúng định dạng, rõ mặt)
  - Thử với ảnh khác
- **Kết quả nhận diện sai:**
  - Đảm bảo đã enroll đủ ảnh chất lượng cho mỗi người
  - Kiểm tra lại ngưỡng `THRESHOLD` trong config.py

### 7. Lỗi khác
- **API báo lỗi 500/503:**
  - Xem log API: `docker-compose logs -f api` hoặc log terminal
  - Kiểm tra lại cấu hình, thư viện, kết nối Milvus

Nếu gặp lỗi không nằm trong danh sách trên, hãy kiểm tra log chi tiết và liên hệ để được hỗ trợ thêm.

## License
MIT License

## Acknowledgments
- ONNX Runtime
- MTCNN
- Milvus
- FastAPI

## Mô tả chi tiết các method chính

### 1. FaceRecognizer (api_interface/face_recognizer.py)

#### __init__(self)
- Khởi tạo class, load model nhận diện và đọc file id_map nếu có.

#### save_id_map(self)
- Lưu thông tin id_map (mapping giữa tên và ID) ra file JSON.

#### enroll_from_folder(self, folder_path: str, folder_name: str) -> dict
- Đăng ký khuôn mặt cho một người từ thư mục ảnh.
- **Tham số:**
  - `folder_path`: Đường dẫn thư mục chứa ảnh của một người.
  - `folder_name`: Tên người (dùng làm key).
- **Trả về:** dict thông tin đăng ký (success, id, name, images_enrolled).
- **Mô tả:**
  - Align từng ảnh, trích xuất vector đặc trưng, lưu vào FAISS & Milvus, cập nhật id_map.

#### recognize(self, image: np.ndarray) -> dict
- Nhận diện khuôn mặt từ ảnh numpy array.
- **Tham số:**
  - `image`: Ảnh đầu vào dạng numpy array (BGR).
- **Trả về:** dict kết quả nhận diện (success, matched, person_id, person_name, confidence, ...).
- **Mô tả:**
  - Align ảnh, trích xuất vector, tìm kiếm trong Milvus, trả về thông tin người nhận diện hoặc unknown.

### 2. Các API endpoint (api.py)

#### POST /recognize
- Nhận diện khuôn mặt từ nhiều ảnh (multipart/form-data).
- **Tham số:** files (danh sách ảnh)
- **Trả về:** Thông tin nhận diện cho từng ảnh (MultiRecognizeResponse)

#### POST /enroll
- Đăng ký khuôn mặt mới.
- **Tham số:** files (danh sách ảnh), person_name (tên người)
- **Trả về:** Thông tin đăng ký (id, name, images_enrolled)

#### GET /database
- Lấy danh sách người đã đăng ký.

#### GET /health
- Kiểm tra trạng thái API và Milvus.

### 3. Một số method hỗ trợ quan trọng

#### align_face(image: np.ndarray) -> np.ndarray (align/aligner.py)
- Căn chỉnh khuôn mặt về kích thước chuẩn (112x112), trả về ảnh đã align hoặc None nếu không phát hiện được mặt.

#### extract_feature(image: np.ndarray) -> np.ndarray (feature/extractor.py)
- Trích xuất vector đặc trưng từ ảnh khuôn mặt đã align.

#### add_to_index(name: str, vector: np.ndarray) (utils/faiss_index.py)
- Thêm vector đặc trưng vào FAISS index để tìm kiếm nhanh.

#### search_embedding(vector: list, top_k=1) (utils/milvus_client.py)
- Tìm kiếm vector gần nhất trong Milvus, trả về thông tin người gần nhất.

---

**Ví dụ sử dụng FaceRecognizer trong Python:**
```python
from api_interface.face_recognizer import FaceRecognizer
import cv2

recognizer = FaceRecognizer()

# Đăng ký khuôn mặt
result = recognizer.enroll_from_folder('data/john_doe', 'john_doe')
print(result)

# Nhận diện khuôn mặt
img = cv2.imread('test.jpg')
result = recognizer.recognize(img)
print(result)

