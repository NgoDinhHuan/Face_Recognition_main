# API Nhận Diện Khuôn Mặt

API nhận diện khuôn mặt sử dụng FastAPI kết hợp với module Face Recognition.

## Cài đặt

```bash
# Cài đặt các thư viện cần thiết
pip install -r requirements.txt
```

## Chạy API

```bash
# Chạy API với Uvicorn
python api.py
```

Hoặc:

```bash
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```

API sẽ được khởi chạy tại địa chỉ: http://localhost:8000

## API Documentation

Sau khi khởi chạy, bạn có thể truy cập vào trang documentation tự động tại:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## Các Endpoint

### 1. Nhận diện khuôn mặt

**Endpoint:** `/recognize`

**Method:** POST

**Body:** Form-data với trường `file` là tệp ảnh (jpg, png)

**Curl Example:**
```bash
curl -X POST "http://localhost:8000/recognize" -H "accept: application/json" -H "Content-Type: multipart/form-data" -F "file=@path/to/image.jpg"
```

**Response:**
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

### 2. Đăng ký khuôn mặt mới

**Endpoint:** `/enroll`

**Method:** POST

**Body:** Form-data với:
- `files`: Danh sách tệp ảnh (ít nhất 1 ảnh)
- `person_name`: Tên người cần đăng ký

**Curl Example:**
```bash
curl -X POST "http://localhost:8000/enroll" -H "accept: application/json" -H "Content-Type: multipart/form-data" -F "files=@image1.jpg" -F "files=@image2.jpg" -F "person_name=nguyenvana"
```

**Response:**
```json
{
  "success": true,
  "id": "002",
  "name": "nguyenvana",
  "score": 0.854
}
```

### 3. Lấy danh sách người đã đăng ký

**Endpoint:** `/database`

**Method:** GET

**Curl Example:**
```bash
curl -X GET "http://localhost:8000/database" -H "accept: application/json"
```

**Response:**
```json
{
  "success": true,
  "database": {
    "congphuong": {
      "id": "001",
      "name": "congphuong",
      "confidence": 0.843,
      "enrolled_at": "2025-04-23T15:00:12Z"
    },
    "nguyenvana": {
      "id": "002",
      "name": "nguyenvana",
      "confidence": 0.854,
      "enrolled_at": "2025-04-23T15:30:45Z"
    }
  }
}
```

## Lưu ý

- API này sử dụng module Face Recognition đã được cài đặt sẵn trong dự án
- Khi nhận diện một khuôn mặt mới (chưa từng đăng ký), hệ thống sẽ tự động gán ID và tên theo định dạng "unknown_XXX"
- API hỗ trợ CORS, cho phép gọi từ bất kỳ nguồn nào (origin) 