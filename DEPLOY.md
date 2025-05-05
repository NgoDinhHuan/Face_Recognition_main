# Hướng Dẫn Deploy API Nhận Diện Khuôn Mặt lên Docker

## 1. Yêu Cầu Hệ Thống
- Docker Engine
- Docker Compose
- Ngrok Authtoken (để expose API ra internet)

## 2. Các Bước Deploy

### 2.1. Cài Đặt Docker
- Cài đặt Docker Engine: https://docs.docker.com/engine/install/
- Cài đặt Docker Compose: https://docs.docker.com/compose/install/

### 2.2. Chuẩn Bị Môi Trường
1. Tạo file `.env` trong thư mục gốc của dự án:
```bash
NGROK_AUTHTOKEN=your_ngrok_authtoken_here
```

2. Tạo các thư mục cần thiết:
```bash
mkdir -p database models
touch database/.gitkeep models/.gitkeep
```

### 2.3. Build và Chạy Container
1. Build và chạy container:
```bash
docker-compose up -d --build
```

2. Kiểm tra trạng thái container:
```bash
docker-compose ps
```

3. Xem logs:
```bash
docker-compose logs -f
```

### 2.4. Truy Cập API
- Sau khi container chạy, ngrok sẽ tự động tạo tunnel và hiển thị URL public
- URL API sẽ có dạng: `https://xxxx-xx-xx-xx-xx.ngrok.io`
- API docs: `https://xxxx-xx-xx-xx-xx.ngrok.io/docs`

## 3. Quản Lý Container

### 3.1. Dừng Container
```bash
docker-compose down
```

### 3.2. Khởi Động Lại Container
```bash
docker-compose restart
```

### 3.3. Xóa Container và Image
```bash
docker-compose down --rmi all
```

## 4. Backup Dữ Liệu
- Dữ liệu được lưu trong thư mục `database/` và `models/` sẽ được mount vào container
- Để backup, chỉ cần sao chép các thư mục này

## 5. Troubleshooting

### 5.1. Lỗi Không Kết Nối Được
- Kiểm tra port 8000 đã được mở chưa
- Kiểm tra logs để xem lỗi cụ thể
- Kiểm tra ngrok authtoken có hợp lệ không

### 5.2. Lỗi Không Tìm Thấy Model
- Kiểm tra file model đã được copy vào thư mục `models/` chưa
- Kiểm tra quyền truy cập thư mục `models/`

### 5.3. Lỗi Không Ghi Được Dữ Liệu
- Kiểm tra quyền truy cập thư mục `database/`
- Kiểm tra logs để xem lỗi cụ thể

## 6. Nâng Cấp
1. Pull code mới nhất
2. Build lại container:
```bash
docker-compose up -d --build
```

## 7. Monitoring
- Sử dụng Docker Dashboard để theo dõi tài nguyên
- Xem logs thường xuyên để phát hiện lỗi
- Monitor ngrok tunnel để đảm bảo kết nối ổn định 