FROM python:3.9-slim
 
# Các thư viện hệ thống
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
&& rm -rf /var/lib/apt/lists/*

WORKDIR /app
 
# Copy và cài Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip \
&& pip install --no-cache-dir -r requirements.txt
 
# Copy toàn bộ source và model
COPY . .
# Nếu muốn tách riêng: COPY models/edgeface_fp16.onnx /app/models/
 
# Expose cổng ứng dụng
EXPOSE 8000
 
# Lệnh khởi chạy
CMD ["python", "run_with_ngrok.py"]