import uvicorn
from pyngrok import ngrok
import sys
import os

# Định nghĩa port
port = 8000

# Khởi tạo ngrok tunnel
public_url = ngrok.connect(port).public_url
print(f"\n✅ Ngrok đã khởi động! API của bạn có thể truy cập tại: {public_url}")
print(f"API docs có thể truy cập tại: {public_url}/docs")

# Import API
from api import app

# Khởi động server
if __name__ == "__main__":
    try:
        uvicorn.run(app, host="0.0.0.0", port=port)
    except KeyboardInterrupt:
        print("\n⚠️ Đang dừng server và ngrok...")
        ngrok.kill() 