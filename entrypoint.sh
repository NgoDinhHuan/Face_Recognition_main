#!/bin/bash
set -e

# Đợi Milvus sẵn sàng (tối đa 60s)
echo "Chờ Milvus sẵn sàng..."
for i in {1..30}; do
    python -c "from pymilvus import connections; connections.connect(host='standalone', port='19530')" && break
    echo "Milvus chưa sẵn sàng, thử lại sau 2s..."
    sleep 2
done

# Tạo collection nếu chưa có
python utils/milvus_setup.py

# Chạy app chính
exec "$@" 