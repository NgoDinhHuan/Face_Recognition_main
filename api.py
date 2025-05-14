import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import shutil
import uuid
import numpy as np
import cv2
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.responses import RedirectResponse
from typing import List, Optional
import tempfile
from pydantic import BaseModel
import time
from datetime import datetime
import logging

from api_interface.face_recognizer import FaceRecognizer
from utils.milvus_client import get_connection

# Cấu hình logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="API Nhận Diện Khuôn Mặt",
    description="API nhận diện khuôn mặt sử dụng ONNX model + MTCNN aligner",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Khởi tạo FaceRecognizer (chỉ một lần)
recognizer = FaceRecognizer()

# Base model cho response
class RecognizeResponse(BaseModel):
    request_id: str
    timestamp: str
    success: bool
    message: Optional[str] = ""
    result: Optional[dict] = None
    processing_time_ms: int

class MultiRecognizeResponse(BaseModel):
    request_id: str
    timestamp: str
    success: bool
    message: Optional[str] = ""
    results: List[dict]
    total_processing_time_ms: int

@app.get("/health")
async def health_check():
    """Kiểm tra trạng thái của API và Milvus"""
    try:
        # Kiểm tra kết nối Milvus
        milvus_status = get_connection()
        
        return {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "services": {
                "api": "up",
                "milvus": "up" if milvus_status else "down"
            }
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy",
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "services": {
                "api": "up",
                "milvus": "down"
            },
            "error": str(e)
        }

@app.get("/", include_in_schema=False)
async def redirect_to_docs():
    return RedirectResponse(url="/docs")

@app.post("/recognize", response_model=MultiRecognizeResponse)
async def recognize_face(files: List[UploadFile] = File(...)):
    """
    Nhận diện khuôn mặt từ nhiều ảnh
    
    - **files**: Danh sách tệp ảnh (jpg, png)
    - **Trả về**: Thông tin người được nhận diện cho từng ảnh
    """
    if not files:
        raise HTTPException(status_code=400, detail="Cần ít nhất 1 ảnh để nhận diện")
    
    start_time = time.time()
    results = []
    
    for file in files:
        if not file.filename.lower().endswith((".jpg", ".jpeg", ".png")):
            results.append({
                "filename": file.filename,
                "success": False,
                "message": "Chỉ hỗ trợ file ảnh .jpg hoặc .png"
            })
            continue
        
        try:
            # Đọc ảnh từ file tạm
            contents = await file.read()
            nparr = np.frombuffer(contents, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if image is None:
                results.append({
                    "filename": file.filename,
                    "success": False,
                    "message": "Không thể đọc ảnh"
                })
                continue
            
            # Nhận diện khuôn mặt
            result = recognizer.recognize(image)
            result["filename"] = file.filename
            results.append(result)
            
        except Exception as e:
            results.append({
                "filename": file.filename,
                "success": False,
                "message": f"Lỗi xử lý: {str(e)}"
            })
    
    total_time = int((time.time() - start_time) * 1000)
    
    return {
        "request_id": str(uuid.uuid4()),
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "success": any(r.get("success", False) for r in results),
        "results": results,
        "total_processing_time_ms": total_time
    }

@app.post("/enroll")
async def enroll_face(
    files: List[UploadFile] = File(...),
    person_name: str = Form(...)
):
    """
    Đăng ký khuôn mặt mới
    
    - **files**: Danh sách tệp ảnh cùng một người (tối thiểu 1 ảnh)
    - **person_name**: Tên người cần đăng ký
    - **Trả về**: Thông tin đăng ký
    """
    if not files:
        raise HTTPException(status_code=400, detail="Cần ít nhất 1 ảnh để đăng ký")
    
    if not person_name:
        raise HTTPException(status_code=400, detail="Tên người không được để trống")
    
    try:
        # Tạo thư mục tạm để lưu ảnh
        temp_dir = tempfile.mkdtemp()
        
        # Lưu các ảnh vào thư mục tạm
        for i, file in enumerate(files):
            if not file.filename.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
                
            contents = await file.read()
            file_path = os.path.join(temp_dir, f"{i}.jpg")
            
            with open(file_path, "wb") as f:
                f.write(contents)
        
        # Đăng ký khuôn mặt
        result = recognizer.enroll_from_folder(folder_path=temp_dir, folder_name=person_name)
        
        # Xóa thư mục tạm
        shutil.rmtree(temp_dir)
        
        return result
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi đăng ký: {str(e)}")

@app.get("/database")
async def get_database():
    """Lấy danh sách người đã đăng ký trong hệ thống"""
    try:
        if hasattr(recognizer, 'id_map'):
            return {"success": True, "database": recognizer.id_map}
        else:
            return {"success": False, "message": "Chưa có dữ liệu"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 