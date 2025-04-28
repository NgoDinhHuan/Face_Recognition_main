import os
import shutil
import uuid
import numpy as np
import cv2
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import List, Optional
import tempfile
from pydantic import BaseModel

from api_interface.face_recognizer import FaceRecognizer

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

@app.get("/")
async def root():
    return {"message": "API Nhận Diện Khuôn Mặt v1.0"}

@app.post("/recognize", response_model=RecognizeResponse)
async def recognize_face(file: UploadFile = File(...)):
    """
    Nhận diện khuôn mặt từ ảnh
    
    - **file**: Tệp ảnh (jpg, png)
    - **Trả về**: Thông tin người được nhận diện hoặc gán ID mới nếu là người lạ
    """
    if not file.filename.lower().endswith((".jpg", ".jpeg", ".png")):
        raise HTTPException(status_code=400, detail="Chỉ hỗ trợ file ảnh .jpg hoặc .png")
    
    try:
        # Đọc ảnh từ file tạm
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Không thể đọc ảnh")
        
        # Nhận diện khuôn mặt
        result = recognizer.recognize(image)
        return result
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi xử lý: {str(e)}")

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