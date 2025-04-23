
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Model
MODEL_PATH = os.path.join(BASE_DIR, "models", "edgeface_fp16.onnx")

# Database folders
DATABASE_DIR = os.path.join(BASE_DIR, "database")
ENROLL_IMAGE_DIR = os.path.join(DATABASE_DIR, "image_enroll")     
TEST_IMAGE_DIR = os.path.join(DATABASE_DIR, "image_test")         
ORIGINAL_IMAGE_DIR = os.path.join(DATABASE_DIR, "images")         
EMBEDDING_DIR = os.path.join(DATABASE_DIR, "embeddings")       
# 
ID_MAP_PATH = os.path.join(BASE_DIR, "database", "id_map.json")
  

# Cấu hình 
IMAGE_SIZE = (112, 112)
THRESHOLD = 0.4
