import os
import numpy as np
import faiss
import pickle
import config

# Đường dẫn lưu FAISS index và mapping
FAISS_INDEX_PATH = os.path.join(config.DATABASE_DIR, "face_index.faiss")
FAISS_MAP_PATH = os.path.join(config.DATABASE_DIR, "faiss_map.pkl")

# FAISS index và mapping tên-vector
faiss_index = None
faiss_map = []  # list tên theo thứ tự index

# Tham số HNSW
DIM = 512
M = 16
EF_CONSTRUCTION = 100
EF_SEARCH = 50


def build_faiss_index():
    """Load toàn bộ embedding vào FAISS HNSW index"""
    global faiss_index, faiss_map
    embeddings = []
    faiss_map = []
    for root, dirs, files in os.walk(config.EMBEDDING_DIR):
        for fname in files:
            if fname.endswith(".npy"):
                name = os.path.splitext(fname)[0]
                vector = np.load(os.path.join(root, fname)).astype(np.float32)
                folder_name = os.path.basename(root)
                full_name = f"{folder_name}_{name}"
                embeddings.append(vector)
                faiss_map.append(full_name)
    if not embeddings:
        faiss_index = None
        return
    xb = np.stack(embeddings).astype(np.float32)
    faiss_index = faiss.IndexHNSWFlat(DIM, M, faiss.METRIC_INNER_PRODUCT)
    faiss_index.hnsw.efConstruction = EF_CONSTRUCTION
    faiss_index.hnsw.efSearch = EF_SEARCH
    faiss.normalize_L2(xb)
    faiss_index.add(xb)
    # Lưu index và map
    faiss.write_index(faiss_index, FAISS_INDEX_PATH)
    with open(FAISS_MAP_PATH, "wb") as f:
        pickle.dump(faiss_map, f)

def load_faiss_index():
    """Load FAISS index và mapping từ file"""
    global faiss_index, faiss_map
    if os.path.exists(FAISS_INDEX_PATH) and os.path.exists(FAISS_MAP_PATH):
        faiss_index = faiss.read_index(FAISS_INDEX_PATH)
        with open(FAISS_MAP_PATH, "rb") as f:
            faiss_map = pickle.load(f)
    else:
        build_faiss_index()

def add_to_index(name: str, vector: np.ndarray):
    """Thêm vector vào FAISS index và mapping, đồng thời lưu ra file"""
    load_faiss_index()
    if faiss_index is None:
        build_faiss_index()
        load_faiss_index()
    # Chuẩn hóa vector
    vec = vector.astype(np.float32)
    faiss.normalize_L2(vec.reshape(1, -1))
    faiss_index.add(vec.reshape(1, -1))
    faiss_map.append(name)
    # Lưu lại
    faiss.write_index(faiss_index, FAISS_INDEX_PATH)
    with open(FAISS_MAP_PATH, "wb") as f:
        pickle.dump(faiss_map, f)

def search_index(vector: np.ndarray):
    """Tìm vector gần nhất bằng FAISS HNSW"""
    load_faiss_index()
    if faiss_index is None or faiss_index.ntotal == 0:
        return {"name": None, "score": -1}
    vec = vector.astype(np.float32)
    faiss.normalize_L2(vec.reshape(1, -1))
    D, I = faiss_index.search(vec.reshape(1, -1), 1)
    idx = int(I[0][0])
    score = float(D[0][0])
    if idx < 0 or idx >= len(faiss_map):
        return {"name": None, "score": -1}
    return {"name": faiss_map[idx], "score": score}
