import os
import numpy as np
import config

# Biến toàn cục để giữ embeddings trong RAM
index_embeddings = {}

def add_to_index(name: str, vector: np.ndarray):
    """Thêm vector vào index (RAM), không lưu ra file"""
    index_embeddings[name] = vector

def load_all_embeddings():
    """Load tất cả vector từ thư mục embeddings (đệ quy trong các folder con)"""
    embeddings = {}
    for root, dirs, files in os.walk(config.EMBEDDING_DIR):
        for fname in files:
            if fname.endswith(".npy"):
                name = os.path.splitext(fname)[0]
                vector = np.load(os.path.join(root, fname))
                folder_name = os.path.basename(root)
                full_name = f"{folder_name}_{name}"
                embeddings[full_name] = vector
    return embeddings

def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    vec1_norm = vec1 / (np.linalg.norm(vec1) + 1e-6)
    vec2_norm = vec2 / (np.linalg.norm(vec2) + 1e-6)
    cosine = np.dot(vec1_norm, vec2_norm)
    return np.clip(cosine, -1.0, 1.0)

def search_index(vector: np.ndarray):
    """Tìm vector giống nhất"""
    if not index_embeddings:
        loaded = load_all_embeddings()
        for name, vec in loaded.items():
            index_embeddings[name] = vec

    best_score = -1
    best_name = None

    for name, vec in index_embeddings.items():
        score = cosine_similarity(vector, vec)
        if score > best_score:
            best_score = score
            best_name = name

    return {
        "name": best_name,
        "score": best_score
    }
