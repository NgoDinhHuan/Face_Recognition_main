# utils/faiss_index.py

import os
import numpy as np
import config

def add_to_index(name: str, vector: np.ndarray):
    """Lưu vector (128D) vào file npy theo tên"""
    os.makedirs(config.EMBEDDING_DIR, exist_ok=True)
    file_path = os.path.join(config.EMBEDDING_DIR, f"{name}.npy")
    np.save(file_path, vector)

def load_all_embeddings():
    """Load tất cả vector từ thư mục embeddings"""
    embeddings = {}
    for fname in os.listdir(config.EMBEDDING_DIR):
        if fname.endswith(".npy"):
            name = os.path.splitext(fname)[0]
            vector = np.load(os.path.join(config.EMBEDDING_DIR, fname))
            embeddings[name] = vector
    return embeddings

def cosine_similarity(vec1, vec2):
    """Tính cosine similarity giữa 2 vector"""
    dot = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    return dot / (norm1 * norm2 + 1e-6)

def search_index(vector: np.ndarray):
    """Tìm vector giống nhất"""
    embeddings = load_all_embeddings()
    best_score = -1
    best_name = None

    for name, vec in embeddings.items():
        score = cosine_similarity(vector, vec)
        if score > best_score:
            best_score = score
            best_name = name

    return {
        "name": best_name,
        "score": best_score
    }
