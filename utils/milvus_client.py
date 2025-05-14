from pymilvus import connections, Collection, utility
import time
import os

# Lấy host từ biến môi trường hoặc dùng giá trị mặc định
MILVUS_HOST = os.getenv("MILVUS_HOST", "standalone")
MILVUS_PORT = os.getenv("MILVUS_PORT", "19530")

def get_connection():
    """Kết nối tới Milvus với retry"""
    max_retries = 3
    retry_delay = 1  # seconds
    
    for i in range(max_retries):
        try:
            connections.connect(host=MILVUS_HOST, port=MILVUS_PORT)
            return True
        except Exception as e:
            if i == max_retries - 1:
                raise Exception(f"Không thể kết nối tới Milvus sau {max_retries} lần thử: {e}")
            print(f"Lỗi kết nối Milvus, thử lại sau {retry_delay}s...")
            time.sleep(retry_delay)
    return False

def insert_embedding(embedding, name, collection_name="face_embeddings"):
    """Insert một vector embedding và tên vào Milvus"""
    try:
        if not get_connection():
            raise Exception("Không thể kết nối tới Milvus")
            
        # Kiểm tra collection tồn tại
        if not utility.has_collection(collection_name):
            raise Exception(f"Collection {collection_name} không tồn tại")
            
        collection = Collection(collection_name)
        # Kiểm tra và tạo index nếu chưa có cho trường embedding
        index_fields = [idx.field_name for idx in collection.indexes]
        if "embedding" not in index_fields:
            index_params = {
                "metric_type": "IP",
                "index_type": "HNSW",
                "params": {"M": 8, "efConstruction": 64}
            }
            collection.create_index(field_name="embedding", index_params=index_params)
            print(f"Đã tạo index cho trường embedding trong collection {collection_name}")
        collection.load()  # Load collection vào memory
        
        # Milvus yêu cầu dữ liệu dạng list các list, mỗi list là 1 field
        data = [
            [embedding],  # embedding: list of 512 float
            [name]        # name: string
        ]
        result = collection.insert(data)
        print(f"Đã insert vector cho {name} vào Milvus. ID: {result.primary_keys}")
        return result.primary_keys
        
    except Exception as e:
        print(f"Lỗi khi insert vào Milvus: {e}")
        raise e
    finally:
        connections.disconnect("default")

def search_embedding(query_embedding, top_k=1, collection_name="face_embeddings"):
    """Tìm kiếm vector gần nhất trong Milvus"""
    try:
        if not get_connection():
            raise Exception("Không thể kết nối tới Milvus")
            
        # Kiểm tra collection tồn tại
        if not utility.has_collection(collection_name):
            raise Exception(f"Collection {collection_name} không tồn tại")
            
        collection = Collection(collection_name)
        collection.load()  # Load collection vào memory
        
        search_params = {
            "metric_type": "IP",
            "params": {"ef": 64}  # Tham số cho HNSW
        }
        
        results = collection.search(
            data=[query_embedding],
            anns_field="embedding",
            param=search_params,
            limit=top_k,
            output_fields=["name"]
        )
        
        if results and len(results[0]) > 0:
            hit = results[0][0]
            return {
                "id": hit.id,
                "name": hit.entity.get("name"),
                "score": hit.score
            }
        return None
        
    except Exception as e:
        print(f"Lỗi khi search trong Milvus: {e}")
        raise e
    finally:
        connections.disconnect("default") 