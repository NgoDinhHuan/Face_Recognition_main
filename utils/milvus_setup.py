from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility
import os

def create_collection():
    """Tạo collection face_embeddings trong Milvus nếu chưa tồn tại"""
    try:
        # Kết nối tới Milvus
        host = os.getenv("MILVUS_HOST", "standalone")
        port = os.getenv("MILVUS_PORT", "19530")
        connections.connect(host=host, port=port)
        
        # Kiểm tra collection đã tồn tại chưa
        if utility.has_collection("face_embeddings"):
            print("Collection face_embeddings đã tồn tại")
            return
            
        # Định nghĩa schema
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=512),
            FieldSchema(name="name", dtype=DataType.VARCHAR, max_length=200)
        ]
        schema = CollectionSchema(fields=fields, description="Face embeddings collection")
        
        # Tạo collection
        collection = Collection(name="face_embeddings", schema=schema)
        
        # Tạo index cho vector field
        index_params = {
            "metric_type": "IP",
            "index_type": "HNSW",
            "params": {"M": 8, "efConstruction": 64}
        }
        collection.create_index(field_name="embedding", index_params=index_params)
        
        print("Đã tạo collection face_embeddings thành công")
        
    except Exception as e:
        print(f"Lỗi khi tạo collection: {e}")
        raise e
    finally:
        connections.disconnect("default")

if __name__ == "__main__":
    create_collection() 