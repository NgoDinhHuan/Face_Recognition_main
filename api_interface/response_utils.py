import uuid
from datetime import datetime

def build_response(success: bool, matched: bool, person_id: str = "", person_name: str = "", confidence: float = 0.0, message: str = "") -> dict:
    return {
        "request_id": str(uuid.uuid4()),
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "success": success,
        "message": message,
        "result": {
            "matched": matched,
            "person_id": person_id,
            "person_name": person_name,
            "confidence": round(confidence, 3)
        },
        "processing_time_ms": 0
    }
