from align.aligner import align_face
from feature.extractor import extract_feature
from utils.faiss_index import search_index
import config

def recognize_face(image) -> dict:
    aligned = align_face(image)
    if aligned is None:
        return {"success": False, "message": "No face detected"}

    vector = extract_feature(aligned)
    result = search_index(vector)

    if result["score"] >= config.THRESHOLD:
        return {
            "success": True,
            "name": result["name"],
            "score": result["score"]
        }
    else:
        return {
            "success": False,
            "message": "New Person",
            "score": result["score"]
        }
