# 🌐 Face Recognition API Guide

## Gọi nhận diện
```python
recognizer = FaceRecognizer()
result = recognizer.recognize(image)
```

## Gọi enroll từ nhiều ảnh
```python
result = recognizer.enroll_from_images(images, folder_name=name)
```

## API:
- POST /recognize: gửi 1 ảnh
- POST /enroll: gửi nhiều ảnh + name

Trả về:
```json
{ "success": true, "id": "002", "name": "van", "score": 0.92 }
```
