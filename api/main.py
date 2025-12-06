from fastapi import FastAPI, UploadFile, File
from src.detector import ObjectDetector
from PIL import Image
import io

app = FastAPI()
detector = ObjectDetector()

@app.get("/health")
def health():
    return {"status": "ok"
            
@app.post("/detect/image")}
async def detect_image(file: UploadFile = File(...)):
    img = Image.open(io.BytesIO(await file.read()))
    results = detector.detect_image(img)
    
    response = []
    for box in results[0].boxes:
        response.append({
            "class": detector.model.names[int(box.cls[0])],
            "confidence": round(float(box.conf[0]) * 100, 2),
            "bbox": list(map(float, box.xyxy[0].tolist()))
        })
    return {"detections": response}