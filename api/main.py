from fastapi import FastAPI, UploadFile, File
from src.detector import ObjectDetector
from PIL import Image
import io

app = FastAPI(title="YOLO Object Detection API")
detector = ObjectDetector()

@app.get("/health")
def health():
    return {"status": "ok"
            
@app.post("/detect/image")}
async def detect_image(file: UploadFile = File(...)):
    img = Image.open(io.BytesIO(await file.read()))
    results = detector.detect_frame(img)
    boxes = results.boxes.data.tolist()
    return {"boxes": boxes}