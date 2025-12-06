from ultralytics import YOLO
import cv2

class ObjectDetector:
    def __init__(self, model_path='yolov8n.pt'):
        self.model = YOLO(model_path)
    
    def detect_image(self, image_path):
        results = self.model(image_path)
        return results[0].boxes.data.tolist()
    
    def detect_frame(self, frame):
        results = self.model(frame)
        return results[0]
    
    def draw_boxes(self, frame, result):
        annotated = result.plot()
        annotated = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB) #Bug 1 Fix: #Rendering format fix when uploading image to streamlit app. Causing the flips from warm tones to cool tones.
        return annotated