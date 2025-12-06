from ultralytics import YOLO
import cv2
import torch

class ObjectDetector:
    def __init__(self, model_path='yolov8n.pt'):
        self.model = YOLO(model_path)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(self.device)
    
    def detect_image(self, image):
        return self.model(image)
    
    def detect_frame(self, frame):
        return self.model(frame)
    
    def draw_boxes(self, frame, result, show_conf=True):
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls_id = int(box.cls[0])
            label = self.model.names[cls_id]

            conf = float(box.conf[0]) * 100
            conf_text = f"{conf:.2f}%" if show_conf else ""

            text = f"{label} {conf_text}"
            cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)