import streamlit as st
from PIL import Image
import numpy as np
import cv2
import sys
import os
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.detector import ObjectDetector

st.title("Real Time Object Detection Dashboard")

 # Initialize the ObjectDetector
detector = ObjectDetector()

mode = st.radio("Select Mode", ("Upload Image", "Webcam"))

conf = st.slider("Confidence Threshold", 0.2, 1.0, 0.5)
iou = st.slider("IOU Threshold", 0.2, 1.0, 0.45)

#---------UPLOAD MODE---------
if mode == "Upload Image":

    file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

    if file:
        img = Image.open(file)
        st.image(img, caption="Uploaded Image")
    
        results = detector.model(img, conf=conf, iou=iou)
        annotated = results[0].plot()

        annotated = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        st.image(annotated, caption="Detections", width=700)

#---------WEBCAM MODE---------
elif mode == "Webcam":
    run = st.checkbox("Start Camera")

    cap = cv2.VideoCapture(0)
    curr_time = time.time()

    frame_placeholder = st.empty()

    while run:
        ret, frame = cap.read()
        if not ret:
            st.warning("Camera not detected!")
            break

        results = detector.detect_frame(frame)

        annotated = detector.draw_boxes(frame, results)

        new_time = time.time()
        fps = 1 / (new_time - curr_time)
        curr_time = new_time

        cv2.putText(annotated, f"FPS: {fps:.1f}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        frame_placeholder.image(annotated, channels="RGB")