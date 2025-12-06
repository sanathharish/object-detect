import streamlit as st
from PIL import Image
import numpy as np
import cv2
import sys
import os
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.detector import ObjectDetector

 # Initialize the ObjectDetector

detector = ObjectDetector()

st.title("Real Time Object Detection Dashboard")

mode = st.radio("Select Mode", ("Image Upload", "Webcam Feed"))

# File uploader for images

if mode == "Image Upload":

    uploaded = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

    if uploaded:
        # Open the uploaded image as PIL Image
        img = Image.open(uploaded).convert("RGB")
    
        # Convert PIL Image to numpy array
        img_np = np.array(img)

        # Run detection using your detector class
        result = detector.detect_frame(img_np)        
        
        # Draw annotated boxes
        annotated_img = detector.draw_boxes(img_np, result)
    
        # Display results
        st.image(annotated_img, caption="Detections", width=700)

        # Debug output if needed
        st.write("Detection Output:", result.boxes.data.tolist())

elif mode == "Webcam Feed":
    stframe = st.empty()
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        st.error("Error: Could not open webcam.")
    else:
        fps_text = st.empty()
        prev_time = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                st.error("Error: Could not read frame from webcam.")
                break

            result = detector.detect_frame(frame)
            annotated_frame = detector.draw_boxes(frame, result)

            # Calculate FPS
            curr_time = time.time()
            fps = 1 / (curr_time - prev_time) if prev_time != 0 else 0
            prev_time = curr_time
            fps_text.text(f"FPS: {fps:.2f}")

            stframe.image(annotated_frame, channels="RGB", width=st.session_state.get('video_width', 800))

            # Streamlit refresh control
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()