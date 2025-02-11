import streamlit as st
import torch
import torchvision.transforms as transforms
from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image
import cv2
import numpy as np
import tempfile
import os
from ultralytics import YOLO  # Import YOLO model

# Load Models
efficientnet_model_path = r"C:\Users\JAYA SOORYA\Downloads\classification_results\defence_efficientnet.pth"
yolo_model_path = r"C:\Users\JAYA SOORYA\Downloads\runs\kaggle\working\runs\detect\train\weights\best.pt"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load EfficientNet Model
model_name = "google/efficientnet-b0"
auto_processor = AutoImageProcessor.from_pretrained(model_name)
efficientnet = AutoModelForImageClassification.from_pretrained(model_name)

# Modify classifier to match trained model (3 output classes)
num_classes = 3
efficientnet.classifier = torch.nn.Linear(in_features=1280, out_features=num_classes)

# Load trained weights
efficientnet.load_state_dict(torch.load(efficientnet_model_path, map_location=device), strict=False)
efficientnet.to(device)
efficientnet.eval()

# Load YOLO Model
yolo = YOLO(yolo_model_path)

# Define class labels
class_labels = {0: "Animal", 1: "Civilian", 2: "Soldier"}

# Predefined valid RFID codes
valid_rfids = {"RFID123456", "SOLDIER98765", "CIVILIAN54321", "FRIENDLY001", "SECURE99999", "TESTRFID2025"}

# Define transformation for images
def transform_image(image):
    inputs = auto_processor(image, return_tensors="pt").to(device)
    return inputs

def classify_image_efficientnet(image):
    """Classifies an image using EfficientNet"""
    inputs = transform_image(image)
    with torch.no_grad():
        outputs = efficientnet(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=1)
        confidence, prediction = torch.max(probs, dim=1)
    
    return class_labels[prediction.item()], confidence.item() * 100  # Convert to percentage

def classify_image_yolo(image):
    """Classifies an image using YOLO"""
    results = yolo(image)
    
    # Extract YOLO detections
    for result in results:
        for box in result.boxes:
            label = int(box.cls)  # Get class index
            conf = float(box.conf) * 100  # Convert to percentage
            if label == 2:  # Soldier detected
                return "Soldier", conf

    return "Unknown", 0  # Default case if YOLO doesn't detect a soldier

def process_video(video_path):
    """Processes video frames and classifies each frame"""
    cap = cv2.VideoCapture(video_path)
    results = []
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % 10 == 0:  # Process every 10th frame
            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            label, conf = classify_image_efficientnet(image)
            if label == "Soldier":
                yolo_label, yolo_conf = classify_image_yolo(image)
                if yolo_label == "Soldier":
                    final_confidence = (conf + yolo_conf) / 2  # Average confidence
                    results.append((label, final_confidence))
                else:
                    results.append((label, conf))
            else:
                results.append((label, conf))

    cap.release()
    return results

# Streamlit UI
st.title("Defence Surveillance System")

option = st.selectbox("Select Input Type", ["Image", "Video"])

video_path = None  # Initialize video_path

if option == "Image":
    uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "png", "jpeg"])
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        label, conf_efficientnet = classify_image_efficientnet(image)
        st.write(f"EfficientNet Prediction: **{label}** (Confidence: {conf_efficientnet:.2f}%)")

        if label == "Animal":
            st.success("✅ Ignored")
        elif label == "Civilian":
            st.error("⚠️ Alert Authorities")
        elif label == "Soldier":
            yolo_label, conf_yolo = classify_image_yolo(image)
            if yolo_label == "Soldier":
                final_confidence = (conf_efficientnet + conf_yolo) / 2
            else:
                final_confidence = conf_efficientnet
            
            st.write(f"Final Confidence Score: **{final_confidence:.2f}%**")

            # Decision-making based on confidence
            if final_confidence >= 80:
                st.warning("⚠️ Informing Authorities - High Threat Detected")
                st.button("⚔️ Attack Non-Lethal Parts")
            elif 60 <= final_confidence < 80:
                st.warning("⚠️ Informing Authorities - Moderate Threat")
                st.button("💉 Use Anesthetic Shots")
            else:
                st.error("🚨 Double Alert! Higher Officials Notified")

            rfid = st.text_input("Enter RFID:")
            if rfid:
                if rfid in valid_rfids:
                    st.success("✅ Soldier Identified as Friendly - Ignore")
                else:
                    st.error("⚠️ Intruder/Enemy - Alert Authorities")

elif option == "Video":
    uploaded_video = st.file_uploader("Upload a Video", type=["mp4", "avi", "mov"])
    if uploaded_video:
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(uploaded_video.read())
            video_path = tmp_file.name

        st.video(video_path)
        st.write("Processing Video...")

        results = process_video(video_path)
        unique_labels = {label for label, _ in results}
        avg_confidence = np.mean([conf for _, conf in results]) if results else 0

        st.write(f"Detected Classes: {unique_labels}")
        st.write(f"Final Confidence Score: **{avg_confidence:.2f}%**")

        if "Animal" in unique_labels:
            st.success("✅ Ignored")
        if "Civilian" in unique_labels:
            st.error("⚠️ Alert Authorities")
        if "Soldier" in unique_labels:
            if avg_confidence >= 80:
                st.warning("⚠️ Informing Authorities - High Threat Detected")
                st.button("⚔️ Attack Non-Lethal Parts")
            elif 60 <= avg_confidence < 80:
                st.warning("⚠️ Informing Authorities - Moderate Threat")
                st.button("💉 Use Anesthetic Shots")
            else:
                st.error("🚨 Double Alert! Higher Officials Notified")

            rfid = st.text_input("Enter RFID:")
            if rfid:
                if rfid in valid_rfids:
                    st.success("✅ Soldier Identified as Friendly - Ignore")
                else:
                    st.error("⚠️ Intruder/Enemy - Alert Authorities")

# Cleanup: Remove the video file if it exists
if video_path and os.path.exists(video_path):
    os.remove(video_path)
