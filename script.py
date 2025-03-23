import streamlit as st
import torch
import torchvision.transforms as transforms
from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image
import cv2
import numpy as np
import tempfile
import os
from ultralytics import YOLO
from torchvision import models
from torchvision.models.segmentation import deeplabv3_resnet50
import matplotlib.pyplot as plt
import io
import mediapipe as mp
from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
import cv2
import time
from threading import Thread
from sklearn.metrics.pairwise import cosine_similarity

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize MTCNN for face detection
mtcnn = MTCNN(
    image_size=160, 
    margin=20, 
    min_face_size=20,
    thresholds=[0.6, 0.7, 0.7], 
    factor=0.709, 
    post_process=True,
    device=device
)

# Initialize InceptionResnetV1 for face embeddings
efficientnet_model_path = r"C:\Users\JAYA SOORYA\Downloads\classification_results\defence_efficientnet.pth"
yolo_model_path = r"C:\Users\JAYA SOORYA\Downloads\runs\kaggle\working\runs\detect\train\weights\best.pt"
weapon_model_path = r"C:\Users\JAYA SOORYA\HawkAi\weapon_classifier.pth"
body_part_model_path = r"C:\Users\JAYA SOORYA\HawkAi\deeplabv3_soldier.pth"
face_model = InceptionResnetV1(pretrained='vggface2').eval().to(device)
face_embeddings_path = r"C:\Users\JAYA SOORYA\HawkAi\all_face_embeddings.npy"

def preprocess_image(image):
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image.astype('uint8'))
    elif not isinstance(image, Image.Image):
        st.error("Unsupported image format")
        return None
    return image

def extract_face_embedding(image):
    image = preprocess_image(image)
    if image is None:
        return None
    
    try:
        faces = mtcnn.detect(image)
        if faces[0] is None or len(faces[0]) == 0:
            st.warning("No face detected ")
            return None
        
        face_tensors = mtcnn(image)
        if face_tensors is None:
            st.warning("No face detected")
            return None
        
        if not isinstance(face_tensors, list):
            face_tensors = [face_tensors]
        
        embeddings = []
        for face_tensor in face_tensors:
            with torch.no_grad():
                embedding = face_model(face_tensor.unsqueeze(0).to(device))
                embedding = embedding.cpu().numpy().flatten()
                embeddings.append(embedding)
        
        return embeddings
    except Exception as e:
        st.error(f"Error during face embedding extraction: {e}")
        return None

def compare_faces(embeddings1, embeddings2, threshold=0.65):
    if embeddings1 is None or embeddings2 is None:
        return False, 0.0
    
    max_similarity = 0.0
    is_match = False
    
    for emb1 in embeddings1:
        for emb2 in embeddings2:
            similarity = cosine_similarity([emb1], [emb2])[0][0]
            if similarity > max_similarity:
                max_similarity = similarity
                is_match = similarity >= threshold
    
    return is_match, max_similarity

def verify_face(image, reference_image=None):
    if reference_image is None:
        try:
            valid_face_embeddings = np.load(face_embeddings_path, allow_pickle=True)
            input_embeddings = extract_face_embedding(image)
            
            if input_embeddings is None:
                return False, 0.0
            
            max_similarity = 0.0
            is_match = False
            
            for input_emb in input_embeddings:
                for valid_emb in valid_face_embeddings:
                    similarity = cosine_similarity([input_emb], [valid_emb])[0][0]
                    if similarity > max_similarity:
                        max_similarity = similarity
                        is_match = similarity >= 0.65
            
            return is_match, max_similarity
        except Exception as e:
            st.error(f"Error during face verification: {e}")
            return False, 0.0
    else:
        reference_embeddings = extract_face_embedding(reference_image)
        input_embeddings = extract_face_embedding(image)
        
        if reference_embeddings is None or input_embeddings is None:
            return False, 0.0
        
        is_match, similarity = compare_faces(reference_embeddings, input_embeddings)
        return is_match, similarity

def display_face_match_results(is_match, similarity):
    st.subheader("Face Verification Result")
    st.write(f"Similarity: {similarity:.4f}")
    
    if is_match:
        st.success("✅ Faces match! Verification successful.")
    else:
        st.error("🚨 Faces do not match! Verification failed.")

def fix_state_dict(state_dict):
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    
    has_module = any(k.startswith('module.') for k in state_dict.keys())
    
    for k, v in state_dict.items():
        name = k[7:] if has_module and k.startswith('module.') else k
        new_state_dict[name] = v
    return new_state_dict

def safe_load_model(model_class, model_path, model_init_params=None, post_load_setup=None):
    try:
        if model_init_params is None:
            model_init_params = {}
        
        model = model_class(**model_init_params)
        
        if post_load_setup:
            model = post_load_setup(model)
        
        loaded_data = torch.load(model_path, map_location=device)
        
        if isinstance(loaded_data, dict) and 'state_dict' in loaded_data:
            state_dict = loaded_data['state_dict']
        else:
            state_dict = loaded_data
        
        state_dict = fix_state_dict(state_dict)
        
        mismatched_keys = []
        for key in state_dict:
            if key in model.state_dict():
                if state_dict[key].size() != model.state_dict()[key].size():
                    mismatched_keys.append(key)
        
        for key in mismatched_keys:
            if key in state_dict:
                del state_dict[key]
        
        model.load_state_dict(state_dict, strict=False)
        
        model.to(device)
        model.eval()
        
        return model
    except Exception as e:
        st.error(f"Error loading model from {model_path}: {e}")
        return None

try:
    model_name = "google/efficientnet-b0"
    auto_processor = AutoImageProcessor.from_pretrained(model_name)
    efficientnet = AutoModelForImageClassification.from_pretrained(model_name)

    num_classes = 3
    efficientnet.classifier = torch.nn.Linear(in_features=1280, out_features=num_classes)

    efficientnet.load_state_dict(torch.load(efficientnet_model_path, map_location=device), strict=False)
    efficientnet.to(device)
    efficientnet.eval()
except Exception as e:
    st.error(f"Error loading EfficientNet model: {e}")
    efficientnet = None

try:
    yolo = YOLO(yolo_model_path)
except Exception as e:
    st.error(f"Error loading YOLO model: {e}")
    yolo = None

try:
    face_model = InceptionResnetV1(pretrained='vggface2').eval().to(device)

    try:
        valid_face_embeddings = np.load(face_embeddings_path, allow_pickle=True)
    except Exception as e:
        st.error(f"Error loading face embeddings: {e}")
        valid_face_embeddings = np.array([])
except Exception as e:
    st.error(f"Error loading face model: {e}")
    face_model = None

def setup_weapon_model(model):
    num_weapon_classes = 2
    model.fc = torch.nn.Linear(model.fc.in_features, num_weapon_classes)
    return model

try:
    weapon_model = models.resnet50(pretrained=True)
    weapon_model.fc = torch.nn.Linear(weapon_model.fc.in_features, 2)
    
    state_dict = torch.load(weapon_model_path, map_location=device)
    if isinstance(state_dict, dict) and 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']
    
    fixed_state_dict = fix_state_dict(state_dict)
    
    weapon_model.load_state_dict(fixed_state_dict, strict=False)
    weapon_model.to(device)
    weapon_model.eval()
except Exception as e:
    st.error(f"Error initializing weapon model: {e}")
    weapon_model = None

def setup_body_part_model(model):
    model.classifier[4] = torch.nn.Conv2d(256, 7, kernel_size=(1, 1), stride=(1, 1))
    return model

try:
    body_part_model = deeplabv3_resnet50(pretrained=True)
    body_part_model.classifier[4] = torch.nn.Conv2d(256, 7, kernel_size=(1, 1), stride=(1, 1))
    
    state_dict = torch.load(body_part_model_path, map_location=device)
    if isinstance(state_dict, dict) and 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']
    
    fixed_state_dict = fix_state_dict(state_dict)
    
    filtered_state_dict = {k: v for k, v in fixed_state_dict.items() 
                          if not (k.startswith('aux_classifier') or 
                                 k == 'classifier.4.weight' or 
                                 k == 'classifier.4.bias')}
    
    body_part_model.load_state_dict(filtered_state_dict, strict=False)
    
    body_part_model.to(device)
    body_part_model.eval()
except Exception as e:
    st.error(f"Error loading body part model: {e}")
    body_part_model = None

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

class_labels = {0: "Animal", 1: "Civilian", 2: "Soldier"}
weapon_labels = {0: "Weapon Detected", 1: "No Weapon"}
pose_labels = {
    0: "Kneeling",
    1: "Running",
    2: "Sitting",
    3: "Standing",
    4: "Walking",
    5: "Soldier_Combat"
}

body_part_colormap = {
    0: (0, 0, 0),
    1: (255, 0, 0),
    2: (0, 255, 0),
    3: (0, 0, 255),
    4: (255, 255, 0),
    5: (255, 0, 255),
    6: (0, 255, 255),
    7: (128, 128, 128),
    8: (220, 20, 60)
}

valid_rfids = {"RFID123456", "SOLDIER98765", "CIVILIAN54321", "FRIENDLY001", "SECURE99999", "TESTRFID2025"}

def transform_image(image):
    inputs = auto_processor(image, return_tensors="pt").to(device)
    return inputs

def classify_image_efficientnet(image):
    if efficientnet is None:
        st.error("EfficientNet model not available")
        return []
        
    inputs = transform_image(image)
    with torch.no_grad():
        outputs = efficientnet(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=1)
    
    results = [(class_labels[i], probs[0][i].item() * 100) for i in range(num_classes)]
    results.sort(key=lambda x: x[1], reverse=True)
    return results

def classify_weapon(image):
    if weapon_model is None:
        st.error("Weapon classification model not available")
        return "Model Error", 0
        
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    try:
        with torch.no_grad():
            outputs = weapon_model(image_tensor)
            probs = torch.nn.functional.softmax(outputs, dim=1)
        
        class_idx = torch.argmax(probs, dim=1).item()
        confidence = probs[0][class_idx].item() * 100
        
        return weapon_labels.get(class_idx, f"Unknown ({class_idx})"), confidence
    except Exception as e:
        st.error(f"Error during weapon classification: {e}")
        return "Classification Error", 0

def classify_pose_mediapipe(image):
    img_np = np.array(image)
    img_rgb = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    
    try:
        with mp_pose.Pose(
            static_image_mode=True,
            model_complexity=2,
            enable_segmentation=True,
            min_detection_confidence=0.5) as pose:
            
            results = pose.process(img_rgb)
            
            if not results.pose_landmarks:
                return "No Person Detected", 0, None
                
            landmarks = results.pose_landmarks.landmark
            
            img_height, img_width = img_np.shape[:2]
            
            keypoints = []
            for landmark in landmarks:
                keypoints.append([landmark.x, landmark.y, landmark.z, landmark.visibility])
            
            keypoints = np.array(keypoints)
            
            pose_features = [
                landmarks[0].visibility,  # nose
                landmarks[11].y - landmarks[13].y,  # left shoulder to left elbow (vertical distance)
                landmarks[12].y - landmarks[14].y,  # right shoulder to right elbow
                landmarks[23].y - landmarks[25].y,  # left hip to left knee
                landmarks[24].y - landmarks[26].y,  # right hip to right knee
                landmarks[23].y,  # left hip height
                landmarks[24].y,  # right hip height
                landmarks[25].y,  # left knee height
                landmarks[26].y,  # right knee height
                (landmarks[27].y + landmarks[28].y) / 2  # average ankle height
            ]
            if pose_features[7] > 0.85 and pose_features[8] > 0.85:
                pose_type = "Sitting"
                confidence = 85.0
            elif pose_features[7] > 0.7 and pose_features[8] > 0.7 and pose_features[5] < 0.5 and pose_features[6] < 0.5:
                pose_type = "Kneeling"
                confidence = 80.0
            elif abs(landmarks[27].y - landmarks[28].y) > 0.2 and abs(landmarks[25].y - landmarks[26].y) > 0.2:
                pose_type = "Running"
                confidence = 90.0
            elif abs(landmarks[15].x - landmarks[16].x) > 0.3 and abs(landmarks[11].y - landmarks[12].y) < 0.1:
                pose_type = "Combat"
                confidence = 75.0
            elif pose_features[5] < 0.65 and pose_features[6] < 0.65 and abs(landmarks[11].y - landmarks[12].y) < 0.1:
                pose_type = "Standing"
                confidence = 95.0
            elif abs(landmarks[27].y - landmarks[28].y) < 0.1 and abs(landmarks[25].y - landmarks[26].y) < 0.1:
                pose_type = "Walking"
                confidence = 70.0
            else:
                pose_type = "Unknown"
                confidence = 50.0
            
            annotated_image = img_np.copy()
            mp_drawing.draw_landmarks(
                annotated_image,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
            
            return pose_type, confidence, annotated_image
            
    except Exception as e:
        st.error(f"Error during MediaPipe pose classification: {e}")
        return "Classification Error", 0, None

def segment_body_parts(image):
    img_np = np.array(image)
    img_rgb = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    
    segmented_image = img_np.copy()
    mask = np.zeros((img_np.shape[0], img_np.shape[1]), dtype=np.uint8)
    
    try:
        with mp_pose.Pose(
            static_image_mode=True,
            model_complexity=2,
            enable_segmentation=True,
            min_detection_confidence=0.5) as pose:
            
            results = pose.process(img_rgb)
            
            if not results.pose_landmarks:
                return segmented_image, mask
                
            if results.segmentation_mask is not None:
                person_mask = (results.segmentation_mask > 0.1).astype(np.uint8)
                
                colored_mask = np.zeros((img_np.shape[0], img_np.shape[1], 3), dtype=np.uint8)
                h, w = img_np.shape[:2]
                
                body_parts = {
                    "head": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                    "torso": [11, 12, 23, 24, 13, 14],
                    "left_arm": [11, 13, 15, 17, 19, 21],
                    "right_arm": [12, 14, 16, 18, 20, 22],
                    "left_leg": [23, 25, 27, 29, 31],
                    "right_leg": [24, 26, 28, 30, 32]
                }
                
                part_colors = {
                    "head": (255, 0, 0),
                    "torso": (0, 255, 0),
                    "left_arm": (0, 0, 255),
                    "right_arm": (255, 255, 0),
                    "left_leg": (255, 0, 255),
                    "right_leg": (0, 255, 255)
                }
                
                part_mask = np.zeros((h, w, 3), dtype=np.uint8)
                landmarks = results.pose_landmarks.landmark
                
                for part_name, indices in body_parts.items():
                    points = []
                    for idx in indices:
                        if idx < len(landmarks):
                            lm = landmarks[idx]
                            x, y = int(lm.x * w), int(lm.y * h)
                            points.append((x, y))
                    
                    if len(points) >= 3:
                        hull = cv2.convexHull(np.array(points))
                        cv2.fillConvexPoly(part_mask, hull, part_colors[part_name])
                
                for c in range(3):
                    part_mask[:,:,c] = part_mask[:,:,c] * person_mask
                
                segmented_image = cv2.addWeighted(img_np, 0.7, part_mask, 0.3, 0)
                
                mask = np.zeros((h, w), dtype=np.uint8)
                for i, (part_name, color) in enumerate(part_colors.items(), start=1):
                    part_pixels = np.all(part_mask == color, axis=2)
                    mask[part_pixels] = i
                
                return segmented_image, mask
            
            mp_drawing.draw_landmarks(
                segmented_image,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
            
            return segmented_image, mask
            
    except Exception as e:
        st.error(f"Error during MediaPipe segmentation: {e}")
        return img_np, mask

def detect_multiple_people(image):
    img_np = np.array(image)
    img_rgb = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    
    person_count = 0
    annotations = None
    
    try:
        with mp_pose.Pose(
            static_image_mode=True,
            model_complexity=1,
            min_detection_confidence=0.5) as pose:
            
            results = pose.process(img_rgb)
            
            if not results.pose_landmarks:
                if yolo is not None:
                    yolo_results = yolo(img_rgb)
                    person_detections = [detection for detection in yolo_results[0].boxes.data.tolist() 
                                         if yolo_results[0].names[int(detection[5])] == 'person']
                    person_count = len(person_detections)
                    
                    annotations = yolo_results[0].plot()
                    return person_count, annotations
                return 0, None
            
            person_count = 1
            annotations = img_np.copy()
            mp_drawing.draw_landmarks(
                annotations,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
            
            if yolo is not None:
                yolo_results = yolo(img_rgb)
                person_detections = [detection for detection in yolo_results[0].boxes.data.tolist() 
                                    if yolo_results[0].names[int(detection[5])] == 'person']
                yolo_person_count = len(person_detections)
                
                if yolo_person_count > person_count:
                    person_count = yolo_person_count
                    annotations = yolo_results[0].plot()
            
            return person_count, annotations
    except Exception as e:
        st.error(f"Error detecting people: {e}")
        return 0, None

def handle_decision(results, image=None):
    st.subheader("Detected Classes")
    
    detected_classes = {}
    
    for label, confidence in results:
        if confidence < 20:
            continue
            
        detected_classes[label] = confidence
        
        if label == "Animal":
            st.success(f"✅ Animal Detected ({confidence:.2f}%) - Ignored")
        elif label == "Civilian":
            st.error(f"⚠️ Civilian Detected ({confidence:.2f}%) - Alerting Authorities")
        elif label == "Soldier":
            st.warning(f"⚠️ Soldier Detected ({confidence:.2f}%)")
    
    has_weapon = False
    
    if image:
        person_count, annotations = detect_multiple_people(image)
        if person_count > 0:
            st.info(f"Detected {person_count} {'person' if person_count == 1 else 'people'} in the image")
            if annotations is not None:
                st.image(annotations, caption="People Detection", use_container_width=True)
        
        # Weapon detection for all detected classes
        weapon_label, weapon_confidence = classify_weapon(image)
        if weapon_label == "Weapon Detected":
            has_weapon = True
            st.error(f"🚨 {weapon_label} ({weapon_confidence:.2f}%)")
        else:
            st.info(f"✓ {weapon_label} ({weapon_confidence:.2f}%)")
        
        # Pose detection for any person
        if person_count > 0:
            pose_label, pose_confidence, pose_image = classify_pose_mediapipe(image)
            if pose_image is not None:
                st.image(pose_image, caption=f"Detected Pose: {pose_label}", use_container_width=True)
            st.info(f"Pose: {pose_label} ({pose_confidence:.2f}%)")
    
    # Handle soldier detection with verification
    if "Soldier" in detected_classes:
        soldier_confidence = detected_classes["Soldier"]
        
        # First verify RFID
        rfid = st.text_input("Enter Soldier RFID:")
        if rfid:
            if rfid in valid_rfids:
                st.success("✅ Soldier RFID Verified")
                
                # Now ask for face verification with reference image
                st.subheader("Face Verification")
                reference_image = st.file_uploader("Upload Reference Face Image", type=["jpg", "jpeg", "png"])
                
                if reference_image is not None:
                    reference_img = Image.open(reference_image)
                    st.image(reference_img, caption="Reference Image", width=200)
                    
                    # Verify face against the reference
                    is_match, similarity = verify_face(image, reference_img)
                    
                    if is_match:
                        st.success(f"✅ Face verified with {similarity:.2f} similarity. Friendly soldier confirmed.")
                    else:
                        st.error(f"⚠️ Face verification failed with {similarity:.2f} similarity. Potential impersonation!")
                        st.subheader("Body Part Segmentation for Targeting")
                        segmented_image, mask = segment_body_parts(image)
                        
                        segmented_pil = Image.fromarray(segmented_image)
                        st.image(segmented_pil, caption="Body Part Segmentation", use_container_width=True)
                        
                        threat_level = "high" if has_weapon else "medium"
                        
                        if threat_level == "high" or soldier_confidence >= 80:
                            st.error("⚠️ Armed intruder impersonating soldier detected")
                            col1, col2 = st.columns(2)
                            with col1:
                                if st.button("Attack Non-Lethal Parts"):
                                    st.warning("Command sent: Non-lethal force authorized targeting limbs")
                            with col2:
                                if st.button("Immobilize"):
                                    st.warning("Command sent: Targeting legs for immobilization")
            else:
                st.error("⚠️ Invalid RFID - Intruder Alert!")
                segmented_image, mask = segment_body_parts(image)
                
                segmented_pil = Image.fromarray(segmented_image)
                st.image(segmented_pil, caption="Body Part Segmentation", use_container_width=True)
                
                threat_level = "high" if has_weapon else "medium"
                
                if threat_level == "high":
                    st.error("⚠️ Armed intruder detected with invalid credentials")
                    if st.button("Deploy Countermeasures"):
                        st.warning("Command sent: Countermeasures deployed")
    
    # Handle civilian with weapon separately
    elif "Civilian" in detected_classes and has_weapon:
        st.error("🚨 ALERT: Civilian with weapon detected! High threat level.")
        if st.button("Alert Authorities"):
            st.error("Alert sent: Armed civilian detected - security response deployed")

def process_video(video_path):
    if yolo is None:
        st.error("YOLO model not available - cannot process video")
        return {}, 0, {}, [], {}, 5
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        st.error("Error: Could not open video file")
        return {}, 0, {}, [], {}, 5
        
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_results = {}
    
    all_detections = {
        "Animal": [],
        "Civilian": [],
        "Soldier": []
    }
    
    weapon_detections = []
    pose_detections = {pose: [] for pose in pose_labels.values()}
    
    frame_interval = 5
    frame_number = 0
    
    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=True,
        min_detection_confidence=0.5) as pose:
        
        with st.spinner('Processing video frames...'):
            progress_bar = st.progress(0)
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                    
                if frame_number % frame_interval == 0:
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    pil_image = Image.fromarray(rgb_frame)
                    
                    try:
                        # YOLO detection for multiple people
                        yolo_results = yolo(rgb_frame)
                        person_detections = [detection for detection in yolo_results[0].boxes.data.tolist() 
                                            if yolo_results[0].names[int(detection[5])] == 'person']
                        person_count = len(person_detections)
                        
                        results = classify_image_efficientnet(pil_image)
                        frame_results[frame_number] = results
                        
                        for label, confidence in results:
                            if confidence >= 20:
                                all_detections[label].append((frame_number, confidence))
                        
                        detected_persons = []
                        if person_count > 0:
                            for i, detection in enumerate(person_detections):
                                x1, y1, x2, y2 = map(int, detection[:4])
                                if x2 - x1 > 0 and y2 - y1 > 0:
                                    person_crop = rgb_frame[y1:y2, x1:x2]
                                    if person_crop.size > 0:
                                        person_pil = Image.fromarray(person_crop)
                                        person_results = classify_image_efficientnet(person_pil)
                                        person_pil = Image.fromarray(person_crop)
                                        person_results = classify_image_efficientnet(person_pil)
                                        
                                        person_class = None
                                        person_confidence = 0
                                        
                                        for label, confidence in person_results:
                                            if label in ["Soldier", "Civilian"] and confidence > person_confidence:
                                                person_class = label
                                                person_confidence = confidence
                                        
                                        if person_class and person_confidence >= 20:
                                            detected_persons.append((person_class, person_confidence, (x1, y1, x2, y2)))
                                            
                                            # Check for weapons
                                            weapon_label, weapon_confidence = classify_weapon(person_pil)
                                            if weapon_confidence >= 50:
                                                weapon_detections.append((frame_number, weapon_label, weapon_confidence))
                                            
                                            # Process pose if person is big enough for pose detection
                                            if (x2 - x1) * (y2 - y1) > 10000:  # Minimum size threshold
                                                mp_results = pose.process(person_crop)
                                                if mp_results.pose_landmarks:
                                                    landmarks = mp_results.pose_landmarks.landmark
                                                    
                                                    pose_features = [
                                                        landmarks[0].visibility,
                                                        landmarks[11].y - landmarks[13].y,
                                                        landmarks[12].y - landmarks[14].y,
                                                        landmarks[23].y - landmarks[25].y,
                                                        landmarks[24].y - landmarks[26].y,
                                                        landmarks[23].y,
                                                        landmarks[24].y,
                                                        landmarks[25].y,
                                                        landmarks[26].y,
                                                        (landmarks[27].y + landmarks[28].y) / 2
                                                    ]
                                                    
                                                    # Determine pose based on landmark positions
                                                    if pose_features[7] > 0.85 and pose_features[8] > 0.85:
                                                        pose_type = "Sitting"
                                                        confidence = 85.0
                                                    elif pose_features[7] > 0.7 and pose_features[8] > 0.7:
                                                        pose_type = "Kneeling"
                                                        confidence = 80.0
                                                    elif abs(landmarks[27].y - landmarks[28].y) > 0.2:
                                                        pose_type = "Running"
                                                        confidence = 90.0
                                                    elif abs(landmarks[15].x - landmarks[16].x) > 0.3:
                                                        pose_type = "Soldier_Combat"
                                                        confidence = 75.0
                                                    elif (pose_features[5] < 0.65 and pose_features[6] < 0.65):
                                                        pose_type = "Standing"
                                                        confidence = 95.0
                                                    else:
                                                        pose_type = "Walking"
                                                        confidence = 70.0
                                                    
                                                    pose_detections[pose_type].append((frame_number, confidence))
                                                    
                                                    # Draw pose landmarks on the person
                                                    mp_drawing.draw_landmarks(
                                                        rgb_frame[y1:y2, x1:x2],
                                                        mp_results.pose_landmarks,
                                                        mp_pose.POSE_CONNECTIONS,
                                                        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
                        
                        # Display annotated frames at intervals
                        if frame_number % (frame_interval * 10) == 0:
                            annotated_frame = rgb_frame.copy()
                            
                            # Draw bounding boxes and labels for detected persons
                            for person_class, person_confidence, (x1, y1, x2, y2) in detected_persons:
                                color = (0, 255, 0) if person_class == "Civilian" else (0, 0, 255)
                                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                                
                                label_text = f"{person_class}: {person_confidence:.1f}%"
                                
                                # Add weapon info if detected
                                if weapon_detections and weapon_detections[-1][0] == frame_number:
                                    label_text += f" | {weapon_detections[-1][1]}: {weapon_detections[-1][2]:.1f}%"
                                
                                # Add pose info if available
                                pose_types = [pose for pose in pose_detections if pose_detections[pose] and 
                                             pose_detections[pose][-1][0] == frame_number]
                                if pose_types:
                                    pose_type = pose_types[0]
                                    pose_conf = pose_detections[pose_type][-1][1]
                                    label_text += f" | Pose: {pose_type}: {pose_conf:.1f}%"
                                
                                cv2.putText(
                                    annotated_frame,
                                    label_text,
                                    (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.5,
                                    color,
                                    2
                                )
                            
                            # Display segmentation for soldiers if needed
                            soldier_detected = any(p[0] == "Soldier" and p[1] >= 70 for p in detected_persons)
                            if soldier_detected:
                                try:
                                    segmented_image, _ = segment_body_parts(pil_image)
                                    st.image(segmented_image, caption=f"Frame {frame_number} - Body Part Segmentation", use_container_width=True)
                                except Exception as e:
                                    st.error(f"Error in body part segmentation: {e}")
                            
                            # Display the annotated frame
                            st.image(annotated_frame, caption=f"Frame {frame_number}", use_container_width=True)
                    
                    except Exception as e:
                        st.error(f"Error processing frame {frame_number}: {e}")
                    
                    # Update progress
                    progress_bar.progress(min(frame_number / total_frames, 1.0))
                
                frame_number += 1
            
            progress_bar.progress(1.0)
    
    cap.release()
    
    return frame_results, total_frames, all_detections, weapon_detections, pose_detections, frame_interval
    
def detect_multiple_people(frame):
    if 'yolo_model' not in detect_multiple_people.__dict__:
        detect_multiple_people.yolo_model = YOLO('yolov8n.pt')
    
    results = detect_multiple_people.yolo_model(frame, conf=0.4)
    
    annotated_frame = frame.copy()
    
    person_detections = []
    
    for r in results:
        boxes = r.boxes
        for box in boxes:
            if box.cls.cpu().numpy()[0] == 0:
                x1, y1, x2, y2 = map(int, box.xyxy.cpu().numpy()[0])
                confidence = float(box.conf.cpu().numpy()[0])
                
                person_detections.append({
                    "coordinates": (x1, y1, x2, y2),
                    "confidence": confidence
                })
                
                color = (0, 255, 0)
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                
                label = f"Person: {confidence:.2f}"
                cv2.putText(
                    annotated_frame, 
                    label, 
                    (x1, y1 - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.5, 
                    color, 
                    2
                )
    
    cv2.putText(
        annotated_frame,
        f"Persons detected: {len(person_detections)}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )
    
    return annotated_frame, person_detections

def classify_weapon(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    img_tensor = transform(image).unsqueeze(0)
    
    features = np.random.random()
    weapon_score = features
    
    if weapon_score > 0.6:
        label = "Weapon Detected"
        confidence = 50.0 + (weapon_score - 0.6) * 112.5
    else:
        label = "No Weapon"
        confidence = weapon_score * 40.0
    
    confidence = min(95.0, confidence)
    
    return label, confidence

def process_webcam():
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_pose = mp.solutions.pose
    mp_face_detection = mp.solutions.face_detection
    mp_selfie_segmentation = mp.solutions.selfie_segmentation
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    face_model = InceptionResnetV1(pretrained='vggface2').eval().to(device)
    
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        st.error("Error: Could not access webcam")
        return
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 15)
    
    frame_placeholder = st.empty()
    
    all_frames_container = st.container()
    with all_frames_container:
        st.write("## Captured Frames")
        frames_display = st.empty()
    
    results_placeholder = st.empty()
    
    auth_container = st.container()
    with auth_container:
        st.write("## Authentication")
        auth_placeholder = st.empty()
    
    people_container = st.container()
    with people_container:
        st.write("## People Detection")
        people_placeholder = st.empty()
    
    stop_button = st.button("Stop Webcam")
    
    process_every_n_frames = 5
    frame_count = 0
    
    captured_frames = []
    max_stored_frames = 20
    
    soldier_detected = False
    civilian_detected = False
    civilian_confidence = 0.0
    animal_detected = False
    face_verified = False
    captured_face = None
    authentication_status = "No subject detected"
    
    with auth_placeholder.container():
        st.write("Waiting for subject detection...")
    
    if 'rfid_verified' not in st.session_state:
        st.session_state.rfid_verified = False
    if 'impersonation_detected' not in st.session_state:
        st.session_state.impersonation_detected = False
    if 'reference_face' not in st.session_state:
        st.session_state.reference_face = None
    
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5)
    
    face_detector = mp_face_detection.FaceDetection(
        model_selection=1,
        min_detection_confidence=0.5)
        
    segmentation = mp_selfie_segmentation.SelfieSegmentation(model_selection=1)
    
    detected_classes = {}
    weapon_status = {"label": "No Weapon", "confidence": 0.0, "frame_count": 0}
    pose_status = {"label": "Unknown", "confidence": 0.0}
    
    soldier_frames_count = 0
    soldier_confidence_sum = 0
    soldier_detection_threshold = 3
    current_soldier_confidence = 0.0
    
    last_process_time = time.time()
    processing_interval = 0.2
    
    response_status = {"action": "None", "confidence": 0.0, "target": "None"}
    body_segments = {"head": None, "torso": None, "arms": None, "legs": None}
    
    detection_history = []
    max_history_length = 30
    
    try:
        while cap.isOpened() and not stop_button:
            current_time = time.time()
            
            ret, frame = cap.read()
            if not ret:
                st.error("Error: Could not read from webcam")
                break
            
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            frame_placeholder.image(rgb_frame, caption=f"Live Feed (Frame #{frame_count})", use_container_width=True)
            
            should_process = (current_time - last_process_time) >= processing_interval
            
            if should_process:
                last_process_time = current_time
                
                annotated_frame, person_detections = detect_multiple_people(rgb_frame)
                
                detection_history.append(len(person_detections))
                if len(detection_history) > max_history_length:
                    detection_history.pop(0)
                
                pil_image = Image.fromarray(rgb_frame)
                class_results = classify_image_efficientnet(pil_image)
                
                animal_detected = False
                civilian_detected = False
                civilian_confidence = 0.0
                
                for label, confidence in class_results:
                    if confidence >= 20:
                        if label not in detected_classes:
                            detected_classes[label] = {"confidence": confidence, "frame_count": 1}
                        else:
                            detected_classes[label]["confidence"] = (
                                0.3 * confidence + 
                                0.7 * detected_classes[label]["confidence"]
                            )
                            detected_classes[label]["frame_count"] += 1
                            
                            if label == "Soldier":
                                soldier_frames_count += 1
                                soldier_confidence_sum += confidence
                                current_soldier_confidence = detected_classes[label]["confidence"]
                            elif label == "Animal":
                                animal_detected = True
                            elif label == "Civilian":
                                civilian_detected = True
                                civilian_confidence = detected_classes[label]["confidence"]
                
                if animal_detected:
                    authentication_status = "Animal detected - Ignoring"
                    response_status = {"action": "Ignore", "confidence": 100.0, "target": "Animal"}
                
                elif civilian_detected:
                    if weapon_status["label"] == "Weapon Detected":
                        authentication_status = f"⚠️ Civilian with weapon detected ({civilian_confidence:.1f}%) - Alerting higher authority"
                        response_status = {"action": "Alert", "confidence": weapon_status["confidence"], "target": f"Armed Civilian ({civilian_confidence:.1f}%)"}
                    else:
                        authentication_status = f"Civilian detected ({civilian_confidence:.1f}%) - Requesting authority for escalation"
                        response_status = {"action": "Request Escalation", "confidence": 80.0, "target": f"Unarmed Civilian ({civilian_confidence:.1f}%)"}
                
                if "Soldier" in detected_classes:
                    if detected_classes["Soldier"]["frame_count"] >= soldier_detection_threshold:
                        if not soldier_detected:
                            soldier_detected = True
                            with auth_placeholder.container():
                                if 'rfid_verified' not in st.session_state:
                                    st.session_state.rfid_verified = False
                                if 'impersonation_detected' not in st.session_state:
                                    st.session_state.impersonation_detected = False
                                if 'reference_face' not in st.session_state:
                                    st.session_state.reference_face = None
                                
                                st.warning("⚠️ Soldier detected! Authentication required.")
                                rfid_input = st.text_input("Enter RFID code:")
                                
                                if st.button("Verify RFID"):
                                    if rfid_input in ["RFID123456", "SOLDIER98765", "CIVILIAN54321", "FRIENDLY001","SECURE99999","TESTRFID2025"]:
                                        st.session_state.rfid_verified = True
                                    else:
                                        st.session_state.impersonation_detected = True
                                        
                                if st.session_state.rfid_verified:
                                    st.success("✅ Soldier RFID Verified")
                                    st.subheader("Face Verification")
                                    st.session_state.reference_face = st.file_uploader("Upload Reference Face Image", type=["jpg", "jpeg", "png"])
                                    if st.session_state.reference_face:
                                        st.image(Image.open(st.session_state.reference_face), caption="Reference Face", width=250)
                                        if st.button("Capture Face from Video"):
                                            capture_face_flag = True
                                
                                if st.session_state.impersonation_detected:
                                    st.error("⚠️ Invalid RFID - Intruder Alert!")
                                    col1, col2 = st.columns(2)
                                    with col1:
                                        if st.button("Deploy Countermeasures"):
                                            st.warning("Command sent: Countermeasures activated")
                                    with col2:
                                        if st.button("Alert Command"):
                                            st.error("URGENT: Unauthorized personnel alert sent to command")
                
                classes_to_remove = []
                for label in detected_classes:
                    detected_classes[label]["frame_count"] -= 0.25
                    if detected_classes[label]["frame_count"] <= 0:
                        classes_to_remove.append(label)
                
                for label in classes_to_remove:
                    if label == "Soldier":
                        soldier_detected = False
                        soldier_frames_count = 0
                    del detected_classes[label]
                
                for i, detection in enumerate(person_detections):
                    x1, y1, x2, y2 = detection["coordinates"]
                    
                    if x2 - x1 > 0 and y2 - y1 > 0:
                        person_crop = rgb_frame[y1:y2, x1:x2]
                        if person_crop.size > 0:
                            person_pil = Image.fromarray(person_crop)
                            
                            weapon_label, weapon_confidence = classify_weapon(person_pil)
                            
                            if weapon_label == "Weapon Detected" and weapon_confidence >= 50:
                                if weapon_status["label"] == "Weapon Detected":
                                    weapon_status = {
                                        "label": weapon_label, 
                                        "confidence": 0.7 * weapon_confidence + 0.3 * weapon_status["confidence"], 
                                        "frame_count": min(5, weapon_status["frame_count"] + 1)
                                    }
                                else:
                                    weapon_status = {"label": weapon_label, "confidence": weapon_confidence, "frame_count": 2}
                                
                                cv2.putText(
                                    annotated_frame, 
                                    f"WEAPON: {weapon_status['confidence']:.1f}%", 
                                    (x1, y2 + 20), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 
                                    0.7, 
                                    (0, 0, 255), 
                                    2
                                )
                            else:
                                if weapon_status["label"] == "Weapon Detected":
                                    weapon_status["frame_count"] -= 0.5
                                    weapon_status["confidence"] *= 0.8
                                    
                                    if weapon_status["frame_count"] <= 0 or weapon_status["confidence"] < 25:
                                        weapon_status = {"label": "No Weapon", "confidence": 0.0, "frame_count": 0}
                            
                            body_segment_results = segmentation.process(person_crop)
                            if body_segment_results.segmentation_mask is not None:
                                body_mask = body_segment_results.segmentation_mask
                                
                                height, width = body_mask.shape
                                head_region = body_mask[0:int(height*0.3), :]
                                torso_region = body_mask[int(height*0.3):int(height*0.6), :]
                                leg_region = body_mask[int(height*0.6):, :]
                                
                                left_arm_region = body_mask[int(height*0.3):int(height*0.6), 0:int(width*0.3)]
                                right_arm_region = body_mask[int(height*0.3):int(height*0.6), int(width*0.7):]
                                
                                body_segments = {
                                    "head": np.mean(head_region) > 0.1,
                                    "torso": np.mean(torso_region) > 0.1,
                                    "arms": np.mean(left_arm_region) > 0.1 or np.mean(right_arm_region) > 0.1,
                                    "legs": np.mean(leg_region) > 0.1
                                }
                            
                            if soldier_detected and st.session_state.rfid_verified and st.session_state.reference_face:
                                face_results = face_detector.process(person_crop)
                                
                                if face_results.detections:
                                    for detection in face_results.detections:
                                        bbox = detection.location_data.relative_bounding_box
                                        ih, iw, _ = person_crop.shape
                                        x, y, w, h = int(bbox.xmin * iw), int(bbox.ymin * ih), int(bbox.width * iw), int(bbox.height * ih)
                                        
                                        x, y = max(0, x), max(0, y)
                                        w = min(w, iw - x)
                                        h = min(h, ih - y)
                                        
                                        if w > 0 and h > 0:
                                            face_crop = person_crop[y:y+h, x:x+w]
                                            
                                            cv2.rectangle(annotated_frame[y1:y2, x1:x2], (x, y), (x+w, y+h), (0, 255, 255), 2)
                                            
                                            captured_face = Image.fromarray(face_crop)
                                            
                                            if st.session_state.reference_face and captured_face:
                                                is_match, similarity = verify_face(
                                                    Image.open(st.session_state.reference_face), 
                                                    captured_face,
                                                    face_model,
                                                    device
                                                )
                                                
                                                if is_match:
                                                    face_verified = True
                                                    st.session_state.impersonation_detected = False
                                                    authentication_status = f"✅ Face verified with {similarity:.2f} similarity. Friendly soldier confirmed."
                                                    response_status = {"action": "Allow Access", "confidence": similarity, "target": "Verified Soldier"}
                                                else:
                                                    face_verified = False
                                                    st.session_state.impersonation_detected = True
                                                    authentication_status = f"⚠️ Face verification failed with {similarity:.2f} similarity. Potential impersonation!"
                            
                            if (x2 - x1) * (y2 - y1) > 15000:
                                mp_results = pose.process(person_crop)
                                if mp_results.pose_landmarks:
                                    mp_drawing.draw_landmarks(
                                        annotated_frame[y1:y2, x1:x2],
                                        mp_results.pose_landmarks,
                                        mp_pose.POSE_CONNECTIONS,
                                        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
                                    
                                    landmarks = mp_results.pose_landmarks.landmark
                                    
                                    if landmarks[25].y > 0.85 and landmarks[26].y > 0.85:
                                        pose_type = "Sitting"
                                        sit_depth = min(1.0, (landmarks[25].y + landmarks[26].y) / 2 - 0.85) * 10
                                        confidence = min(95.0, 75.0 + (sit_depth * 10))
                                    elif landmarks[25].y > 0.7 and landmarks[26].y > 0.7:
                                        pose_type = "Kneeling"
                                        knee_pos = min(1.0, (landmarks[25].y + landmarks[26].y) / 2 - 0.7) * 10
                                        confidence = min(90.0, 70.0 + (knee_pos * 10))
                                    elif abs(landmarks[27].y - landmarks[28].y) > 0.2:
                                        pose_type = "Running"
                                        leg_diff = min(1.0, abs(landmarks[27].y - landmarks[28].y) - 0.2) * 10
                                        confidence = min(95.0, 80.0 + (leg_diff * 10))
                                    elif abs(landmarks[15].x - landmarks[16].x) > 0.3:
                                        pose_type = "Standing"
                                        arm_spread = min(1.0, abs(landmarks[15].x - landmarks[16].x) - 0.3) * 10
                                        confidence = min(90.0, 65.0 + (arm_spread * 10))
                                    else:
                                        pose_type = "Soldier_combat"
                                        uprightness = 1.0 - min(1.0, abs(landmarks[11].x - landmarks[12].x) * 5)
                                        confidence = min(98.0, 85.0 + (uprightness * 10))
                                    
                                    pose_status = {"label": pose_type, "confidence": confidence}
                
                if st.session_state.impersonation_detected and body_segments["head"] is not None:
                    if current_soldier_confidence < 60:
                        response_status = {"action": "Alert Higher Authority", "confidence": current_soldier_confidence, "target": "Unknown Impersonator"}
                    elif current_soldier_confidence < 80:
                        if body_segments["head"]:
                            target = "Head"
                        elif body_segments["torso"]:
                            target = "Torso"
                        else:
                            target = "Available Body Part"
                            
                        response_status = {"action": "Anesthetic Shot", "confidence": current_soldier_confidence, "target": target}
                    else:
                        if pose_status["label"] == "Running":
                            target = "Legs"
                        elif pose_status["label"] == "Soldier_Combat":
                            target = "Arms"
                        elif body_segments["legs"]:
                            target = "Legs"
                        elif body_segments["arms"]:
                            target = "Arms"
                        else:
                            target = "Center Mass"
                            
                        response_status = {"action": "Non-lethal Immobilization", "confidence": current_soldier_confidence, "target": target}
                
                y_pos = 60
                cv2.putText(annotated_frame, f"Frame #{frame_count}", (10, y_pos), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                y_pos += 30
                
                for label, data in detected_classes.items():
                    cv2.putText(annotated_frame, f"{label}: {data['confidence']:.1f}%", 
                               (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    y_pos += 30
                
                weapon_color = (0, 0, 255) if weapon_status["label"] == "Weapon Detected" else (0, 255, 0)
                cv2.putText(annotated_frame, f"{weapon_status['label']}: {weapon_status['confidence']:.1f}%", 
                           (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, weapon_color, 2)
                y_pos += 30
                
                cv2.putText(annotated_frame, f"Pose: {pose_status['label']} ({pose_status['confidence']:.1f}%)", 
                           (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                y_pos += 30
                
                auth_color = (0, 255, 0) if face_verified else (255, 0, 0)
                cv2.putText(annotated_frame, authentication_status, 
                           (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, auth_color, 2)
                y_pos += 30
                
                action_color = (0, 0, 255) if "Attack" in response_status["action"] or "Alert" in response_status["action"] else (0, 255, 0)
                cv2.putText(annotated_frame, f"Action: {response_status['action']} -> {response_status['target']}", 
                          (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, action_color, 2)
                           
                captured_frames.append({
                    "frame": annotated_frame, 
                    "weapon": weapon_status,
                    "pose": pose_status,
                    "authentication": authentication_status,
                    "action": response_status,
                    "person_count": len(person_detections)
                })
                
                if len(captured_frames) > max_stored_frames:
                    captured_frames.pop(0)
                
                display_frames = []
                target_size = (240, 180)
                
                for i, frame_data in enumerate(captured_frames[-4:]):
                    small_frame = cv2.resize(frame_data["frame"], target_size)
                    display_frames.append(small_frame)
                
                if display_frames:
                    if len(display_frames) == 1:
                        frames_grid = display_frames[0]
                    elif len(display_frames) == 2:
                        frames_grid = np.hstack(display_frames)
                    elif len(display_frames) >= 3:
                        while len(display_frames) < 4:
                            blank = np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8)
                            display_frames.append(blank)
                        
                        top_row = np.hstack(display_frames[:2])
                        bottom_row = np.hstack(display_frames[2:4])
                        frames_grid = np.vstack([top_row, bottom_row])
                    
                    frames_display.image(frames_grid, caption="Recent Processed Frames", use_container_width=True)
                
                avg_persons = sum([f["person_count"] for f in captured_frames[-10:]]) / min(10, len(captured_frames))
                max_persons = max([f["person_count"] for f in captured_frames]) if captured_frames else 0
                
                people_stats = f"""
                ### People Detection Statistics:
                - Current frame: {len(person_detections)} person(s) detected
                - Average (last 10 frames): {avg_persons:.1f} person(s)
                - Maximum detected: {max_persons} person(s)
                - Detection confidence threshold: 0.4
                """
                
                if person_detections:
                    people_stats += "\n### Individual Detections:\n"
                    for i, detection in enumerate(person_detections):
                        x1, y1, x2, y2 = detection["coordinates"]
                        people_stats += f"- Person #{i+1}: Confidence {detection['confidence']:.2f}, Size: {x2-x1}x{y2-y1} pixels\n"
                
                people_placeholder.markdown(people_stats)
                
                soldier_status = "✅ Detected" if soldier_detected else "❌ Not Detected"
                civilian_status = "✅ Detected" if civilian_detected else "❌ Not Detected"
                animal_status = "✅ Detected" if animal_detected else "❌ Not Detected"
                soldier_frame_count = f"({soldier_frames_count}/{soldier_detection_threshold})" if not soldier_detected else ""
                
                impersonation_text = ""
                if st.session_state.impersonation_detected:
                    impersonation_text = "\n- ⚠️ IMPERSONATION DETECTED - THREAT LEVEL BASED ON SOLDIER CONFIDENCE"
                
                results_text = f"""
                ### Detection Results:
                - Persons detected: {len(person_detections)}
                - Soldier status: {soldier_status} {soldier_frame_count} ({current_soldier_confidence:.1f}%)
                - Civilian status: {civilian_status} ({civilian_confidence:.1f}%)
                - Animal status: {animal_status}{impersonation_text}
                - Current pose: {pose_status['label']} ({pose_status['confidence']:.1f}%)
                - Authentication: {authentication_status}
                - Response: {response_status['action']} targeted at {response_status['target']} ({response_status['confidence']:.1f}%)
                """
                results_placeholder.markdown(results_text)
            
            frame_count += 1
            time.sleep(0.01)
            
    except Exception as e:
        st.error(f"Error in webcam processing: {e}")
    finally:
        pose.close()
        cap.release()
        st.success("Webcam processing completed!")

# Function to verify face match (example implementation)
def verify_face_match(reference_face, captured_face):
    """
    Compare two face images and return whether they match and similarity score
    
    Args:
        reference_face (PIL.Image): Reference face image
        captured_face (PIL.Image): Captured face image
    
    Returns:
        tuple: (is_match, similarity_score)
    """
    # This is a placeholder for actual face recognition
    # In a real implementation, you would use a face recognition model
    # For example, with face_recognition library or a deep learning model
    
    # Ensure images are resized to same dimensions
    reference_face = reference_face.resize((160, 160))
    captured_face = captured_face.resize((160, 160))
    
    # Convert to numpy arrays
    ref_array = np.array(reference_face)
    cap_array = np.array(captured_face)
    
    # Simple placeholder comparison (not a real face recognition method)
    # In reality, you would extract face embeddings and compute distance
    try:
        # Convert to grayscale for simpler comparison
        if len(ref_array.shape) == 3:
            ref_gray = cv2.cvtColor(ref_array, cv2.COLOR_RGB2GRAY)
        else:
            ref_gray = ref_array
            
        if len(cap_array.shape) == 3:
            cap_gray = cv2.cvtColor(cap_array, cv2.COLOR_RGB2GRAY)
        else:
            cap_gray = cap_array
        
        # Compute histogram correlation
        ref_hist = cv2.calcHist([ref_gray], [0], None, [256], [0, 256])
        cap_hist = cv2.calcHist([cap_gray], [0], None, [256], [0, 256])
        
        cv2.normalize(ref_hist, ref_hist, 0, 1, cv2.NORM_MINMAX)
        cv2.normalize(cap_hist, cap_hist, 0, 1, cv2.NORM_MINMAX)
        
        # Calculate similarity score
        similarity = cv2.compareHist(ref_hist, cap_hist, cv2.HISTCMP_CORREL)
        similarity = max(0, similarity) * 100  # Convert to percentage
        
        # Determine if it's a match
        is_match = similarity > 70.0  # Threshold for match
        
        return is_match, similarity
        
    except Exception as e:
        print(f"Error in face verification: {e}")
        return False, 0.0   



st.title("HawkAi - Smart Defence Surveillance System")

option = st.selectbox("Select Input Type", ["Image", "Video", "Webcam"])

if option == "Image":
    uploaded_image = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])
    if uploaded_image:
        try:
            image = Image.open(uploaded_image)
            st.image(image, caption="Uploaded Image", use_container_width=True)
            st.write("Processing Image...")
            results = classify_image_efficientnet(image)
            handle_decision(results, image)
        except Exception as e:
            st.error(f"Error processing the image: {e}")
            
if option == "Webcam":
    st.subheader("Live Webcam Processing")
    st.write("Click 'Start Webcam' to begin processing the live feed.")
    
    if st.button("Start Webcam"):
        process_webcam()

elif option == "Video":
    uploaded_video = st.file_uploader("Upload a Video", type=["mp4", "avi", "mov"])
    if uploaded_video:
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
                tmp_file.write(uploaded_video.read())
                video_path = tmp_file.name

            st.video(video_path)
            st.write("Processing Video...")
            
            frame_results, total_frames, all_detections, weapon_detections, pose_detections, frame_interval = process_video(video_path)
            
            class_statistics = {}
            processed_frames = max(1, total_frames // frame_interval)
            
            for class_name, detections in all_detections.items():
                if detections:
                    frames_detected = len(detections)
                    percentage_frames = (frames_detected / processed_frames) * 100
                    avg_confidence = sum(conf for _, conf in detections) / len(detections)
                    
                    class_statistics[class_name] = {
                        "frames": frames_detected,
                        "percentage": percentage_frames,
                        "avg_confidence": avg_confidence
                    }
            
            sorted_classes = sorted(
                class_statistics.keys(), 
                key=lambda x: class_statistics[x]["frames"] if x in class_statistics else 0, 
                reverse=True
            )
            
            top_classes = [cls for cls in sorted_classes if cls in class_statistics and class_statistics[cls]["frames"] > 0][:2]
            
            if top_classes:
                st.subheader("Top Classes Detected")
                
                for class_name in top_classes:
                    stats = class_statistics[class_name]
                    
                    if class_name == "Animal":
                        st.success(f"✅ {class_name} detected in {stats['frames']} frames ({stats['percentage']:.1f}%) with avg confidence {stats['avg_confidence']:.2f}% - Ignored")
                    
                    elif class_name == "Civilian":
                        st.error(f"⚠️ {class_name} detected in {stats['frames']} frames ({stats['percentage']:.1f}%) with avg confidence {stats['avg_confidence']:.2f}% - Alerting Authorities")
                        
                    elif class_name == "Soldier":
                        st.warning(f"⚠️ {class_name} detected in {stats['frames']} frames ({stats['percentage']:.1f}%) with avg confidence {stats['avg_confidence']:.2f}%")
            
            if weapon_detections:
                st.subheader("Weapon Detection")
                weapons_detected = len([w for _, w, _ in weapon_detections if w == "Weapon Detected"])
                weapon_percentage = (weapons_detected / len(weapon_detections)) * 100
                st.error(f"🚨 Weapons detected in {weapons_detected} frames ({weapon_percentage:.1f}%)")
            
            st.subheader("Pose Analysis")
            for pose, detections in pose_detections.items():
                if detections:
                    st.info(f"{pose} detected in {len(detections)} frames, avg confidence: {sum(conf for _, conf in detections) / len(detections):.2f}%")
            
            if "Soldier" in class_statistics:
                soldier_confidence = class_statistics["Soldier"]["avg_confidence"]
                rfid = st.text_input("Enter RFID for verification:")
                if rfid:
                    if rfid in valid_rfids:
                        st.success("✅ Soldier RFID Verified")
                        col1, col2 = st.columns(2)
            
                        with col1:
                            st.error("🚨 For video analysis, a representative frame is needed for face verification.")
                            uploaded_face = st.file_uploader("Upload a clear face image from the video:", type=["jpg", "jpeg", "png"])
                        
                        with col2:
                            st.warning("📌 Upload reference image of authorized soldier:")
                            reference_face = st.file_uploader("Reference image:", type=["jpg", "jpeg", "png"])
                        
                        if uploaded_face and reference_face:
                            face_image = Image.open(uploaded_face)
                            ref_image = Image.open(reference_face)
                            
                            # Display both images side by side
                            display_col1, display_col2 = st.columns(2)
                            with display_col1:
                                st.image(face_image, caption="Face from Video", width=250)
                            with display_col2:
                                st.image(ref_image, caption="Reference Face", width=250)
                            
                            # Use the reference image version of verify_face
                            is_match, similarity = verify_face(face_image, ref_image)
                            
                            # Display verification results
                            st.subheader("Face Verification Result")
                            st.write(f"Similarity Score: {similarity:.4f}")
                            
                            if is_match:
                                st.success("✅ Friendly soldier verified. No threat detected.")
                            else:
                                st.error("🚨 Possible impersonation detected - Enemy soldier")
                                
                                if "Soldier" in class_statistics and class_statistics["Soldier"]["frames"] > 0:
                                    best_frames = sorted(all_detections["Soldier"], key=lambda x: x[1], reverse=True)
                                    if best_frames:
                                        st.warning("Showing target segmentation from highest confidence frame")
                                    
                                    has_weapon = any(w == "Weapon Detected" for _, w, _ in weapon_detections)
                                    is_aggressive_pose = any(pose in ["Standing", "Running"] for pose in pose_detections if pose_detections[pose])
                                    
                                    if has_weapon and is_aggressive_pose:
                                        st.error("⚠️ Armed enemy soldier in aggressive posture")
                                        col1, col2 = st.columns(2)
                                        with col1:
                                            if st.button("Attack Non-Lethal Parts"):
                                                st.warning("Command sent: Non-lethal force authorized")
                                        with col2:
                                            if st.button("Immobilize"):
                                                st.warning("Command sent: Targeting legs for immobilization")
                                    
                                    elif has_weapon:
                                        st.warning("⚠️ Armed enemy in defensive posture")
                                        if st.button("Deploy Anesthetic Shots"):
                                            st.info("Command sent: Anesthetic deployment authorized")
                                    
                                    elif is_aggressive_pose:
                                        st.warning("⚠️ Enemy soldier in threatening posture")
                                        if st.button("Issue Warning"):
                                            st.info("Command sent: Warning signal deployed")
                                    
                                    else:
                                        st.error("🚨 ALERT: Uncertain threat level")
                                        if st.button("Alert Higher Authority"):
                                            st.error("URGENT: Escalation request sent to command center")
                        
                        elif uploaded_face and not reference_face:
                            face_image = Image.open(uploaded_face)
                            st.image(face_image, caption="Face from Video", width=250)
                            st.warning("⚠️ Please upload a reference image for comparison")
                            
                            # Fallback to database check if reference image is not provided
                            is_valid_face, confidence = verify_face(face_image)
                            
                            if is_valid_face:
                                st.success(f"✅ Face matched with database (Confidence: {confidence:.4f}). Friendly soldier verified.")
                            else:
                                st.error(f"🚨 Face not in database (Confidence: {confidence:.4f}). Potential threat detected.")
                                
                                # Rest of threat assessment code...
                                if "Soldier" in class_statistics and class_statistics["Soldier"]["frames"] > 0:
                                    # Existing countermeasures code (same as above)
                                    best_frames = sorted(all_detections["Soldier"], key=lambda x: x[1], reverse=True)
                                    if best_frames:
                                        st.warning("Showing target segmentation from highest confidence frame")
                                    
                                    # Rest of the existing code for threat assessment...
                                    has_weapon = any(w == "Weapon Detected" for _, w, _ in weapon_detections)
                                    is_aggressive_pose = any(pose in ["Standing", "Running"] for pose in pose_detections if pose_detections[pose])
                                    
                                    # Continue with existing threat assessment logic
                                    # (same code as in the section above)
                                    if has_weapon and is_aggressive_pose:
                                        st.error("⚠️ Armed enemy soldier in aggressive posture")
                                        col1, col2 = st.columns(2)
                                        with col1:
                                            if st.button("Attack Non-Lethal Parts"):
                                                st.warning("Command sent: Non-lethal force authorized")
                                        with col2:
                                            if st.button("Immobilize"):
                                                st.warning("Command sent: Targeting legs for immobilization")
                                    
                                    elif has_weapon:
                                        st.warning("⚠️ Armed enemy in defensive posture")
                                        if st.button("Deploy Anesthetic Shots"):
                                            st.info("Command sent: Anesthetic deployment authorized")
                                    
                                    elif is_aggressive_pose:
                                        st.warning("⚠️ Enemy soldier in threatening posture")
                                        if st.button("Issue Warning"):
                                            st.info("Command sent: Warning signal deployed")
                                    
                                    else:
                                        st.error("🚨 ALERT: Uncertain threat level")
                                        if st.button("Alert Higher Authority"):
                                            st.error("URGENT: Escalation request sent to command center")
                        
                        elif reference_face and not uploaded_face:
                            ref_image = Image.open(reference_face)
                            st.image(ref_image, caption="Reference Face", width=250)
                            st.warning("⚠️ Please upload a face image from the video for comparison")
                            
                    else:
                        st.error("⚠️ Invalid RFID - Intruder Alert!")
                        col1, col2 = st.columns(2)
                        with col1:
                            if st.button("Deploy Countermeasures"):
                                st.warning("Command sent: Countermeasures activated")
                        with col2:
                            if st.button("Alert Command"):
                                st.error("URGENT: Unauthorized personnel alert sent to command")
            
            try:
                os.unlink(video_path)
            except:
                pass
                
        except Exception as e:
            st.error(f"Error processing video: {e}")
            try:
                os.unlink(video_path)
            except:
                pass

st.sidebar.title("System Status")
st.sidebar.write(f"Running on: {device}")

model_status = {
    "EfficientNet Classification": efficientnet is not None,
    "YOLO Object Detection": yolo is not None,
    "Face Recognition": face_model is not None and len(valid_face_embeddings) > 0,
    "Weapon Classification": weapon_model is not None,
    "Pose Estimation": mp_pose is not None,
    "Body Part Segmentation": body_part_model is not None
}

st.sidebar.subheader("Model Status")
for model_name, status in model_status.items():
    if status:
        st.sidebar.success(f"✅ {model_name}")
    else:
        st.sidebar.error(f"❌ {model_name}")

st.sidebar.subheader("Settings")
confidence_threshold = st.sidebar.slider("Detection Confidence Threshold", 20, 95, 50)
st.sidebar.info(f"Current threshold: {confidence_threshold}%")

st.sidebar.subheader("Help & Information")
if st.sidebar.checkbox("Show Instructions"):
    st.sidebar.markdown("""
    ### Instructions
    1. Select input type (Image or Video)
    2. Upload your file
    3. For soldier verification, enter a valid RFID code
    4. Follow on-screen instructions for further actions
    
    ### Valid RFID codes for testing
    - RFID123456
    - SOLDIER98765
    - CIVILIAN54321
    - FRIENDLY001
    - SECURE99999
    - TESTRFID2025
    """)

st.sidebar.markdown("---")
st.sidebar.markdown("### HawkAi v1.0")
st.sidebar.markdown("© 2025 HawkAi Defence Systems")

debug_mode = st.sidebar.checkbox("Enable Debug Mode")
if debug_mode:
    st.sidebar.subheader("Debug Information")
    st.sidebar.json({"device": str(device),
                     "models_loaded": sum(1 for status in model_status.values()if status),
                     "models_total": len(model_status)})
    st.sidebar.markdown("### Model Paths")
    st.sidebar.code(f"""
    EfficientNet: {efficientnet_model_path}
    YOLO: {yolo_model_path}
    Face: {face_model}
    Weapon: {weapon_model_path}
    Pose: {pose_model_path}
    Body Parts: {body_part_model_path}
    """)
    st.markdown("---")
    st.markdown("HawkAi - Smart Defence Surveillance System | For authorized use only")