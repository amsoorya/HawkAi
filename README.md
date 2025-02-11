# HawkAi 🚀

## Overview
HawkAi is an AI-driven defense system designed to detect, classify, and respond to unauthorized human incursions into restricted military zones. By leveraging deep learning models and real-time video feeds, the system can differentiate between animals, civilians, and soldiers and take appropriate actions based on classification confidence levels.

---

## System Architecture

### 1️⃣ Classification (EfficientNet-B0)
- Identifies whether the detected entity is:
  - **Animal** → Ignored
  - **Human** → Further classified into:
    - **Civilian**
    - **Soldier**

### 2️⃣ Dress Classification (YOLOv8n)
- Distinguishes between:
  - **Civilian attire**
  - **Military attire (Soldier)**

### 3️⃣ Pose Classification (ResNet50)
- If the entity is a **soldier**, their posture is further analyzed:
  - **Combat stance**
  - **Kneeling**
  - **Standing**
- If the entity is a **civilian**, their posture is also identified:
  - **Standing**
  - **Sitting**
  - **Running**

### 4️⃣ Body Part Detection (DeepLabV3)
- Identifies body parts for non-lethal targeting:
  - **Torso**
  - **Right/Left Leg**
  - **Right/Left Arm**
  - **Right/Left Hand**

### 5️⃣ RFID Verification
- Scans soldiers to determine if they are **friendly** or **enemy**.
- Categories:
  - ✅ **Valid (Friendly Soldier)**
  - ❌ **Invalid (Enemy or Unidentified)**

---

## 📊 Decision Making Based on Confidence Levels

| Confidence Level  | Action Taken |
|------------------|-------------|
| **≥ 80%**  | Non-lethal targeting (e.g., hands, legs) |
| **60 - 80%**  | Anesthetic shots for temporary incapacitation |
| **< 60%**  | Alert higher authorities for further action |

---

## 🛠 Technologies Used
- **YOLOv8n & YOLOv8s** → Object detection and human classification
- **EfficientNet-B0** → High-accuracy classification model
- **ResNet50** → Human pose classification
- **DeepLabV3** → Semantic segmentation for body part identification
- **RFID Scanners** → Soldier verification

---

## ⚙️ Hardware Components
- **Infrared & Thermal Cameras** → Night vision and poor-weather adaptation
- **Edge AI Devices (NVIDIA Jetson)** → Real-time processing at deployment sites
- **RFID Readers** → Identify friendly soldiers
- **Robotic Firearm Systems** → For non-lethal response

---

## 🎯 Project Goals
- Minimize false attacks on friendly forces and civilians.
- Ensure accurate identification of enemies.
- Provide non-lethal response mechanisms whenever possible.

---
