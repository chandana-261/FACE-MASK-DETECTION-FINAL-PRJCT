# FACE-MASK-DETECTION-FINAL-PRJCT
# Face Mask Detection with Live Alert System

A real-time computer vision system that detects whether a person is wearing a face mask or not using a webcam feed. Built using Python, TensorFlow/Keras, and OpenCV.

## 📌 Features

- 🔍 Real-time face detection using webcam
- 🎯 Face mask classification using a CNN model
- 🚨 Visual alert for people not wearing masks
- ⚙️ Easy to integrate into surveillance systems

## 🧠 Model Overview

A custom Convolutional Neural Network (CNN) trained on a dataset of images with and without face masks.

- Input size: 224x224
- Architecture: Convolutional + MaxPooling + Dropout + Dense
- Output: Softmax layer (2 classes: Mask, No Mask)

## 🗃️ Dataset

The dataset consists of labeled images:
- `with_mask`
- `without_mask`

Each image is resized and normalized. Data is split into training and validation sets.

## 🚀 Installation

1. **Clone the repository:**
```bash
git clone https://github.com/yourusername/face-mask-detection.git
cd face-mask-detection

INSTALL DEPENDENCIES:

bash
Copy
Edit
pip install -r requirements.txt
Note: Ensure Python 3.7+ and pip are installed.

📦 Required Libraries
Python

TensorFlow / Keras

OpenCV

NumPy

Matplotlib

scikit-learn

🛠️ How to Use
Train the model (if needed):

python
Copy
Edit
python train_model.py
Run real-time detection:

python
Copy
Edit
python detect_mask_video.py
The webcam will start and display live video with labeled bounding boxes.

📊 Results
High training/validation accuracy

Smooth real-time webcam performance

Minimal false detections

📍 Real-Time Use Cases of Face Mask Detection System
🏥 Hospitals & Clinics

Ensure patients, doctors, and visitors are wearing masks in critical zones like ICUs, operation theaters, and general wards.

Reduces risk of infection spread in healthcare settings.

🏫 Schools & Colleges

Monitor mask compliance among students and faculty during pandemics or flu seasons.

Automatically detect rule violations without manual intervention.

🏢 Corporate Offices

Use at building entrances or reception areas to ensure employees and visitors wear masks before entering.

Supports HR or security teams in enforcing health policies.

🏬 Shopping Malls & Airports

Deployed in public areas with heavy footfall to monitor crowds for compliance.

Enhances public safety without requiring extra manpower.

🚇 Public Transport Terminals

Monitor passengers in real time at metro stations, bus stops, and railway platforms.

Prevent entry of unmasked individuals into transport facilities.

🧪 Research Labs & Cleanrooms

Ensure personnel wear required safety gear (e.g., masks) in bio-sensitive or dust-controlled environments.

Helps maintain strict compliance with laboratory protocols.

🧩 Future Improvements
Add audio alerts using pyttsx3 or buzzer

Web/mobile deployment

Use MobileNetV2 or ResNet for enhanced accuracy

Add face recognition for attendance systems

📄 License
This project is open-source and available under the MIT License.

🙋‍♂️ Author
LAKSHMI CHANDANA YANAMANDRA
GitHub: @chandana-261


