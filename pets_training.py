"""
YOLOv8 Training and Validation Script for PETS Detection
"""

import os

import torch
from ultralytics import YOLO

# GPU configuration
# Suppress MKL duplicate library warnings
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Print available CUDA devices and their memory
for i in range(torch.cuda.device_count()):
    info = torch.cuda.get_device_properties(i)
    print(f"CUDA:{i} {info.name}, {info.total_memory / 1024 ** 2}MB")

# Set up device (GPU if available, else CPU)
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"There are {torch.cuda.device_count()} GPU(s) available.")
    print("Device name:", torch.cuda.get_device_name(0))
else:
    print("No GPU available, using the CPU instead.")
    device = torch.device("cpu")

# Configuration
CONF = 0.1  # Confidence threshold for predictions
IMG = 1280  # Image size for model input

# Load pretrained YOLOv8 large model
model = YOLO("yolov8l.pt")

# Train model on PETS dataset
model.train(
    data="pets.yaml",
    epochs=300,
    imgsz=IMG,
    batch=16,
    patience=40,
    name="pets_v8l_1280_16",
    exist_ok=True,
    device=[0, 1],
)

# Validate trained model on train and validation splits
metrics_train = model.val(split="train", conf=CONF)
metrics_val = model.val(split="val", conf=CONF)

# Run inference on test set and save predictions
model.predict(
    source="/home/politikos/datasets/pets/images/val", imgsz=IMG, conf=CONF, save=True
)
