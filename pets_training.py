"""Train and evaluate one selected YOLO26 model on the PETS dataset."""

import os

import torch
from ultralytics import YOLO

# Suppress MKL duplicate library warnings
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Select one model for this run: "l" for YOLO26-L or "x" for YOLO26-X.
MODEL_VARIANT = "x"
MODEL_CONFIGS = {
    "l": ("yolo26l.pt", "large"),
    "x": ("yolo26x.pt", "xlarge"),
}
MODEL_WEIGHTS, MODEL_SIZE = MODEL_CONFIGS[MODEL_VARIANT]

# Shared training and evaluation configuration
CONF = 0.1  # Confidence threshold
IMG = 640
EPOCHS = 300
BATCH_SIZE = 24
PATIENCE = 40
DEVICE = [0, 1]
TEST_IMAGES = "/home/politikos/datasets/pets_create_train_val_test_set/images/test"
EXPERIMENT_NAME = f"pets_y26_{MODEL_SIZE}_{IMG}_{BATCH_SIZE}"


def main():
    # Print available CUDA devices and their memory
    for i in range(torch.cuda.device_count()):
        info = torch.cuda.get_device_properties(i)
        print(
            f"CUDA:{i} {info.name}, "
            f"{info.total_memory / 1024**2:.0f} MB"
        )

    # Check GPU availability
    if torch.cuda.is_available():
        print(f"There are {torch.cuda.device_count()} GPU(s) available.")
        print("Device name:", torch.cuda.get_device_name(0))
    else:
        print("No GPU available, using CPU.")

    print(f"\nRunning experiment: {EXPERIMENT_NAME} ({MODEL_WEIGHTS})")
    model = YOLO(MODEL_WEIGHTS)

    model.train(
        data="pets.yaml",
        epochs=EPOCHS,
        imgsz=IMG,
        batch=BATCH_SIZE,
        patience=PATIENCE,
        name=EXPERIMENT_NAME,
        flipud=0.0,
        fliplr=0.0,
        dropout=0.0,
        exist_ok=False,
        device=DEVICE,
        seed=42,
    )

    metrics_train = model.val(split="train", imgsz=IMG, conf=CONF, device=0)
    metrics_val = model.val(split="val", imgsz=IMG, conf=CONF, device=0)
    metrics_test = model.val(split="test", imgsz=IMG, conf=CONF, device=0)

    model.predict(
        source=TEST_IMAGES,
        imgsz=IMG,
        conf=CONF,
        save=True,
        device=0,
    )

    print(f"{EXPERIMENT_NAME} train mAP50:", metrics_train.box.map50)
    print(f"{EXPERIMENT_NAME} validation mAP50:", metrics_val.box.map50)
    print(f"{EXPERIMENT_NAME} test mAP50:", metrics_test.box.map50)


if __name__ == "__main__":
    main()
