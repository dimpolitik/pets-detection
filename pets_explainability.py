"""Generate and save LRP/CRP explanations for images using a YOLOv8 model.

This script loads a trained YOLO model, wraps it with a Layer-wise Relevance
Propagation (LRP) helper, then iterates over images to
compute and save visual explanations (LRP and contrastive relevance
propagation - CRP) for specific classes (here: 'skate' and 'shark').

Outputs are saved to the `Explanations/` directory as PNG files.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from PIL import Image
from ultralytics import YOLO

from lrp.yolo import YOLOv8LRP

# Force CPU device by default for reproducibility; uncomment to use GPU when available
device = torch.device("cpu")
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Load the trained YOLOv8 model
model = YOLO("runs/detect/pets_v8l_1280_16/weights/best.pt").to(device)

# Instantiate the LRP wrapper for the YOLOv8 model. Parameters:
# - power: exponent for w^power rule
# - eps: small stabilizer to avoid division by zero
lrp = YOLOv8LRP(model, power=2, eps=1e-05, device=device)

# Image preprocessing: resize to a fixed square and convert to tensor
desired_size = (1280, 1280)
transform = torchvision.transforms.Compose(
    [
        torchvision.transforms.Resize(desired_size),
        torchvision.transforms.ToTensor(),
    ]
)

# Ensure output directory exists
if not os.path.exists("Explanations"):
    os.makedirs("Explanations")

# Process each image file in the `Raw_images` folder
for image_file in os.listdir("Raw_images"):
    print(f"Processing: {image_file}")
    # Load and preprocess the image
    image = Image.open(f"Raw_images/{image_file}").convert("RGB")
    image = transform(image).to(device).float()

    # Compute explanations for two target classes. `contrastive=False` gives
    # a standard LRP heatmap, while `contrastive=True` computes a contrastive
    # relevance propagation (CRP) map that highlights both positive and
    # negative contributions.
    explanation_lrp_skate = lrp.explain(image, cls="skate", contrastive=False)
    explanation_lrp_shark = lrp.explain(image, cls="shark", contrastive=False)
    explanation_crp_skate = lrp.explain(image, cls="skate", contrastive=True)
    explanation_crp_shark = lrp.explain(image, cls="shark", contrastive=True)

    # --- Plot and save LRP (non-contrastive) visualizations ---
    plt.figure(figsize=(16, 4))
    plt.subplot(1, 3, 1)
    # Reconstruct an HxWx3 image tensor for display
    plt.imshow(torch.cat([image[c, :, :].unsqueeze(-1) for c in range(3)], axis=-1))
    plt.title("Original Image")
    plt.xticks([])
    plt.yticks([])

    plt.subplot(1, 3, 2)
    plt.imshow(explanation_lrp_skate, cmap="Reds")
    maxrel = np.abs(explanation_lrp_skate).max()
    plt.clim(vmin=0, vmax=maxrel)
    plt.title('Explanation for Class "Skate"')
    plt.xticks([])
    plt.yticks([])

    plt.subplot(1, 3, 3)
    plt.imshow(explanation_lrp_shark, cmap="Reds")
    maxrel = np.abs(explanation_lrp_shark).max()
    plt.clim(vmin=0, vmax=maxrel)
    plt.title('Explanation for Class "Shark"')
    plt.xticks([])
    plt.yticks([])

    plt.savefig(f'Explanations/lrp_{image_file.split(".")[0]}.png')
    plt.close()

    # --- Plot and save CRP (contrastive) visualizations ---
    plt.figure(figsize=(16, 4))
    plt.subplot(1, 3, 1)
    plt.imshow(torch.cat([image[c, :, :].unsqueeze(-1) for c in range(3)], axis=-1))
    plt.title("Original Image")
    plt.xticks([])
    plt.yticks([])

    plt.subplot(1, 3, 2)
    plt.imshow(explanation_crp_skate, cmap="seismic")
    maxrel = np.abs(explanation_crp_skate).max()
    plt.clim(vmin=-maxrel, vmax=maxrel)
    plt.title('Explanation for Class "Skate"')
    plt.xticks([])
    plt.yticks([])

    plt.subplot(1, 3, 3)
    plt.imshow(explanation_crp_shark, cmap="seismic")
    maxrel = np.abs(explanation_crp_shark).max()
    plt.clim(vmin=-maxrel, vmax=maxrel)
    plt.title('Explanation for Class "Shark"')
    plt.xticks([])
    plt.yticks([])

    plt.savefig(f'Explanations/crp_{image_file.split(".")[0]}.png')
    plt.close()
