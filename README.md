# PETS Detection with YOLOv8

Object detection system for identifying pets using YOLOv8 models trained on the PETS (sharks and skates) dataset.

## Overview

This project implements a comprehensive pipeline for training, validating, and performing inference on pet detection tasks using Ultralytics YOLOv8. 

## Project Structure

```
Repository root
├── README.md                          # This file
├── pets.yaml                          # Dataset 
├── pets_code.py                       # Main training 
├── pets_explainability.py             # Model explainability analysis 
└── lrp/                               # Layer-wise Relevance Propagation
    ├── initializers.py
    ├── inverter.py
    ├── inverter_util.py
    ├── rules.py
    ├── utils.py
    ├── yolo.py
    └── common.py
```

Install dependencies:
```bash
pip install -r requirements.txt 
```

## Usage

Only scripts present in this repository are listed below. Run them from the repository root.

Training:
```bash
python pets_training.py
```

Generate model explanations (LRP):
```bash
python pets_explainability.py
```

## Configuration

The `pets.yaml` set up dataset paths, class names, and the number of classes.

## Model Explainability

Layer-wise Relevance Propagation (LRP) helper code is available in the `lrp/` folder.
Generate explanations using `pets_explainability.py`.

## Data access

The PETS dataset and trained model checkpoints are not publicly distributable and are available from the project owner upon request. A small set of representative images is included for inspection in the `sample_images/` folder.

