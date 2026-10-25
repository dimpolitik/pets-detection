# PETS Detection with YOLO26

Train YOLO26 models to detect sharks and skates in the PETS dataset, then evaluate detection and per-image count accuracy.

## Project files

- `pets_training.py` — train and evaluate one YOLO26-L or YOLO26-X model.
- `pets_inference.py` — evaluate configured training runs and save split metrics.
- `pets_y26l_960_optimal_inference.py` — detailed evaluation for the YOLO26-L 960px run, including count-error plots.
- `weights/best.pt` — checkpoint used by the focused YOLO26-L 960px evaluation.
- `pets.yaml` — dataset paths and class names.
- `pets_explainability.py` and `lrp/` — explainability entry point and Layer-wise Relevance Propagation helpers.
- `sample_images/` — example PETS images.
- `requirements.txt` — Python dependencies.

![Example PETS dataset image](sample_images/20190813_122602.jpg)

*Example image from the PETS dataset.*

## Dataset

The dataset configuration in `pets.yaml` expects `images/` and `labels/`
directories with `train`, `val`, and `test` splits. Its classes are `shark`
and `skate`. Update the paths in `pets.yaml` to match your local dataset
layout. The training script also has a `TEST_IMAGES` path that specifies which
test images to predict on; update it in `pets_training.py` if your test images
are elsewhere.

Keep the test split independent: use training and validation data to make
model or threshold choices, and reserve test results for final evaluation. The
dataset and most trained checkpoints are not included in this repository.
The focused YOLO26-L 960px checkpoint is included under `weights/`.

## Train a model

1. From the workspace root, install the dependencies:

   ```bash
   pip install -r pets-detection/requirements.txt
   ```

2. Make sure the dataset paths in `pets-detection/pets.yaml` and
   `TEST_IMAGES` in `pets-detection/pets_training.py` point to your data.

3. In `pets_training.py`, set `MODEL_VARIANT` to `"l"` for YOLO26-L or `"x"`
   for YOLO26-X. The script uses the matching pretrained weights
   (`yolo26l.pt` or `yolo26x.pt`). Adjust `IMG`, `EPOCHS`, `BATCH_SIZE`, or
   other training constants in the same file if needed.

4. Run training from the `pets-detection` directory so `pets.yaml` is found:

   ```bash
   cd pets-detection
   python pets_training.py
   ```

The current configuration trains for up to 300 epochs with early-stopping
patience 40, image size 640, batch size 24, and GPUs 0 and 1. It validates on
the train, validation, and test splits, predicts on the test images, and prints
mAP50 for each split. The run name is based on the selected model, image size,
and batch size (for example, `pets_y26_xlarge_640_24`). Ultralytics determines
the exact output location from its `runs/detect` project setting and the
current working directory; check the training log for the saved run path.

## Run inference and evaluation

### Evaluate configured training runs

`pets_inference.py` evaluates the experiments listed in its `EXPERIMENTS`
configuration. Before running it, make sure each configured run exists under
the `runs/detect/` directory the script will use and that its image size
matches the run's `args.yaml`. It checks `../runs/detect/` first when that
directory exists; otherwise it uses `pets-detection/runs/detect/`. The
default experiment names in `pets_inference.py` use a `new_pets_...` prefix,
whereas `pets_training.py` creates names beginning with `pets_y26_`; update
`EXPERIMENTS` to match the runs you trained. The `new_pets_y26_large_960_16`
run uses the local checkpoint in `weights/`; other runs still need their
checkpoints under `runs/detect/`.

From the `pets-detection` directory, run:

```bash
python pets_inference.py
```

For each run, the script chooses a shared confidence threshold by maximizing
mean per-class F1 across pooled train and validation predictions, then
evaluates train, validation, and test with that threshold. The test split is
not used to select it. The script writes `pets_inference_metrics.csv`,
`pets_inference.log`, and additional inference artifacts under
`runs/pets_inference/`.

### Run the focused YOLO26-L 960px evaluation

`pets_y26l_960_optimal_inference.py` loads its checkpoint from
`weights/best.pt` in this project. It expects the dataset at
`../../datasets/pets_create_train_val_test_set` relative to this script. Make
sure the dataset paths are configured before running:

```bash
python pets_y26l_960_optimal_inference.py
```

This evaluates train, validation, and test at 960px using a shared confidence
threshold selected from train and validation. It saves confidence and
precision-recall curves, confusion matrices, prediction labels, class counts,
and a validation-versus-test metrics plot under
`runs/pets_inference/YOLO26l-960-16_optimal/`. It also writes
`per_image_count_errors.csv` and `per_image_count_error_histogram.png`. The
histogram shows shark and skate count errors per image, where count error is
Ground Truth minus Predicted: positive values indicate undercounting and
negative values indicate overcounting. Annotations show mean absolute error.

Both inference scripts use YOLO26's native end-to-end inference path without
adding a separate non-maximum suppression (NMS) postprocessing step.

## Explainability

The `lrp/` directory contains Layer-wise Relevance Propagation helpers, and
`pets_explainability.py` is the explainability entry point.
