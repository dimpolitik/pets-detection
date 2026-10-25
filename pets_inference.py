"""
Validate PETS detection models on the train, validation, and test splits.
"""
import csv
import os
from pathlib import Path
from typing import Optional

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import yaml
from ultralytics import YOLO

CURVE_CONF = 0.001
ROOT = Path(__file__).resolve().parent
DATASET = ROOT / "pets.yaml"
OUTPUT_CSV = ROOT / "pets_inference_metrics.csv"
OUTPUT_LOG = ROOT / "pets_inference.log"
# Training runs stay in the parent workspace; inference artifacts stay here.
WORKSPACE_ROOT = ROOT.parent if (ROOT.parent / "runs" / "detect").is_dir() else ROOT
RUNS_DIR = WORKSPACE_ROOT / "runs" / "detect"
INFERENCE_DIR = ROOT / "runs" / "pets_inference"

EXPERIMENTS = {
    "new_pets_y26_large_1280_16": 1280,
    "new_pets_y26_large_960_16": 960,
    "new_pets_y26_xlarge_640_24": 640,
    "new_pets_y26_large_640_16": 640,
    "new_pets_y26_xlarge_640_16": 640,
    "new_pets_y26_xlarge_960_16": 960,
}
LOCAL_CHECKPOINTS = {
    "new_pets_y26_large_960_16": ROOT / "weights" / "best.pt",
}

SPLITS = ("train", "val", "test")
CSV_FIELDS = (
    "Run",
    "Split",
    "Image size",
    "Confidence",
    "Class",
    "Images",
    "Instances",
    "Precision",
    "Recall",
    "mAP50",
    "mAP50-95",
)


def capture_validator(validator) -> dict:
    return {
        "images": int(validator.seen),
        "instances_by_class": validator.metrics.nt_per_class.copy(),
        "names": validator.names,
        "metrics": validator.metrics,
    }


def get_image_size(run_dir: Path, expected_image_size: int) -> int:
    args_path = run_dir / "args.yaml"
    if not args_path.is_file():
        raise FileNotFoundError("Training arguments not found: {}".format(args_path))

    with args_path.open(encoding="utf-8") as args_file:
        args = yaml.safe_load(args_file)

    image_size = args.get("imgsz") if isinstance(args, dict) else None
    if not isinstance(image_size, int) or image_size <= 0:
        raise ValueError("Invalid or missing imgsz in {}".format(args_path))
    if image_size != expected_image_size:
        raise ValueError(
            "Expected imgsz={} for {}, but args.yaml specifies {}".format(
                expected_image_size, run_dir.name, image_size
            )
        )
    return image_size


def get_split_confidence_curves(
    model,
    run_name: str,
    split: str,
    image_size: int,
    output_dir: Optional[Path] = None,
) -> dict:
    # Capture class instance counts alongside the per-class confidence curves.
    captured = {}
    project_dir = output_dir if output_dir is not None else INFERENCE_DIR
    output_name = (
        "threshold_selection/{}".format(split)
        if output_dir is not None
        else "{}_confidence_{}".format(run_name, split)
    )

    def on_val_end(validator) -> None:
        captured["instances_by_class"] = np.asarray(
            validator.metrics.nt_per_class
        ).copy()
        captured["names"] = validator.names

    model.add_callback("on_val_end", on_val_end)
    try:
        results = model.val(
            data=str(DATASET),
            split=split,
            imgsz=image_size,
            conf=CURVE_CONF,
            plots=False,
            project=str(project_dir),
            name=output_name,
            exist_ok=True,
            workers=0,
        )
    finally:
        model.callbacks["on_val_end"].remove(on_val_end)

    if not captured:
        raise RuntimeError(
            "Validation did not provide counts for {} ({})".format(run_name, split)
        )

    curves = results.box.curves_results
    if len(curves) < 4:
        raise RuntimeError(
            "Validation did not provide precision-confidence and "
            "recall-confidence curves for {} ({})".format(run_name, split)
        )

    confidence_values = np.asarray(curves[2][0])
    precision_by_class = np.asarray(curves[2][1])
    recall_by_class = np.asarray(curves[3][1])
    names = captured["names"]
    class_ids = sorted(names if isinstance(names, dict) else range(len(names)))
    expected_shape = (len(class_ids), confidence_values.size)
    if (
        confidence_values.ndim != 1
        or confidence_values.size == 0
        or precision_by_class.shape != expected_shape
        or recall_by_class.shape != expected_shape
        or captured["instances_by_class"].size <= max(class_ids)
        or not np.array_equal(confidence_values, np.asarray(curves[3][0]))
    ):
        raise RuntimeError(
            "Validation returned incompatible confidence curves for {} ({})".format(
                run_name, split
            )
        )

    return {
        "confidence_values": confidence_values,
        "precision_by_class": precision_by_class,
        "recall_by_class": recall_by_class,
        "instances_by_class": captured["instances_by_class"],
        "class_ids": class_ids,
        "names": names,
    }


def select_confidence(
    model,
    run_name: str,
    image_size: int,
    output_dir: Optional[Path] = None,
) -> float:
    """Maximize macro F1 pooled over training and validation predictions."""
    split_curves = {
        split: get_split_confidence_curves(
            model, run_name, split, image_size, output_dir
        )
        for split in ("train", "val")
    }
    train_curves = split_curves["train"]
    val_curves = split_curves["val"]
    if train_curves["class_ids"] != val_curves["class_ids"]:
        raise RuntimeError(
            "Train and validation classes differ for {}".format(run_name)
        )
    if not np.array_equal(
        train_curves["confidence_values"], val_curves["confidence_values"]
    ):
        raise RuntimeError(
            "Train and validation confidence grids differ for {}".format(run_name)
        )

    class_f1_curves = []
    for class_index, class_id in enumerate(train_curves["class_ids"]):
        # Pool true/false positives and false negatives before computing F1.
        instance_count = int(
            train_curves["instances_by_class"][class_id]
            + val_curves["instances_by_class"][class_id]
        )
        if instance_count <= 0:
            continue

        true_positives = (
            train_curves["recall_by_class"][class_index]
            * train_curves["instances_by_class"][class_id]
            + val_curves["recall_by_class"][class_index]
            * val_curves["instances_by_class"][class_id]
        )
        false_positives = np.zeros_like(true_positives)
        for curves in (train_curves, val_curves):
            recall = curves["recall_by_class"][class_index]
            precision = curves["precision_by_class"][class_index]
            split_true_positives = recall * curves["instances_by_class"][class_id]
            predicted_positives = np.divide(
                split_true_positives,
                precision,
                out=np.zeros_like(split_true_positives),
                where=precision > 0,
            )
            false_positives += np.maximum(
                predicted_positives - split_true_positives, 0
            )

        false_negatives = instance_count - true_positives
        f1 = np.divide(
            2 * true_positives,
            2 * true_positives + false_positives + false_negatives,
            out=np.zeros_like(true_positives),
            where=(2 * true_positives + false_positives + false_negatives) > 0,
        )
        class_f1_curves.append(f1)

    if not class_f1_curves:
        raise RuntimeError("No labeled classes found for {}".format(run_name))

    mean_f1 = np.mean(class_f1_curves, axis=0)
    confidence_values = train_curves["confidence_values"]
    threshold = float(confidence_values[int(mean_f1.argmax())])
    print(
        "Selected confidence for {} from combined train+validation macro F1: "
        "{:.4f} (mean F1={:.4f})".format(run_name, threshold, float(mean_f1.max())),
        flush=True,
    )
    return threshold


def evaluate_split(
    model, run_name: str, split: str, image_size: int, confidence: float
) -> dict:
    captured = {}

    def on_val_end(validator) -> None:
        captured.update(capture_validator(validator))

    model.add_callback("on_val_end", on_val_end)
    try:
        model.val(
            data=str(DATASET),
            split=split,
            imgsz=image_size,
            save_json=True,
            plots=False,
            conf=confidence,
            project=str(INFERENCE_DIR),
            name="{}_{}".format(run_name, split),
            exist_ok=True,
            workers=0,
        )
    finally:
        model.callbacks["on_val_end"].remove(on_val_end)

    if not captured:
        raise RuntimeError(
            "Validation did not provide metrics for {} ({})".format(run_name, split)
        )
    return captured


def metric_rows(
    run_name: str, split: str, image_size: int, confidence: float, captured: dict
) -> list:
    metrics = captured["metrics"]
    box = metrics.box
    images = captured["images"]
    counts = captured["instances_by_class"]
    names = captured["names"]
    rows = [
        {
            "Run": run_name,
            "Split": split,
            "Image size": image_size,
            "Confidence": confidence,
            "Class": "all",
            "Images": images,
            "Instances": int(sum(counts)),
            "Precision": float(box.mp),
            "Recall": float(box.mr),
            "mAP50": float(box.map50),
            "mAP50-95": float(box.map),
        }
    ]

    for metric_index, class_id in enumerate(box.ap_class_index):
        rows.append(
            {
                "Run": run_name,
                "Split": split,
                "Image size": image_size,
                "Confidence": confidence,
                "Class": names[class_id],
                "Images": images,
                "Instances": int(counts[class_id]),
                "Precision": float(box.p[metric_index]),
                "Recall": float(box.r[metric_index]),
                "mAP50": float(box.ap50[metric_index]),
                "mAP50-95": float(box.ap[metric_index]),
            }
        )
    return rows


def write_log_rows(log_file, rows: list) -> None:
    log_file.write(
        "{:<22}{:>11}{:>11}{:>11}{:>11}{:>11}{:>11}\n".format(
            "Class", "Images", "Instances", "Box(P)", "R", "mAP50", "mAP50-95"
        )
    )
    for row in rows:
        log_file.write(
            "{:<22}{:>11}{:>11}{:>11.3f}{:>11.3f}{:>11.3f}{:>11.3f}\n".format(
                row["Class"],
                row["Images"],
                row["Instances"],
                row["Precision"],
                row["Recall"],
                row["mAP50"],
                row["mAP50-95"],
            )
        )
    log_file.flush()


def main() -> None:
    if not DATASET.is_file():
        raise FileNotFoundError("Dataset YAML not found: {}".format(DATASET))

    experiments = [
        (
            RUNS_DIR / run_name,
            image_size,
            LOCAL_CHECKPOINTS.get(
                run_name, RUNS_DIR / run_name / "weights" / "best.pt"
            ),
        )
        for run_name, image_size in EXPERIMENTS.items()
    ]
    missing_files = [
        path
        for run_dir, _, checkpoint in experiments
        for path in (checkpoint, run_dir / "args.yaml")
        if not path.is_file()
    ]
    if missing_files:
        raise FileNotFoundError(
            "Required experiment files not found: {}".format(
                ", ".join(str(path) for path in missing_files)
            )
        )

    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as csv_file, OUTPUT_LOG.open(
        "w", encoding="utf-8"
    ) as log_file:
        writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS)
        writer.writeheader()

        for run_dir, expected_image_size, checkpoint in experiments:
            run_name = run_dir.name
            model = YOLO(str(checkpoint))
            image_size = get_image_size(run_dir, expected_image_size)
            confidence = select_confidence(model, run_name, image_size)
            for split in SPLITS:
                print(
                    "Evaluating {} on {} (imgsz={}, conf={:.4f})".format(
                        run_name, split, image_size, confidence
                    ),
                    flush=True,
                )
                log_file.write("\n{} - {}\n".format(run_name, split))
                log_file.write(
                    "Image size: {}; confidence selected on train+validation: "
                    "{:.4f}\n".format(
                        image_size, confidence
                    )
                )
                captured = evaluate_split(
                    model, run_name, split, image_size, confidence
                )
                rows = metric_rows(
                    run_name, split, image_size, confidence, captured
                )
                writer.writerows(rows)
                csv_file.flush()
                write_log_rows(log_file, rows)

    print("Wrote metrics to {} and {}".format(OUTPUT_CSV, OUTPUT_LOG))


if __name__ == "__main__":
    main()
