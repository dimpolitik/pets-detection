"""Evaluate YOLO26l-960-16 on all splits and save curves and count reports."""

import csv
import json
import os
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from ultralytics import YOLO

from pets_inference import DATASET, select_confidence


ROOT = Path(__file__).resolve().parent
RUN_NAME = "new_pets_y26_large_960_16"
# The checkpoint is in the parent workspace; generated artifacts are local.
WORKSPACE_ROOT = ROOT.parent if (ROOT.parent / "runs" / "detect").is_dir() else ROOT
RUN_DIR = WORKSPACE_ROOT / "runs" / "detect" / RUN_NAME
CHECKPOINT = RUN_DIR / "weights" / "best.pt"
IMAGE_SIZE = 960
SPLITS = ("train", "val", "test")
DATASET_ROOT = (
    ROOT / ".." / ".." / "datasets" / "pets_create_train_val_test_set"
).resolve()
OUTPUT_DIR = ROOT / "runs" / "pets_inference" / "YOLO26l-960-16_optimal"
REPORT_PATH = OUTPUT_DIR / "inference_summary.json"
CSV_PATH = OUTPUT_DIR / "split_metrics_and_counts.csv"
COMPARISON_PLOT_PATH = OUTPUT_DIR / "validation_test_metrics_comparison.png"
CLASS_NAMES = ("shark", "skate")
COMPARISON_METRICS = (
    ("Precision", "precision"),
    ("Recall", "recall"),
    ("F1", "f1"),
    ("AP @ IoU 0.50 (AP50)", "ap50"),
    ("AP @ IoU 0.50:0.95 (AP50-95)", "ap50_95"),
)
CSV_FIELDS = (
    "Split",
    "Class",
    "Images",
    "Ground truth",
    "Predicted",
    "Count difference",
    "Precision",
    "Recall",
    "F1",
    "mAP50",
    "mAP50-95",
)


def count_saved_predictions(labels_dir: Path, class_names: dict) -> Counter:
    if not labels_dir.is_dir():
        raise FileNotFoundError("Saved prediction labels not found: {}".format(labels_dir))

    counts = Counter()
    for label_path in labels_dir.glob("*.txt"):
        with label_path.open(encoding="utf-8") as label_file:
            for line_number, line in enumerate(label_file, start=1):
                fields = line.split()
                if not fields:
                    continue
                try:
                    class_id = int(fields[0])
                except ValueError as error:
                    raise ValueError(
                        "Invalid predicted class ID in {} at line {}".format(
                            label_path, line_number
                        )
                    ) from error
                if class_id not in class_names:
                    raise ValueError(
                        "Unexpected predicted class ID {} in {}".format(
                            class_id, label_path
                        )
                    )
                counts[class_names[class_id]] += 1
    return counts


def f1_score(precision: float, recall: float) -> float:
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def plot_validation_test_metrics(summaries: dict, output_path: Path) -> None:
    # Match the repository's reference chart: validation blue, test green.
    figure, axes = plt.subplots(2, 3, figsize=(14, 7), sharey=True)
    axes = axes.ravel()
    positions = range(len(CLASS_NAMES))
    bar_width = 0.34
    colors = {"val": "#5b9bd5", "test": "#70c28a"}
    panel_labels = ("(a)", "(b)", "(c)", "(d)", "(e)")
    font = {"family": "serif"}

    for panel_index, (axis, (metric_title, metric_key)) in enumerate(
        zip(axes, COMPARISON_METRICS)
    ):
        for split_index, split in enumerate(("val", "test")):
            values = [
                summaries[split]["classes"][class_name][metric_key]
                for class_name in CLASS_NAMES
            ]
            offsets = [
                position + (split_index - 0.5) * bar_width
                for position in positions
            ]
            bars = axis.bar(
                offsets,
                values,
                width=bar_width,
                color=colors[split],
            )
            axis.bar_label(bars, fmt="%.2f", padding=3, fontsize=9)

        axis.set_title(
            "{} {}".format(panel_labels[panel_index], metric_title),
            fontdict={"family": "serif", "weight": "bold"},
            pad=8,
        )
        axis.set_xticks(
            list(positions),
            [name.title() for name in CLASS_NAMES],
            fontfamily="serif",
        )
        axis.tick_params(axis="y", labelsize=9)
        axis.set_ylim(0, 1.08)
        axis.grid(axis="y", linestyle="--", alpha=0.3)
        axis.set_axisbelow(True)

    axes[-1].set_visible(False)
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=colors["val"]),
        plt.Rectangle((0, 0), 1, 1, color=colors["test"]),
    ]
    val_counts = summaries["val"]["classes"]
    test_counts = summaries["test"]["classes"]
    labels = [
        "Validation (n = {} images, {} sharks, {} skates)".format(
            summaries["val"]["images"],
            val_counts["shark"]["ground_truth"],
            val_counts["skate"]["ground_truth"],
        ),
        "Test (n = {} images, {} sharks, {} skates)".format(
            summaries["test"]["images"],
            test_counts["shark"]["ground_truth"],
            test_counts["skate"]["ground_truth"],
        ),
    ]
    figure.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.015),
        ncol=2,
        frameon=False,
        prop=font,
    )
    figure.suptitle(
        "YOLO26l-960-16: Validation vs Test Metrics",
        fontsize=15,
        fontweight="bold",
        fontfamily="serif",
        y=0.985,
    )
    axes[0].set_ylabel("Score", fontfamily="serif")
    axes[3].set_ylabel("Score", fontfamily="serif")
    figure.tight_layout(rect=(0.02, 0.08, 1, 0.94))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def evaluate_split(model, split: str, confidence: float) -> dict:
    captured = {}

    def on_val_end(validator) -> None:
        captured["save_dir"] = Path(validator.save_dir)
        captured["images"] = int(validator.seen)
        captured["instances_by_class"] = validator.metrics.nt_per_class.copy()
        captured["names"] = validator.names
        captured["metrics"] = validator.metrics

    model.add_callback("on_val_end", on_val_end)
    try:
        # Use the same selected threshold for every split; test never tunes it.
        model.val(
            data=str(DATASET),
            split=split,
            imgsz=IMAGE_SIZE,
            conf=confidence,
            device=0,
            workers=0,
            plots=True,
            save_txt=True,
            save_conf=True,
            save_json=True,
            project=str(OUTPUT_DIR),
            name=split,
            exist_ok=True,
        )
    finally:
        model.callbacks["on_val_end"].remove(on_val_end)

    if not captured:
        raise RuntimeError("Validation returned no results for split {}".format(split))

    names = captured["names"]
    class_names = {
        int(class_id): str(name).strip().casefold()
        for class_id, name in (
            names.items() if isinstance(names, dict) else enumerate(names)
        )
    }
    if set(class_names.values()) != set(CLASS_NAMES):
        raise ValueError("Unexpected checkpoint class names: {}".format(class_names))

    predicted_counts = count_saved_predictions(
        captured["save_dir"] / "labels", class_names
    )
    ground_truth_counts = {
        class_names[class_id]: int(captured["instances_by_class"][class_id])
        for class_id in class_names
    }
    metrics = captured["metrics"]
    box = metrics.box
    class_metrics = {}
    csv_rows = []

    overall_precision = float(box.mp)
    overall_recall = float(box.mr)
    csv_rows.append(
        {
            "Split": split,
            "Class": "all",
            "Images": captured["images"],
            "Ground truth": int(sum(ground_truth_counts.values())),
            "Predicted": int(sum(predicted_counts.values())),
            "Count difference": int(
                sum(predicted_counts.values()) - sum(ground_truth_counts.values())
            ),
            "Precision": overall_precision,
            "Recall": overall_recall,
            "F1": f1_score(overall_precision, overall_recall),
            "mAP50": float(box.map50),
            "mAP50-95": float(box.map),
        }
    )

    metric_index_by_class = {
        int(class_id): index for index, class_id in enumerate(box.ap_class_index)
    }
    for class_id, class_name in class_names.items():
        if class_id not in metric_index_by_class:
            raise RuntimeError(
                "No validation metrics found for class {!r} on {}".format(
                    class_name, split
                )
            )
        metric_index = metric_index_by_class[class_id]
        precision = float(box.p[metric_index])
        recall = float(box.r[metric_index])
        class_metrics[class_name] = {
            "ground_truth": ground_truth_counts[class_name],
            "predicted": int(predicted_counts[class_name]),
            "count_difference": int(
                predicted_counts[class_name] - ground_truth_counts[class_name]
            ),
            "precision": precision,
            "recall": recall,
            "f1": f1_score(precision, recall),
            # AP summarizes the full confidence ranking, unlike point metrics above.
            "ap50": float(box.ap50[metric_index]),
            "ap50_95": float(box.ap[metric_index]),
        }
        csv_rows.append(
            {
                "Split": split,
                "Class": class_name,
                "Images": captured["images"],
                "Ground truth": ground_truth_counts[class_name],
                "Predicted": int(predicted_counts[class_name]),
                "Count difference": int(
                    predicted_counts[class_name] - ground_truth_counts[class_name]
                ),
                "Precision": precision,
                "Recall": recall,
                "F1": f1_score(precision, recall),
                "mAP50": float(box.ap50[metric_index]),
                "mAP50-95": float(box.ap[metric_index]),
            }
        )

    return {
        "images": captured["images"],
        "confidence": confidence,
        "ground_truth_total": int(sum(ground_truth_counts.values())),
        "predicted_total": int(sum(predicted_counts.values())),
        "count_difference": int(
            sum(predicted_counts.values()) - sum(ground_truth_counts.values())
        ),
        "overall": {
            "precision": overall_precision,
            "recall": overall_recall,
            "f1": f1_score(overall_precision, overall_recall),
            "map50": float(box.map50),
            "map50_95": float(box.map),
        },
        "classes": class_metrics,
        "output_dir": str(captured["save_dir"]),
        "csv_rows": csv_rows,
    }


def main() -> None:
    if not CHECKPOINT.is_file():
        raise FileNotFoundError("Model checkpoint not found: {}".format(CHECKPOINT))
    if not DATASET.is_file():
        raise FileNotFoundError("Dataset YAML not found: {}".format(DATASET))
    for split in SPLITS:
        image_dir = DATASET_ROOT / "images" / split
        label_dir = DATASET_ROOT / "labels" / split
        if not image_dir.is_dir() or not label_dir.is_dir():
            raise FileNotFoundError(
                "Dataset images or labels missing for split {}".format(split)
            )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(CHECKPOINT))
    # Pick one threshold from training and validation before looking at test data.
    confidence = select_confidence(
        model, RUN_NAME, IMAGE_SIZE, output_dir=OUTPUT_DIR
    )

    summaries = {}
    all_rows = []
    for split in SPLITS:
        print(
            "Evaluating {} on {} at imgsz={} conf={:.4f}".format(
                RUN_NAME, split, IMAGE_SIZE, confidence
            ),
            flush=True,
        )
        summary = evaluate_split(model, split, confidence)
        if summary["confidence"] != confidence:
            raise RuntimeError(
                "Split {} was evaluated at confidence {}, expected {}".format(
                    split, summary["confidence"], confidence
                )
            )
        all_rows.extend(summary.pop("csv_rows"))
        summaries[split] = summary

    if any(
        summaries[split]["confidence"] != confidence
        for split in ("val", "test")
    ):
        raise RuntimeError(
            "Validation and test confidence thresholds must be identical"
        )

    plot_validation_test_metrics(summaries, COMPARISON_PLOT_PATH)

    with CSV_PATH.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(all_rows)

    report = {
        "run": RUN_NAME,
        "checkpoint": str(CHECKPOINT),
        "image_size": IMAGE_SIZE,
        "confidence": confidence,
        "threshold_selection": (
            "maximum mean per-class F1 pooled over training and validation; "
            "test split excluded"
        ),
        "splits": summaries,
        "artifacts": {
            "metrics_and_counts_csv": str(CSV_PATH),
            "validation_test_metrics_plot": str(COMPARISON_PLOT_PATH),
            "threshold_selection": str(OUTPUT_DIR / "threshold_selection"),
        },
        "notes": (
            "Each split directory contains Ultralytics confidence/F1, "
            "precision, recall, precision-recall curves, confusion matrices, "
            "and per-image prediction labels."
        ),
    }
    with REPORT_PATH.open("w", encoding="utf-8") as report_file:
        json.dump(report, report_file, indent=2)
        report_file.write("\n")

    print("\nYOLO26l-960-16 results (shared confidence {:.4f})".format(confidence))
    print(
        "{:<8} {:<10} {:>8} {:>9} {:>8} {:>9} {:>9} {:>8}".format(
            "Split", "Class", "GT", "Predicted", "Diff", "Precision", "Recall", "F1"
        )
    )
    for split, summary in summaries.items():
        for class_name in CLASS_NAMES:
            row = summary["classes"][class_name]
            print(
                "{:<8} {:<10} {:>8} {:>9} {:>+8} {:>9.3f} {:>9.3f} {:>8.3f}".format(
                    split,
                    class_name,
                    row["ground_truth"],
                    row["predicted"],
                    row["count_difference"],
                    row["precision"],
                    row["recall"],
                    row["f1"],
                )
            )
        print(
            "{:<8} {:<10} {:>8} {:>9} {:>+8} {:>9.3f} {:>9.3f} {:>8.3f}".format(
                split,
                "all",
                summary["ground_truth_total"],
                summary["predicted_total"],
                summary["count_difference"],
                summary["overall"]["precision"],
                summary["overall"]["recall"],
                summary["overall"]["f1"],
            )
        )
    print("\nResults saved to {}".format(OUTPUT_DIR))
    print("Summary report: {}".format(REPORT_PATH))
    print("Split metrics/counts CSV: {}".format(CSV_PATH))


if __name__ == "__main__":
    main()
