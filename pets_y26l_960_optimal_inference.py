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
CHECKPOINT = ROOT / "weights" / "best.pt"
IMAGE_SIZE = 960
SPLITS = ("train", "val", "test")
DATASET_ROOT = (
    ROOT / ".." / ".." / "datasets" / "pets_create_train_val_test_set"
).resolve()
OUTPUT_DIR = ROOT / "runs" / "pets_inference" / "YOLO26l-960-16_optimal"
REPORT_PATH = OUTPUT_DIR / "inference_summary.json"
CSV_PATH = OUTPUT_DIR / "split_metrics_and_counts.csv"
PER_IMAGE_COUNT_ERRORS_CSV_PATH = OUTPUT_DIR / "per_image_count_errors.csv"
COMPARISON_PLOT_PATH = OUTPUT_DIR / "validation_test_metrics_comparison.png"
COUNT_ERROR_HISTOGRAM_PATH = OUTPUT_DIR / "per_image_count_error_histogram.png"
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


def count_labels_by_image(labels_dir: Path, class_names: dict) -> dict:
    if not labels_dir.is_dir():
        raise FileNotFoundError("Dataset labels not found: {}".format(labels_dir))

    counts_by_image = {}
    for label_path in labels_dir.glob("*.txt"):
        image_counts = Counter()
        with label_path.open(encoding="utf-8") as label_file:
            for line_number, line in enumerate(label_file, start=1):
                fields = line.split()
                if not fields:
                    continue
                try:
                    class_id = int(fields[0])
                except ValueError as error:
                    raise ValueError(
                        "Invalid class ID in {} at line {}".format(
                            label_path, line_number
                        )
                    ) from error
                if class_id not in class_names:
                    raise ValueError(
                        "Unexpected class ID {} in {}".format(class_id, label_path)
                    )
                image_counts[class_names[class_id]] += 1
        counts_by_image[label_path.stem] = image_counts
    return counts_by_image


def calculate_per_image_count_errors(
    split: str,
    expected_image_count: int,
    class_names: dict,
    ground_truth_counts: dict,
    predicted_counts: Counter,
    prediction_labels_dir: Path,
) -> tuple[list[dict], dict]:
    image_dir = DATASET_ROOT / "images" / split
    ground_truth_labels_dir = DATASET_ROOT / "labels" / split
    if not image_dir.is_dir():
        raise FileNotFoundError("Dataset images not found: {}".format(image_dir))
    image_paths = sorted(
        path
        for path in image_dir.iterdir()
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    if not image_paths:
        raise FileNotFoundError("No dataset images found for split {}".format(split))
    if len(image_paths) != expected_image_count:
        raise RuntimeError(
            "Found {} images for {}, expected {}".format(
                len(image_paths), split, expected_image_count
            )
        )

    ground_truth_by_image = count_labels_by_image(
        ground_truth_labels_dir, class_names
    )
    predicted_by_image = count_labels_by_image(prediction_labels_dir, class_names)
    image_stems = {path.stem for path in image_paths}
    for label_source, labels_by_image in (
        ("ground truth", ground_truth_by_image),
        ("predictions", predicted_by_image),
    ):
        unmatched_stems = set(labels_by_image) - image_stems
        if unmatched_stems:
            raise ValueError(
                "{} labels do not match dataset images for {}: {}".format(
                    label_source, split, sorted(unmatched_stems)[:5]
                )
            )

    per_image_rows = []
    absolute_error_totals = Counter()
    count_totals = Counter()
    for image_path in image_paths:
        ground_truth = ground_truth_by_image.get(image_path.stem, Counter())
        predicted = predicted_by_image.get(image_path.stem, Counter())
        row = {"Split": split, "Image": image_path.name}
        total_ground_truth = 0
        total_predicted = 0
        total_absolute_error = 0
        for class_name in CLASS_NAMES:
            gt_count = int(ground_truth[class_name])
            pred_count = int(predicted[class_name])
            signed_error = pred_count - gt_count
            absolute_error = abs(signed_error)
            row["GT {}".format(class_name)] = gt_count
            row["Predicted {}".format(class_name)] = pred_count
            row["Error {}".format(class_name)] = signed_error
            row["Absolute error {}".format(class_name)] = absolute_error
            absolute_error_totals[class_name] += absolute_error
            total_ground_truth += gt_count
            total_predicted += pred_count
            total_absolute_error += absolute_error
        row.update(
            {
                "GT total": total_ground_truth,
                "Predicted total": total_predicted,
                "Total signed error": total_predicted - total_ground_truth,
                "Total absolute error": total_absolute_error,
            }
        )
        per_image_rows.append(row)
        count_totals["ground_truth"] += total_ground_truth
        count_totals["predicted"] += total_predicted
        count_totals["absolute_error"] += total_absolute_error

    if any(
        sum(counts[class_name] for counts in ground_truth_by_image.values())
        != ground_truth_counts[class_name]
        for class_name in CLASS_NAMES
    ):
        raise RuntimeError(
            "Per-image ground-truth counts do not match validator metrics for {}".format(
                split
            )
        )
    if any(
        sum(counts[class_name] for counts in predicted_by_image.values())
        != predicted_counts[class_name]
        for class_name in CLASS_NAMES
    ):
        raise RuntimeError(
            "Per-image predicted counts do not match saved predictions for {}".format(
                split
            )
        )

    image_count = len(image_paths)
    summary = {
        "mean_absolute_count_error": count_totals["absolute_error"] / image_count,
        "mean_absolute_error_by_class": {
            class_name: absolute_error_totals[class_name] / image_count
            for class_name in CLASS_NAMES
        },
        "total_absolute_count_error": count_totals["absolute_error"],
    }
    return per_image_rows, summary


def f1_score(precision: float, recall: float) -> float:
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def plot_per_image_count_error_histograms(
    rows: list[dict], output_path: Path
) -> None:
    splits = ("val", "test")
    colors = {"val": "#5b9bd5", "test": "#70c28a"}
    figure, axes = plt.subplots(1, len(CLASS_NAMES), figsize=(12, 5.5))

    for axis, class_name in zip(axes, CLASS_NAMES):
        errors_by_split = {}
        for split in splits:
            split_rows = [row for row in rows if row["Split"] == split]
            errors = [
                int(row["GT {}".format(class_name)])
                - int(row["Predicted {}".format(class_name)])
                for row in split_rows
            ]
            if not split_rows:
                raise ValueError("No per-image count rows found for {}".format(split))
            errors_by_split[split] = errors

        min_error = min(min(errors) for errors in errors_by_split.values())
        max_error = max(max(errors) for errors in errors_by_split.values())
        bins = [edge - 0.5 for edge in range(min_error, max_error + 2)]
        axis.hist(
            [errors_by_split[split] for split in splits],
            bins=bins,
            label=[
                "{} (n={})".format(split.title(), len(errors_by_split[split]))
                for split in splits
            ],
            color=[colors[split] for split in splits],
            alpha=0.82,
            rwidth=0.88,
            edgecolor="white",
            linewidth=0.6,
        )
        axis.axvline(
            0,
            color="#333333",
            linestyle="--",
            linewidth=1.2,
            label="Exact count",
        )
        annotation_lines = []
        for split in splits:
            errors = errors_by_split[split]
            mae = sum(abs(error) for error in errors) / len(errors)
            annotation_lines.append(
                "{}: MAE {:.2f}".format(split.title(), mae)
            )
        axis.text(
            0.04,
            0.96,
            "\n".join(annotation_lines),
            transform=axis.transAxes,
            va="top",
            ha="left",
            fontsize=9,
            bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "alpha": 0.88},
        )
        axis.set_title(class_name.title(), fontfamily="serif", fontweight="bold")
        axis.set_xlabel(
            "Count error (Ground Truth - Predicted)",
            fontfamily="serif",
        )
        axis.set_ylabel("Number of images", fontfamily="serif")
        axis.set_xticks(range(min_error, max_error + 1))
        axis.grid(axis="y", linestyle="--", alpha=0.28)
        axis.set_axisbelow(True)
        axis.legend(loc="upper right", frameon=True, fontsize=9)

    figure.suptitle(
        "YOLO26l-960-16: Per-image Count Error Distribution",
        fontsize=14,
        fontweight="bold",
        fontfamily="serif",
    )
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_validation_test_metrics(summaries: dict, output_path: Path) -> None:
    # Match the repository's reference chart: validation blue, test green.
    figure, axes = plt.subplots(2, 3, figsize=(14, 7), sharey=True)
    axes = axes.ravel()
    plot_classes = (*CLASS_NAMES, "all")
    display_names = ("Shark", "Skate", "Overall")
    positions = range(len(plot_classes))
    bar_width = 0.34
    colors = {"val": "#5b9bd5", "test": "#70c28a"}
    panel_labels = ("(a)", "(b)", "(c)", "(d)", "(e)")
    font = {"family": "serif"}

    for panel_index, (axis, (metric_title, metric_key)) in enumerate(
        zip(axes, COMPARISON_METRICS)
    ):
        for split_index, split in enumerate(("val", "test")):
            values = [
                (
                    summaries[split]["overall"][
                        {"ap50": "map50", "ap50_95": "map50_95"}.get(
                            metric_key, metric_key
                        )
                    ]
                    if class_name == "all"
                    else summaries[split]["classes"][class_name][metric_key]
                )
                for class_name in plot_classes
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
            display_names,
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
    per_image_count_rows, per_image_count_summary = calculate_per_image_count_errors(
        split,
        captured["images"],
        class_names,
        ground_truth_counts,
        predicted_counts,
        captured["save_dir"] / "labels",
    )
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
        "per_image_count_error": per_image_count_summary,
        "output_dir": str(captured["save_dir"]),
        "csv_rows": csv_rows,
        "per_image_count_rows": per_image_count_rows,
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
    all_per_image_count_rows = []
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
        all_per_image_count_rows.extend(summary.pop("per_image_count_rows"))
        summaries[split] = summary

    if any(
        summaries[split]["confidence"] != confidence
        for split in ("val", "test")
    ):
        raise RuntimeError(
            "Validation and test confidence thresholds must be identical"
        )

    plot_validation_test_metrics(summaries, COMPARISON_PLOT_PATH)
    plot_per_image_count_error_histograms(
        all_per_image_count_rows, COUNT_ERROR_HISTOGRAM_PATH
    )

    with CSV_PATH.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(all_rows)

    per_image_fields = [
        "Split",
        "Image",
        *[
            field
            for class_name in CLASS_NAMES
            for field in (
                "GT {}".format(class_name),
                "Predicted {}".format(class_name),
                "Error {}".format(class_name),
                "Absolute error {}".format(class_name),
            )
        ],
        "GT total",
        "Predicted total",
        "Total signed error",
        "Total absolute error",
    ]
    with PER_IMAGE_COUNT_ERRORS_CSV_PATH.open(
        "w", newline="", encoding="utf-8"
    ) as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=per_image_fields)
        writer.writeheader()
        writer.writerows(all_per_image_count_rows)

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
            "per_image_count_errors_csv": str(PER_IMAGE_COUNT_ERRORS_CSV_PATH),
            "validation_test_metrics_plot": str(COMPARISON_PLOT_PATH),
            "per_image_count_error_histogram": str(COUNT_ERROR_HISTOGRAM_PATH),
            "threshold_selection": str(OUTPUT_DIR / "threshold_selection"),
        },
        "notes": (
            "Each split directory contains Ultralytics confidence/F1, "
            "precision, recall, precision-recall curves, confusion matrices, "
            "and per-image prediction labels."
            "Per-image absolute count errors sum class-specific errors so "
            "shark and skate errors cannot cancel each other."
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
        count_error = summary["per_image_count_error"]
        print(
            "  Mean absolute count error per image: {:.2f} "
            "(shark {:.2f}, skate {:.2f})".format(
                count_error["mean_absolute_count_error"],
                count_error["mean_absolute_error_by_class"]["shark"],
                count_error["mean_absolute_error_by_class"]["skate"],
            )
        )
    print("\nResults saved to {}".format(OUTPUT_DIR))
    print("Summary report: {}".format(REPORT_PATH))
    print("Split metrics/counts CSV: {}".format(CSV_PATH))
    print("Per-image count errors CSV: {}".format(PER_IMAGE_COUNT_ERRORS_CSV_PATH))


if __name__ == "__main__":
    main()
