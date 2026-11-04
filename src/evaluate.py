"""
Phase 6: Comprehensive Scientific Evaluation & Benchmarking Engine.
Domain-Generalized Deep Learning for Real-World Plant Disease Recognition.

Computes:
1. Top-1 & Top-3 Diagnostic Accuracy on unseen held-out Target Domain (PlantDoc).
2. Per-class Precision, Recall, and Macro/Weighted F1-Scores.
3. High-resolution 28x28 Normalized Confusion Matrix Heatmap.
4. Domain Generalization Gap (Delta_DG = Acc_Source - Acc_Target).
5. Exports reports to results/metrics/ and results/figures/.
"""

import sys
import json
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import tensorflow as tf
import keras
from sklearn.metrics import classification_report, confusion_matrix

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from src.taxonomy import CANONICAL_CLASSES, CANONICAL_DISPLAY_NAMES, NUM_CLASSES
from src.domain_generalization import MixStyle
from src.dataset import create_dataset_from_manifest


def evaluate_benchmark(
    model_path: str = "models/best_model.keras",
    batch_size: int = 32,
    max_val_batches: int = None,
    max_test_batches: int = None,
):
    print("=" * 72)
    print("      PHASE 6: SCIENTIFIC BENCHMARK & OUT-OF-DOMAIN EVALUATION         ")
    print("=" * 72)

    model_file = Path(model_path)
    if not model_file.is_absolute():
        model_file = PROJECT_ROOT / model_file

    if not model_file.exists():
        raise FileNotFoundError(f"Model file not found: {model_file}")

    processed_dir = PROJECT_ROOT / "data" / "processed"
    val_csv = processed_dir / "val_manifest.csv"
    test_csv = processed_dir / "test_manifest.csv"
    figures_dir = PROJECT_ROOT / "results" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir = PROJECT_ROOT / "results" / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load Trained Model
    print(f"\n[1] Loading Trained Weights from: {model_file.name}...")
    model = keras.models.load_model(
        str(model_file),
        custom_objects={"MixStyle": MixStyle},
    )
    print(f"    Model Architecture: {model.name} ({model.count_params():,} parameters)")

    # 2. Load Datasets
    print("\n[2] Loading Evaluation Datasets...")
    val_ds = create_dataset_from_manifest(str(val_csv), batch_size=batch_size, is_training=False)
    test_ds = create_dataset_from_manifest(str(test_csv), batch_size=batch_size, is_training=False)

    if max_val_batches is not None:
        val_ds = val_ds.take(max_val_batches)
    if max_test_batches is not None:
        test_ds = test_ds.take(max_test_batches)

    # 3. Evaluate In-Domain (PlantVillage Val)
    print("\n[3] Evaluating In-Domain Performance (PlantVillage Source Validation)...")
    val_results = model.evaluate(val_ds, verbose=1)
    val_loss = float(val_results[0])
    val_acc = float(val_results[1]) * 100.0

    # 4. Evaluate Out-of-Domain (PlantDoc Held-out Test)
    print("\n[4] Evaluating Out-of-Domain Performance (PlantDoc Unseen Farm Test)...")
    all_preds = []
    all_targets = []

    for img_batch, lbl_batch in test_ds:
        preds = model(img_batch, training=False)
        all_preds.append(preds.numpy())
        all_targets.append(lbl_batch.numpy())

    y_pred_probs = np.concatenate(all_preds, axis=0)
    y_true_onehot = np.concatenate(all_targets, axis=0)

    y_pred = np.argmax(y_pred_probs, axis=-1)
    y_true = np.argmax(y_true_onehot, axis=-1)

    # Top-1 Accuracy
    test_acc = float(np.mean(y_pred == y_true)) * 100.0

    # Top-3 Accuracy
    top3_indices = np.argsort(y_pred_probs, axis=-1)[:, -3:]
    top3_correct = [y_true[i] in top3_indices[i] for i in range(len(y_true))]
    test_top3 = float(np.mean(top3_correct)) * 100.0

    # Domain Generalization Gap
    domain_gap = val_acc - test_acc
    transfer_ratio = (test_acc / val_acc) * 100.0 if val_acc > 0 else 0.0

    print("\n" + "=" * 72)
    print("                    DOMAIN GENERALIZATION BENCHMARK REPORT             ")
    print("=" * 72)
    print(f"  Source Domain Validation Accuracy (Lab):  {val_acc:.2f}% (Loss: {val_loss:.4f})")
    print(f"  Target Domain Test Accuracy (Field):      {test_acc:.2f}% (Top-3: {test_top3:.2f}%)")
    print(f"  Domain Generalization Gap (Delta_DG):     {domain_gap:.2f}% (Lower is better)")
    print(f"  Relative Transfer Ratio (tau):            {transfer_ratio:.2f}%")
    print("=" * 72)

    # 5. Compute Classification Report
    display_names = [CANONICAL_DISPLAY_NAMES[c] for c in CANONICAL_CLASSES]
    present_labels = sorted(list(set(y_true)))
    target_names_present = [display_names[i] for i in present_labels]

    report_dict = classification_report(
        y_true,
        y_pred,
        labels=present_labels,
        target_names=target_names_present,
        output_dict=True,
        zero_division=0,
    )

    report_path = metrics_dir / "target_domain_classification_report.json"
    with open(report_path, "w") as f:
        json.dump(report_dict, f, indent=2)
    print(f"\n[5] Classification report saved: {report_path}")

    # 6. Generate 28x28 Normalized Confusion Matrix Heatmap
    print("\n[6] Generating 28x28 Confusion Matrix Heatmap...")
    cm = confusion_matrix(y_true, y_pred, labels=range(NUM_CLASSES))
    # Row normalize
    cm_norm = cm.astype("float") / (cm.sum(axis=1)[:, np.newaxis] + 1e-9)

    fig, ax = plt.subplots(figsize=(18, 15))
    sns.heatmap(
        cm_norm,
        annot=False,
        cmap="Blues",
        xticklabels=display_names,
        yticklabels=display_names,
        ax=ax,
        cbar_kws={"label": "Normalized True Positive Rate"},
    )
    ax.set_title("Out-of-Domain Confusion Matrix: PlantDoc In-The-Wild Evaluation", fontsize=15, fontweight="bold", pad=20)
    ax.set_xlabel("Predicted Disease Class", fontsize=12, fontweight="bold")
    ax.set_ylabel("True Ground Truth Class", fontsize=12, fontweight="bold")
    plt.xticks(rotation=90, fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()

    cm_path = figures_dir / "confusion_matrix.png"
    fig.savefig(cm_path, dpi=200)
    plt.close(fig)
    print(f"  [SAVED] Confusion matrix saved: {cm_path}")

    # 7. Export Master Benchmark Summary
    summary = {
        "model_file": model_file.name,
        "total_test_samples": len(y_true),
        "source_in_domain_accuracy": round(val_acc, 2),
        "target_out_of_domain_accuracy": round(test_acc, 2),
        "target_top3_accuracy": round(test_top3, 2),
        "domain_generalization_gap": round(domain_gap, 2),
        "macro_precision": round(report_dict["macro avg"]["precision"] * 100.0, 2),
        "macro_recall": round(report_dict["macro avg"]["recall"] * 100.0, 2),
        "macro_f1": round(report_dict["macro avg"]["f1-score"] * 100.0, 2),
    }

    summary_file = metrics_dir / "benchmark_evaluation_summary.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  [SAVED] Benchmark summary saved: {summary_file}")
    print("\n>>> PHASE 6 EVALUATION COMPLETE! <<<")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Domain-Generalized Plant Disease Recognition Model")
    parser.add_argument("--model", type=str, default="models/best_model.keras", help="Path to .keras model file")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--max_val_batches", type=int, default=None, help="Optional max validation batches")
    parser.add_argument("--max_test_batches", type=int, default=None, help="Optional max test batches")
    args = parser.parse_args()
    evaluate_benchmark(
        model_path=args.model,
        batch_size=args.batch_size,
        max_val_batches=args.max_val_batches,
        max_test_batches=args.max_test_batches,
    )

