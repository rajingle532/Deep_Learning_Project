"""
Comprehensive Scientific Prediction Validation & Reliability Benchmark Pipeline.
AgriGuard AI: Domain-Generalized Deep Learning for Real-World Foliar Pathology.
"""

import os
import sys
import time
import json
import csv
from pathlib import Path
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import tensorflow as tf
import keras
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    brier_score_loss,
)

from src.taxonomy import CANONICAL_CLASSES, CANONICAL_DISPLAY_NAMES, NUM_CLASSES
from src.domain_generalization import MixStyle
from src.dataset import create_dataset_from_manifest

# -------------------------------------------------------------
# 1. SCIENTIFIC METRIC FUNCTIONS
# -------------------------------------------------------------
def compute_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10):
    """Computes Expected Calibration Error (Naeini et al., 2015)."""
    confs = np.max(probs, axis=1)
    preds = np.argmax(probs, axis=1)
    accuracies = (preds == labels).astype(float)

    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    bin_stats = []

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        mask = (confs > bin_lower) & (confs <= bin_upper) if i > 0 else (confs >= bin_lower) & (confs <= bin_upper)
        bin_count = np.sum(mask)

        if bin_count > 0:
            bin_acc = float(np.mean(accuracies[mask]))
            bin_conf = float(np.mean(confs[mask]))
            ece += (bin_count / len(confs)) * abs(bin_acc - bin_conf)
            bin_stats.append({
                "bin_idx": i,
                "range": [round(bin_lower, 2), round(bin_upper, 2)],
                "count": int(bin_count),
                "accuracy": round(bin_acc, 4),
                "confidence": round(bin_conf, 4),
            })
        else:
            bin_stats.append({
                "bin_idx": i,
                "range": [round(bin_lower, 2), round(bin_upper, 2)],
                "count": 0,
                "accuracy": 0.0,
                "confidence": 0.0,
            })

    return float(ece), bin_stats


def compute_multiclass_brier(probs: np.ndarray, labels: np.ndarray):
    """Computes multiclass Brier score: mean squared difference from one-hot ground truth."""
    n_samples, n_classes = probs.shape
    one_hot = np.zeros_like(probs)
    one_hot[np.arange(n_samples), labels] = 1.0
    return float(np.mean(np.sum((probs - one_hot) ** 2, axis=1)))


# -------------------------------------------------------------
# 2. EVALUATION EXECUTION ENGINE WITH CACHING
# -------------------------------------------------------------
def evaluate_domain(model, manifest_csv: str, cache_name: str, domain_name="Target Field (PlantDoc)", max_batches=None):
    cache_dir = PROJECT_ROOT / "results" / "metrics"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"{cache_name}_cache.npz"

    if cache_file.exists():
        print(f"\nLoading cached inference predictions from {cache_file}...", flush=True)
        cached_data = np.load(cache_file)
        probs = cached_data["probs"]
        labels = cached_data["labels"]
        avg_latency_ms = float(cached_data.get("latency_ms", 80.0))
    else:
        print(f"\nEvaluating {domain_name} from {manifest_csv}...", flush=True)
        batch_size = 64
        ds = create_dataset_from_manifest(
            manifest_csv=manifest_csv,
            batch_size=batch_size,
            is_training=False,
        )

        @tf.function
        def predict_step(x):
            return model(x, training=False)

        all_probs = []
        all_labels = []
        latencies = []

        batch_idx = 0
        t_start = time.time()
        for batch_x, batch_y_one_hot in ds:
            t0 = time.time()
            preds = predict_step(batch_x).numpy()
            latencies.append((time.time() - t0) / len(batch_x))

            all_probs.append(preds)
            all_labels.append(np.argmax(batch_y_one_hot.numpy(), axis=1))

            batch_idx += 1
            if batch_idx % 10 == 0:
                print(f"  Processed {batch_idx * batch_size} samples ({time.time() - t_start:.1f}s)...", flush=True)

            if max_batches and batch_idx >= max_batches:
                break

        probs = np.vstack(all_probs)
        labels = np.concatenate(all_labels)
        avg_latency_ms = float(np.mean(latencies) * 1000.0)

        # Cache predictions to avoid recomputing
        np.savez_compressed(cache_file, probs=probs, labels=labels, latency_ms=avg_latency_ms)
        print(f"Inference complete: {len(labels):,} samples in {time.time() - t_start:.2f}s. Saved cache to {cache_file}", flush=True)

    preds = np.argmax(probs, axis=1)
    confs = np.max(probs, axis=1)

    # Top-3 Accuracy
    top3_preds = np.argsort(probs, axis=1)[:, -3:]
    top3_correct = np.any(top3_preds == labels[:, None], axis=1)
    top3_acc = float(np.mean(top3_correct))

    # Standard metrics
    acc = float(accuracy_score(labels, preds))
    bal_acc = float(balanced_accuracy_score(labels, preds))
    prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(labels, preds, average="macro", zero_division=0)
    prec_weighted, rec_weighted, f1_weighted, _ = precision_recall_fscore_support(labels, preds, average="weighted", zero_division=0)

    # Uncertainty & Rejection Metrics
    entropies = -np.sum(probs * np.log(np.maximum(probs, 1e-12)), axis=1)
    # Safe rejection rule: confidence < 28% OR entropy > 2.35
    rejected_mask = (confs < 0.28) | (entropies > 2.35)
    rejection_rate = float(np.mean(rejected_mask))
    accepted_mask = ~rejected_mask

    accepted_acc = float(accuracy_score(labels[accepted_mask], preds[accepted_mask])) if np.sum(accepted_mask) > 0 else 0.0

    # Calibration Metrics
    ece, bin_stats = compute_ece(probs, labels)
    brier = compute_multiclass_brier(probs, labels)

    # Per-class breakdown
    per_prec, per_rec, per_f1, per_supp = precision_recall_fscore_support(labels, preds, average=None, zero_division=0)
    per_class_metrics = []
    for cid in range(NUM_CLASSES):
        per_class_metrics.append({
            "class_id": cid,
            "canonical_name": CANONICAL_CLASSES[cid],
            "display_name": CANONICAL_DISPLAY_NAMES[CANONICAL_CLASSES[cid]],
            "support": int(per_supp[cid]) if cid < len(per_supp) else 0,
            "precision": round(float(per_prec[cid]), 4) if cid < len(per_prec) else 0.0,
            "recall": round(float(per_rec[cid]), 4) if cid < len(per_rec) else 0.0,
            "f1_score": round(float(per_f1[cid]), 4) if cid < len(per_f1) else 0.0,
        })

    worst_classes = sorted(per_class_metrics, key=lambda x: x["f1_score"])
    cm = confusion_matrix(labels, preds, labels=list(range(NUM_CLASSES)))

    results = {
        "domain_name": domain_name,
        "sample_count": len(labels),
        "top1_accuracy": round(acc, 4),
        "top3_accuracy": round(top3_acc, 4),
        "balanced_accuracy": round(bal_acc, 4),
        "macro_precision": round(float(prec_macro), 4),
        "macro_recall": round(float(rec_macro), 4),
        "macro_f1": round(float(f1_macro), 4),
        "weighted_f1": round(float(f1_weighted), 4),
        "expected_calibration_error": round(ece, 4),
        "brier_score": round(brier, 4),
        "avg_predictive_entropy": round(float(np.mean(entropies)), 4),
        "uncertainty_rejection_rate": round(rejection_rate, 4),
        "selective_accuracy_accepted": round(accepted_acc, 4),
        "avg_latency_ms_per_image": round(avg_latency_ms, 2),
        "reliability_bins": bin_stats,
        "worst_5_performing_classes": worst_classes[:5],
        "per_class_metrics": per_class_metrics,
        "confusion_matrix": cm.tolist(),
    }

    return results, cm


# -------------------------------------------------------------
# 3. MAIN RUNNER
# -------------------------------------------------------------
def main():
    print("======================================================================", flush=True)
    print("      AGRIGUARD AI: COMPREHENSIVE PREDICTION VALIDATION SUITE        ", flush=True)
    print("======================================================================", flush=True)

    model_path = PROJECT_ROOT / "models" / "best_model.keras"
    assert model_path.exists(), f"Model not found at {model_path}"
    print(f"Loading checkpoint: {model_path}...", flush=True)
    model = keras.models.load_model(str(model_path), compile=False, custom_objects={"MixStyle": MixStyle})
    print(f"Model successfully loaded. Parameter count: {model.count_params():,}", flush=True)

    test_manifest = str(PROJECT_ROOT / "data" / "processed" / "test_manifest.csv")
    val_manifest = str(PROJECT_ROOT / "data" / "processed" / "val_manifest.csv")

    # 1. Target Domain Field Evaluation (Held-out PlantDoc, ALL 2,572 images)
    test_results, cm_test = evaluate_domain(model, test_manifest, cache_name="target_plantdoc", domain_name="Target Field (PlantDoc)")

    # 2. Source In-Domain Evaluation (PlantVillage, 16 batches = 1,024 images)
    val_results, cm_val = evaluate_domain(model, val_manifest, cache_name="source_plantvillage", domain_name="Source Laboratory (PlantVillage)", max_batches=16)

    # Print Comparison Table
    print("\n" + "=" * 70, flush=True)
    print("       SCIENTIFIC VALIDATION SUMMARY: IN-DOMAIN VS. OUT-OF-DOMAIN      ", flush=True)
    print("=" * 70, flush=True)
    print(f"{'Metric':<30} | {'Source (PlantVillage)':<20} | {'Target (PlantDoc)':<18}", flush=True)
    print("-" * 70, flush=True)
    print(f"{'Top-1 Accuracy':<30} | {val_results['top1_accuracy']*100:>18.2f}% | {test_results['top1_accuracy']*100:>16.2f}%", flush=True)
    print(f"{'Top-3 Accuracy':<30} | {val_results['top3_accuracy']*100:>18.2f}% | {test_results['top3_accuracy']*100:>16.2f}%", flush=True)
    print(f"{'Balanced Accuracy':<30} | {val_results['balanced_accuracy']*100:>18.2f}% | {test_results['balanced_accuracy']*100:>16.2f}%", flush=True)
    print(f"{'Macro F1-Score':<30} | {val_results['macro_f1']*100:>18.2f}% | {test_results['macro_f1']*100:>16.2f}%", flush=True)
    print(f"{'Weighted F1-Score':<30} | {val_results['weighted_f1']*100:>18.2f}% | {test_results['weighted_f1']*100:>16.2f}%", flush=True)
    print(f"{'Expected Calib Error (ECE)':<30} | {val_results['expected_calibration_error']:>19.4f} | {test_results['expected_calibration_error']:>17.4f}", flush=True)
    print(f"{'Brier Score (Lower=Better)':<30} | {val_results['brier_score']:>19.4f} | {test_results['brier_score']:>17.4f}", flush=True)
    print(f"{'Uncertainty Rejection Rate':<30} | {val_results['uncertainty_rejection_rate']*100:>18.2f}% | {test_results['uncertainty_rejection_rate']*100:>16.2f}%", flush=True)
    print(f"{'Selective Accepted Accuracy':<30} | {val_results['selective_accuracy_accepted']*100:>18.2f}% | {test_results['selective_accuracy_accepted']*100:>16.2f}%", flush=True)
    print(f"{'Inference Latency (batch)':<30} | {val_results['avg_latency_ms_per_image']:>17.2f}ms | {test_results['avg_latency_ms_per_image']:>15.2f}ms", flush=True)
    print("=" * 70, flush=True)

    # Plot Confusion Matrix for Target Domain
    fig_dir = PROJECT_ROOT / "results" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    cm_path = fig_dir / "target_domain_confusion_matrix.png"

    plt.figure(figsize=(14, 12))
    # Normalized confusion matrix
    cm_norm = cm_test.astype('float') / np.maximum(cm_test.sum(axis=1)[:, np.newaxis], 1.0)
    sns.heatmap(cm_norm, cmap="viridis", cbar=True, square=True)
    plt.title("Target Domain (PlantDoc) Normalized Confusion Matrix\n28 Canonical Classes Under Real-World Covariate Shift", fontsize=13, pad=15)
    plt.xlabel("Predicted Class Index (0–27)", fontsize=11)
    plt.ylabel("Ground Truth Class Index (0–27)", fontsize=11)
    plt.tight_layout()
    plt.savefig(cm_path, dpi=200)
    plt.close()
    print(f"\nSaved Target Domain Confusion Matrix Heatmap to: {cm_path}", flush=True)

    # Save Complete Validation Results JSON
    summary_report = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "checkpoint": str(model_path.relative_to(PROJECT_ROOT)),
        "source_evaluation": val_results,
        "target_evaluation": test_results,
    }

    out_json = PROJECT_ROOT / "results" / "metrics" / "comprehensive_prediction_validation.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary_report, f, indent=2)

    print(f"Saved Complete Empirical Validation Report to: {out_json}", flush=True)
    print("\n======================================================================", flush=True)
    print(">>> PREDICTION VALIDATION COMPLETE: EMPIRICAL EVIDENCE GENERATED! <<<", flush=True)
    print("======================================================================", flush=True)


if __name__ == "__main__":
    main()
