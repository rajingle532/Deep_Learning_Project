"""
Phase 5: Production Training Engine & Experiment Tracking.
Domain-Generalized Deep Learning for Real-World Plant Disease Recognition.

Key Specifications Implemented:
1. Hardware Acceleration:
   - Conditional mixed_float16 precision policy for NVIDIA Tensor Cores (GPU).
   - CPU-safe fallback with float32 for native Windows execution.
2. Optimization Suite:
   - Optimizer: AdamW (learning_rate=1e-4, weight_decay=1e-4).
   - Loss: Categorical Crossentropy with Label Smoothing (alpha=0.1).
   - Weighted Loss: Inverse-frequency class weights loaded from data/processed/class_weights.json.
3. Callbacks:
   - ModelCheckpoint: Saves best weights to models/best_model.keras based on val_loss (mode='min').
   - EarlyStopping: Halts training if val_loss does not improve for 5 epochs (patience=5, mode='min').
   - ReduceLROnPlateau: Halves learning rate on stagnation (patience=3, factor=0.5, mode='min').
   - TensorBoard: Logs scalars and LR to logs/tensorboard/.
   - CSVLogger: Logs epoch-by-epoch loss/accuracy to results/metrics/.
"""

import sys
import os
import json
import argparse
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import tensorflow as tf
import keras

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.taxonomy import NUM_CLASSES
from src.dataset import get_dataset_splits
from src.models import build_model
from src.domain_generalization import MixStyle


def setup_hardware():
    """Detect GPU hardware and configure mixed precision policy."""
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            keras.mixed_precision.set_global_policy("mixed_float16")
            print(f"[1] Hardware Acceleration: {len(gpus)} GPU(s) DETECTED -> 'mixed_float16' policy active!")
            return "gpu_mixed_float16"
        except Exception as e:
            print(f"[1] GPU configuration warning: {e}. Falling back to float32.")
            return "gpu_float32"
    else:
        print("[1] Hardware: Native Windows execution on CPU (using float32 precision).")
        return "cpu_float32"


def plot_training_history(history_dict: dict, save_path: Path, model_name: str):
    """Plot and save training loss and accuracy dynamics."""
    epochs = range(1, len(history_dict["loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Loss Curves
    ax1.plot(epochs, history_dict["loss"], "b-o", label="Train Loss", linewidth=2)
    ax1.plot(epochs, history_dict["val_loss"], "r--s", label="Val Loss", linewidth=2)
    ax1.set_title(f"Loss Convergence ({model_name})", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Epoch", fontsize=10)
    ax1.set_ylabel("Categorical Crossentropy (Label Smoothed)", fontsize=10)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(fontsize=10)

    # Accuracy Curves
    ax2.plot(epochs, history_dict["accuracy"], "b-o", label="Train Acc", linewidth=2)
    ax2.plot(epochs, history_dict["val_accuracy"], "r--s", label="Val Acc", linewidth=2)
    if "val_top_3_accuracy" in history_dict:
        ax2.plot(epochs, history_dict["val_top_3_accuracy"], "g-.^", label="Val Top-3 Acc", linewidth=1.5)
    ax2.set_title(f"Diagnostic Accuracy ({model_name})", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Epoch", fontsize=10)
    ax2.set_ylabel("Accuracy", fontsize=10)
    ax2.set_ylim(0, 1.02)
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(fontsize=10)

    plt.suptitle(f"Phase 5 Training History: {model_name}", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  [SAVED] Training curves plot saved to: {save_path}")


def train(
    model_name: str = "resnet50_mixstyle",
    epochs: int = 10,
    batch_size: int = 32,
    lr: float = 1e-4,
    weight_decay: float = 1e-4,
    label_smoothing: float = 0.1,
    patience_es: int = 5,
    patience_lr: int = 3,
    factor_lr: float = 0.5,
    apply_mixup: bool = True,
    use_class_weights: bool = True,
    steps_per_epoch: int = None,
    validation_steps: int = None,
):
    print("=" * 72)
    print(f"       PHASE 5: TRAINING ENGINE - MODEL: {model_name.upper()}        ")
    print("=" * 72)

    # 1. Hardware acceleration & Precision Policy
    hardware_mode = setup_hardware()

    # 2. Setup output directories
    models_dir = PROJECT_ROOT / "models"
    saved_models_dir = models_dir / "saved_models"
    saved_models_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir = PROJECT_ROOT / "results" / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = PROJECT_ROOT / "results" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logs_dir = PROJECT_ROOT / "logs" / "tensorboard" / f"{model_name}_{timestamp}"
    logs_dir.mkdir(parents=True, exist_ok=True)

    best_model_path = models_dir / "best_model.keras"
    named_model_path = saved_models_dir / f"{model_name}.keras"
    history_csv_path = metrics_dir / f"{model_name}_history.csv"
    curves_plot_path = figures_dir / f"{model_name}_training_curves.png"

    # 3. Load High-Performance Dataset Pipeline
    print(f"\n[2] Loading High-Performance Data Input Pipeline (Batch Size: {batch_size})...")
    use_mixup = apply_mixup if ("mixstyle" in model_name or "dg" in model_name) else False
    train_ds, val_ds, test_ds = get_dataset_splits(
        batch_size=batch_size,
        apply_mixup=use_mixup,
    )
    print(f"    MixUp Batch Regularization: {'ENABLED (alpha=0.2)' if use_mixup else 'DISABLED (ERM Standard)'}")

    # 4. Load Precomputed Class Weights
    weights_dict = None
    if use_class_weights:
        weights_file = PROJECT_ROOT / "data" / "processed" / "class_weights.json"
        if weights_file.exists():
            with open(weights_file, "r") as f:
                raw_weights = json.load(f)
                weights_dict = {int(k): float(v) for k, v in raw_weights.items()}
            print(f"    Class Weights: LOADED ({len(weights_dict)} classes balanced)")

    # 5. Build Model Architecture
    print(f"\n[3] Instantiating Model Architecture: {model_name}...")
    model = build_model(model_name, num_classes=NUM_CLASSES, pretrained=True)
    print(f"    Total Parameters:     {model.count_params():,}")
    print(f"    Trainable Parameters: {sum(np.prod(p.shape) for p in model.trainable_weights):,}")

    # 6. Optimization Suite
    print(f"\n[4] Optimization Suite Configuration:")
    print(f"    Optimizer:       AdamW (lr={lr}, weight_decay={weight_decay})")
    print(f"    Loss Function:   CategoricalCrossentropy (label_smoothing={label_smoothing})")
    print(f"    Primary Metric:  Categorical Accuracy & Top-3 Accuracy")

    optimizer = keras.optimizers.AdamW(learning_rate=lr, weight_decay=weight_decay)
    loss_fn = keras.losses.CategoricalCrossentropy(label_smoothing=label_smoothing)
    top3_acc = tf.keras.metrics.TopKCategoricalAccuracy(k=3, name="top_3_accuracy")

    model.compile(
        optimizer=optimizer,
        loss=loss_fn,
        metrics=["accuracy", top3_acc],
        jit_compile=False,
    )

    # 7. Production Callbacks
    print(f"\n[5] Configuring Production Callbacks:")
    print(f"    - ModelCheckpoint: Monitoring 'val_loss' (mode='min') -> {best_model_path.name}")
    print(f"    - EarlyStopping:   Monitoring 'val_loss' (patience={patience_es}, mode='min')")
    print(f"    - ReduceLROnPlateau: (patience={patience_lr}, factor={factor_lr}, min_lr=1e-6)")
    print(f"    - TensorBoard:     Logging to {logs_dir.name}")
    print(f"    - CSVLogger:       Logging to {history_csv_path.name}")

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath=str(best_model_path),
            monitor="val_loss",
            save_best_only=True,
            mode="min",
            verbose=1,
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=patience_es,
            restore_best_weights=True,
            mode="min",
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=factor_lr,
            patience=patience_lr,
            min_lr=1e-6,
            mode="min",
            verbose=1,
        ),
        keras.callbacks.CSVLogger(str(history_csv_path)),
        keras.callbacks.TensorBoard(log_dir=str(logs_dir)),
    ]

    # 8. Start Model Training
    print(f"\n[6] Commencing Training Loop: {epochs} Epochs...")
    print("=" * 72)

    start_time = datetime.now()
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        class_weight=weights_dict,
        callbacks=callbacks,
        verbose=1,
    )
    elapsed_time = datetime.now() - start_time
    print(f"\n[DONE] Training complete in {elapsed_time}!")

    # 9. Save Named Model Weights
    model.save(str(named_model_path))
    print(f"  [SAVED] Named model checkpoint saved to: {named_model_path}")
    print(f"  [SAVED] Best production model saved to:  {best_model_path}")

    # 10. Plot & Save Convergence Dynamics
    plot_training_history(history.history, curves_plot_path, model_name)

    # 11. Initial In-Domain vs Out-of-Domain Scientific Evaluation
    print("\n" + "=" * 72)
    print("        SCIENTIFIC EVALUATION: DOMAIN GENERALIZATION BENCHMARK     ")
    print("=" * 72)

    print("\nEvaluating on Source Domain Validation Set (PlantVillage In-Domain)...")
    val_results = model.evaluate(val_ds, steps=validation_steps, verbose=0)
    val_acc = val_results[1] * 100.0
    val_top3 = val_results[2] * 100.0

    print("Evaluating on Target Domain Test Set (PlantDoc Unseen Real Field)...")
    test_results = model.evaluate(test_ds, steps=validation_steps, verbose=0)
    test_acc = test_results[1] * 100.0
    test_top3 = test_results[2] * 100.0

    domain_gap = val_acc - test_acc
    transfer_ratio = (test_acc / val_acc) * 100.0 if val_acc > 0 else 0.0

    print("\n" + "=" * 72)
    print(f"  Domain Metric                           Score")
    print("  " + "-" * 68)
    print(f"  In-Domain Validation Accuracy (Lab):    {val_acc:.2f}% (Top-3: {val_top3:.2f}%)")
    print(f"  Out-of-Domain Test Accuracy (Field):    {test_acc:.2f}% (Top-3: {test_top3:.2f}%)")
    print(f"  Domain Generalization Gap (Delta_DG):   {domain_gap:.2f}% (Lower is better)")
    print(f"  Relative Transfer Ratio (tau):          {transfer_ratio:.2f}%")
    print("=" * 72)

    # Save benchmark summary
    summary_data = {
        "model_name": model_name,
        "hardware_mode": hardware_mode,
        "epochs_trained": len(history.history["loss"]),
        "in_domain_val_accuracy": round(val_acc, 2),
        "in_domain_val_top3": round(val_top3, 2),
        "out_of_domain_test_accuracy": round(test_acc, 2),
        "out_of_domain_test_top3": round(test_top3, 2),
        "domain_gap_delta": round(domain_gap, 2),
        "relative_transfer_ratio": round(transfer_ratio, 2),
        "training_time_seconds": round(elapsed_time.total_seconds(), 1),
    }
    summary_path = metrics_dir / f"{model_name}_evaluation_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary_data, f, indent=2)

    print(f"\n[SUCCESS] Benchmark summary saved: {summary_path}")
    return summary_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Domain-Generalized Plant Disease Recognition Training Pipeline")
    parser.add_argument("--model", type=str, default="resnet50_mixstyle", help="Model to train: resnet50_mixstyle, resnet50_erm, mobilenet_v2, custom_cnn")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--label_smoothing", type=float, default=0.1, help="Label smoothing")
    parser.add_argument("--no_mixup", action="store_true", help="Disable MixUp augmentation")
    parser.add_argument("--no_class_weights", action="store_true", help="Disable class weighting")
    parser.add_argument("--steps_per_epoch", type=int, default=None, help="Optional steps per epoch for fast benchmark run")
    parser.add_argument("--val_steps", type=int, default=None, help="Optional validation steps")

    args = parser.parse_args()
    train(
        model_name=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        label_smoothing=args.label_smoothing,
        apply_mixup=not args.no_mixup,
        use_class_weights=not args.no_class_weights,
        steps_per_epoch=args.steps_per_epoch,
        validation_steps=args.val_steps,
    )
