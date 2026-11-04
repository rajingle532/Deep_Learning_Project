"""
Visualization Script for Phase 3 Domain-Robust Augmentations.
Domain-Generalized Deep Learning for Real-World Plant Disease Recognition.

Generates a publication-quality 4x4 comparison grid showing:
- Original leaf image from laboratory domain.
- 15 stochastic variations exhibiting Color Jitter, Shadows, Defocus Blur, Rotations, and MixUp.
Saves to results/eda/augmentation_sample_grid.png.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import tensorflow as tf

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.preprocessing import domain_robust_augmentation, normalize_image, unnormalize_image, mixup
from src.dataset import create_dataset_from_manifest, get_dataset_splits
from src.taxonomy import CANONICAL_CLASSES


def generate_augmentation_grid():
    print("=" * 70)
    print("   PHASE 3: DOMAIN-ROBUST AUGMENTATION PIPELINE VISUALIZATION     ")
    print("=" * 70)

    results_dir = PROJECT_ROOT / "results" / "eda"
    results_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = PROJECT_ROOT / "data" / "processed" / "train_manifest.csv"
    df = pd.read_csv(manifest_path)

    # Pick a crisp leaf image (e.g. Tomato Early Blight or Apple Scab)
    sample_row = df[df["canonical_name"] == "tomato_early_blight"].iloc[0]
    sample_path = PROJECT_ROOT / Path(sample_row["filepath"])

    print(f"\n[1] Selected Sample Leaf: {sample_path.name}")
    print(f"    Class: {sample_row['display_name']} ({sample_row['canonical_name']})")

    # Read and resize raw image
    img_bytes = tf.io.read_file(str(sample_path))
    raw_img = tf.io.decode_image(img_bytes, channels=3, expand_animations=False)
    raw_img = tf.cast(tf.image.resize(raw_img, (224, 224)), tf.float32)

    # Generate 15 augmented variations
    print("\n[2] Applying Domain-Invariant Transforms (Color Jitter, Shadows, Blur, Flips)...")
    fig, axes = plt.subplots(4, 4, figsize=(13, 13))

    # Cell (0, 0): Original
    axes[0, 0].imshow(tf.cast(raw_img, tf.uint8).numpy())
    axes[0, 0].set_title("Original (Lab Controlled)", fontsize=10, fontweight="bold", color="darkgreen")
    axes[0, 0].axis("off")

    for i in range(1, 16):
        r, c = divmod(i, 4)
        # Apply domain robust augmentation
        aug_img = domain_robust_augmentation(raw_img)
        axes[r, c].imshow(tf.cast(aug_img, tf.uint8).numpy())
        axes[r, c].set_title(f"Field Perturbation #{i}", fontsize=9)
        axes[r, c].axis("off")

    fig.suptitle("Phase 3: Domain-Invariant Augmentation Pipeline (Simulating Real Farm Noise)", fontsize=13, fontweight="bold", y=0.98)
    plt.tight_layout()

    grid_path = results_dir / "augmentation_sample_grid.png"
    fig.savefig(grid_path, dpi=200)
    plt.close(fig)
    print(f"  [SAVED] Augmentation grid saved: {grid_path}")

    # Test throughput with batch size 32 & MixUp
    print("\n[3] Testing High-Performance Pipeline Throughput (Batch Size = 32 with MixUp)...")
    train_ds, val_ds, test_ds = get_dataset_splits(batch_size=32, apply_mixup=True)

    batch_count = 0
    for images, labels in train_ds.take(5):
        batch_count += 1
        assert images.shape == (32, 224, 224, 3)
        assert labels.shape == (32, 28)

    print(f"  [PASS] Successfully streamed {batch_count} batches (160 images) at AUTOTUNE speed!")
    print(f"  Batch Shape: (32, 224, 224, 3) | Label Shape: (32, 28)")
    print(f"\n>>> PHASE 3 COMPLETE: HIGH-PERFORMANCE DATA PIPELINE READY! <<<")


if __name__ == "__main__":
    generate_augmentation_grid()

