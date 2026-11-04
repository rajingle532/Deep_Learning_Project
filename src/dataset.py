"""
Manifest-Driven High-Performance Dataset Pipeline.
Domain-Generalized Deep Learning for Real-World Plant Disease Recognition.

Loads pre-split CSV manifests (train_manifest.csv, val_manifest.csv, test_manifest.csv)
into optimized tf.data.Dataset pipelines with:
- Multithreaded I/O & Decoding
- Deterministic 28-class one-hot label encoding
- ImageNet normalization & domain-robust augmentations
- Memory caching & AUTOTUNE prefetching
"""

import json
from pathlib import Path
from typing import Tuple, Optional
import pandas as pd
import tensorflow as tf

from src.taxonomy import NUM_CLASSES
from src.preprocessing import normalize_image, domain_robust_augmentation, mixup

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def parse_image_and_label(
    filepath: tf.Tensor,
    label_idx: tf.Tensor,
    image_size: Tuple[int, int] = (224, 224),
    is_training: bool = False,
    normalization: str = "imagenet",
) -> Tuple[tf.Tensor, tf.Tensor]:
    """Reads image from disk, decodes, resizes, normalizes and one-hot encodes label."""
    # Read file
    img_bytes = tf.io.read_file(filepath)
    # Decode image to 3-channel RGB tensor
    img = tf.io.decode_image(img_bytes, channels=3, expand_animations=False)
    img.set_shape([None, None, 3])

    # Resize to uniform dimension
    img = tf.image.resize(img, image_size, method="bilinear")

    # Domain augmentations if training
    if is_training:
        img = domain_robust_augmentation(img)

    # Normalize with ImageNet mean and std
    img = normalize_image(img, normalization=normalization)

    # One-hot encode label across 28 canonical classes
    one_hot = tf.one_hot(label_idx, depth=NUM_CLASSES)
    return img, one_hot


def create_dataset_from_manifest(
    manifest_csv: str,
    image_size: Tuple[int, int] = (224, 224),
    batch_size: int = 32,
    is_training: bool = False,
    apply_mixup: bool = False,
    mixup_alpha: float = 0.2,
    shuffle_buffer: int = 2048,
    seed: int = 42,
) -> tf.data.Dataset:
    """
    Construct an end-to-end tf.data.Dataset from a CSV manifest.

    Args:
        manifest_csv: Path to manifest CSV (train_manifest.csv, etc.)
        image_size: Target (height, width) dimensions.
        batch_size: Batch size.
        is_training: Whether to shuffle and apply augmentations.
        apply_mixup: Whether to apply batch-level MixUp interpolation.
        mixup_alpha: MixUp beta distribution parameter.
        shuffle_buffer: Buffer size for training shuffle.
        seed: Random seed.

    Returns:
        tf.data.Dataset yielding (batch_images, batch_one_hot_labels).
    """
    manifest_path = Path(manifest_csv)
    if not manifest_path.is_absolute():
        manifest_path = PROJECT_ROOT / manifest_path

    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    df = pd.read_csv(manifest_path)
    
    # Resolve absolute paths from relative paths stored in manifest
    abs_paths = [str(PROJECT_ROOT / Path(p)) for p in df["filepath"]]
    class_ids = df["class_id"].astype("int32").tolist()

    # Create tensor slices
    ds = tf.data.Dataset.from_tensor_slices((abs_paths, class_ids))

    if is_training:
        ds = ds.shuffle(buffer_size=min(len(df), shuffle_buffer), seed=seed, reshuffle_each_iteration=True)

    # Parallel parsing
    ds = ds.map(
        lambda p, l: parse_image_and_label(p, l, image_size=image_size, is_training=is_training),
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    # Batching
    ds = ds.batch(batch_size, drop_remainder=is_training)

    # Batch-level MixUp augmentation
    if is_training and apply_mixup:
        ds = ds.map(lambda x, y: mixup(x, y, alpha=mixup_alpha), num_parallel_calls=tf.data.AUTOTUNE)

    # Prefetch
    ds = ds.prefetch(buffer_size=tf.data.AUTOTUNE)
    return ds


def get_dataset_splits(
    image_size: Tuple[int, int] = (224, 224),
    batch_size: int = 32,
    apply_mixup: bool = False,
) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
    """Convenience function returning (train_ds, val_ds, test_ds)."""
    processed_dir = PROJECT_ROOT / "data" / "processed"
    
    train_ds = create_dataset_from_manifest(
        str(processed_dir / "train_manifest.csv"),
        image_size=image_size,
        batch_size=batch_size,
        is_training=True,
        apply_mixup=apply_mixup,
    )
    val_ds = create_dataset_from_manifest(
        str(processed_dir / "val_manifest.csv"),
        image_size=image_size,
        batch_size=batch_size,
        is_training=False,
        apply_mixup=False,
    )
    test_ds = create_dataset_from_manifest(
        str(processed_dir / "test_manifest.csv"),
        image_size=image_size,
        batch_size=batch_size,
        is_training=False,
        apply_mixup=False,
    )
    return train_ds, val_ds, test_ds
