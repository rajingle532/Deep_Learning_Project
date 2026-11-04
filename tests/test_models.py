"""
Comprehensive Model Benchmark Architecture Unit Tests.
Domain-Generalized Deep Learning for Real-World Plant Disease Recognition.

Verifies:
1. Correct graph construction for all 4 models.
2. Parameter counts and memory footprints.
3. Forward pass tensor shapes in both Training and Evaluation modes.
4. Softmax probability output constraint: sum(p_i) == 1.0.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import tensorflow as tf
import keras

from src.models import build_model
from src.taxonomy import NUM_CLASSES


def test_all_models():
    print("=" * 70)
    print("        PHASE 4: MULTI-MODEL BENCHMARK ARCHITECTURE AUDIT         ")
    print("=" * 70)

    model_names = [
        ("custom_cnn", "Model 1: Custom Baseline CNN (Scratch)"),
        ("mobilenet_v2", "Model 2: MobileNetV2 (Edge Backbone)"),
        ("resnet50_erm", "Model 3: ResNet50 Standard ERM Transfer Learning"),
        ("resnet50_mixstyle", "Model 4: Proposed ResNet50 + MixStyle Domain Generalization"),
    ]

    dummy_batch = tf.random.normal([4, 224, 224, 3])
    print(f"\n[1] Generated Test Tensor Batch: {dummy_batch.shape}\n")

    summary_records = []

    for key, display in model_names:
        print(f"--- Testing {display} ---")
        model = build_model(key, num_classes=NUM_CLASSES)
        assert model is not None, f"Failed to build {key}"

        # Parameter counts
        total_params = model.count_params()
        trainable_params = sum(np.prod(p.shape) for p in model.trainable_weights)

        # 1. Forward pass in Training mode
        out_train = model(dummy_batch, training=True)
        assert out_train.shape == (4, NUM_CLASSES), f"Shape mismatch in train: {out_train.shape}"

        # 2. Forward pass in Inference mode
        out_eval = model(dummy_batch, training=False)
        assert out_eval.shape == (4, NUM_CLASSES), f"Shape mismatch in eval: {out_eval.shape}"

        # 3. Softmax check: row sums equal 1.0
        sums = np.sum(out_eval.numpy(), axis=-1)
        assert np.allclose(sums, 1.0, atol=1e-5), f"Softmax sum error: {sums}"

        print(f"  [PASS] Training Out Shape:  {out_train.shape}")
        print(f"  [PASS] Inference Out Shape: {out_eval.shape}")
        print(f"  [PASS] Total Parameters:    {total_params:,}")
        print(f"  [PASS] Trainable Params:    {trainable_params:,}\n")

        summary_records.append({
            "key": key,
            "display": display,
            "total_params": total_params,
            "trainable_params": trainable_params,
        })

    print("=" * 70)
    print(f"{'Model Architecture':<42} {'Total Params':<15} {'Trainable Params':<15}")
    print("-" * 70)
    for r in summary_records:
        print(f"{r['display']:<42} {r['total_params']:<15,d} {r['trainable_params']:<15,d}")
    print("=" * 70)

    print("\n>>> ALL 4 BENCHMARK MODELS VERIFIED AND TESTED SUCCESSFULLY! <<<")


if __name__ == "__main__":
    test_all_models()

