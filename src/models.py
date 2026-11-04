"""
Multi-Model Benchmark Architecture Suite.
Domain-Generalized Deep Learning for Real-World Plant Disease Recognition.

Models Implemented:
1. Model 1: Custom Baseline CNN (4 Conv Blocks from scratch, proves foundational DL).
2. Model 2: MobileNetV2 (Depthwise Separable, lightweight edge model for 4GB RTX 2050).
3. Model 3: ResNet50 ERM (Standard Empirical Risk Minimization transfer learning baseline).
4. Model 4: ResNet50 + MixStyle (Proposed SOTA Domain Generalization architecture).
"""

from pathlib import Path
from typing import Tuple, Optional
import tensorflow as tf
import keras
from keras.src.applications import resnet as resnet_lib

from src.domain_generalization import MixStyle
from src.taxonomy import NUM_CLASSES

PROJECT_ROOT = Path(__file__).resolve().parent.parent
WEIGHTS_DIR = PROJECT_ROOT / "models" / "pretrained_weights"


def build_custom_cnn(
    num_classes: int = NUM_CLASSES,
    input_shape: Tuple[int, int, int] = (224, 224, 3),
    dropout_rate: float = 0.4,
) -> keras.Model:
    """
    Model 1: 4-Block Convolutional Neural Network built from scratch.
    Demonstrates baseline representation learning without pretraining.
    """
    inputs = keras.Input(shape=input_shape, name="input_image")

    # Block 1
    x = keras.layers.Conv2D(32, (3, 3), padding="same", name="conv1_1")(inputs)
    x = keras.layers.BatchNormalization(name="bn1_1")(x)
    x = keras.layers.Activation("relu", name="relu1_1")(x)
    x = keras.layers.Conv2D(32, (3, 3), padding="same", name="conv1_2")(x)
    x = keras.layers.BatchNormalization(name="bn1_2")(x)
    x = keras.layers.Activation("relu", name="relu1_2")(x)
    x = keras.layers.MaxPooling2D((2, 2), name="pool1")(x)
    x = keras.layers.Dropout(0.2, name="drop1")(x)

    # Block 2
    x = keras.layers.Conv2D(64, (3, 3), padding="same", name="conv2_1")(x)
    x = keras.layers.BatchNormalization(name="bn2_1")(x)
    x = keras.layers.Activation("relu", name="relu2_1")(x)
    x = keras.layers.Conv2D(64, (3, 3), padding="same", name="conv2_2")(x)
    x = keras.layers.BatchNormalization(name="bn2_2")(x)
    x = keras.layers.Activation("relu", name="relu2_2")(x)
    x = keras.layers.MaxPooling2D((2, 2), name="pool2")(x)
    x = keras.layers.Dropout(0.25, name="drop2")(x)

    # Block 3
    x = keras.layers.Conv2D(128, (3, 3), padding="same", name="conv3_1")(x)
    x = keras.layers.BatchNormalization(name="bn3_1")(x)
    x = keras.layers.Activation("relu", name="relu3_1")(x)
    x = keras.layers.Conv2D(128, (3, 3), padding="same", name="conv3_2")(x)
    x = keras.layers.BatchNormalization(name="bn3_2")(x)
    x = keras.layers.Activation("relu", name="relu3_2")(x)
    x = keras.layers.MaxPooling2D((2, 2), name="pool3")(x)
    x = keras.layers.Dropout(0.3, name="drop3")(x)

    # Block 4
    x = keras.layers.Conv2D(256, (3, 3), padding="same", name="conv4_1")(x)
    x = keras.layers.BatchNormalization(name="bn4_1")(x)
    x = keras.layers.Activation("relu", name="relu4_1")(x)
    x = keras.layers.MaxPooling2D((2, 2), name="pool4")(x)
    x = keras.layers.Dropout(0.35, name="drop4")(x)

    # Classification Head
    x = keras.layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
    x = keras.layers.Dense(256, activation="relu", name="dense_features")(x)
    x = keras.layers.BatchNormalization(name="head_bn")(x)
    x = keras.layers.Dropout(dropout_rate, name="dense_drop")(x)
    outputs = keras.layers.Dense(num_classes, activation="softmax", name="classification_head")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="baseline_custom_cnn")
    return model


def build_mobilenet_v2(
    num_classes: int = NUM_CLASSES,
    input_shape: Tuple[int, int, int] = (224, 224, 3),
    pretrained: bool = True,
    dropout_rate: float = 0.35,
) -> keras.Model:
    """
    Model 2: MobileNetV2 with Depthwise Separable Convolutions.
    Lightweight, edge-optimized for real-time mobile agricultural deployment.
    """
    inputs = keras.Input(shape=input_shape, name="input_image")

    # Backbone
    base_model = keras.applications.MobileNetV2(
        include_top=False,
        weights=None,
        input_tensor=inputs,
    )

    if pretrained:
        weight_file = WEIGHTS_DIR / "mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_224_no_top.h5"
        if weight_file.exists():
            base_model.load_weights(str(weight_file))
        else:
            print("  [WARN] Local MobileNetV2 weights not found, using uninitialized weights.")

    # Classification Head
    x = base_model.output
    x = keras.layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
    x = keras.layers.BatchNormalization(name="head_bn")(x)
    x = keras.layers.Dropout(dropout_rate, name="head_dropout")(x)
    outputs = keras.layers.Dense(num_classes, activation="softmax", name="classification_head")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="mobilenet_v2_erm")
    return model


def build_resnet50_erm(
    num_classes: int = NUM_CLASSES,
    input_shape: Tuple[int, int, int] = (224, 224, 3),
    pretrained: bool = True,
    dropout_rate: float = 0.4,
) -> keras.Model:
    """
    Model 3: Standard ResNet50 trained under Empirical Risk Minimization (ERM).
    Industry benchmark transfer learning baseline.
    """
    inputs = keras.Input(shape=input_shape, name="input_image")

    # Backbone
    base_model = keras.applications.ResNet50(
        include_top=False,
        weights=None,
        input_tensor=inputs,
    )

    if pretrained:
        weight_file = WEIGHTS_DIR / "resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5"
        if weight_file.exists():
            base_model.load_weights(str(weight_file))
        else:
            print("  [WARN] Local ResNet50 weights not found, using uninitialized weights.")

    # Classification Head
    x = base_model.output
    x = keras.layers.GlobalAveragePooling2D(name="avg_pool")(x)
    x = keras.layers.BatchNormalization(name="head_bn")(x)
    x = keras.layers.Dense(512, activation="relu", name="head_dense")(x)
    x = keras.layers.Dropout(dropout_rate, name="head_dropout")(x)
    outputs = keras.layers.Dense(num_classes, activation="softmax", name="classification_head")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="resnet50_erm_baseline")
    return model


def build_resnet50_mixstyle(
    num_classes: int = NUM_CLASSES,
    input_shape: Tuple[int, int, int] = (224, 224, 3),
    p: float = 0.5,
    alpha: float = 0.1,
    pretrained: bool = True,
    dropout_rate: float = 0.4,
) -> keras.Model:
    """
    Model 4 (Proposed SOTA): ResNet50 with MixStyle Domain Generalization Layer.
    Mixes feature statistics across instances after Stage 1 and Stage 2 to learn
    domain-invariant leaf pathology representations.
    """
    inputs = keras.Input(shape=input_shape, name="input_image")

    # Stem: conv1 (112x112x64)
    x = keras.layers.ZeroPadding2D(padding=((3, 3), (3, 3)), name="conv1_pad")(inputs)
    x = keras.layers.Conv2D(64, 7, strides=2, use_bias=True, name="conv1_conv")(x)
    x = keras.layers.BatchNormalization(axis=3, epsilon=1.001e-5, name="conv1_bn")(x)
    x = keras.layers.Activation("relu", name="conv1_relu")(x)
    x = keras.layers.ZeroPadding2D(padding=((1, 1), (1, 1)), name="pool1_pad")(x)
    x = keras.layers.MaxPooling2D(3, strides=2, name="pool1_pool")(x)

    # Stage 1: conv2 (56x56x256)
    x = resnet_lib.stack_residual_blocks_v1(x, 64, 3, stride1=1, name="conv2")
    # MixStyle Layer 1 (Inter-stage feature mixing)
    x = MixStyle(p=p, alpha=alpha, name="mixstyle_stage1")(x)

    # Stage 2: conv3 (28x28x512)
    x = resnet_lib.stack_residual_blocks_v1(x, 128, 4, name="conv3")
    # MixStyle Layer 2 (Inter-stage feature mixing)
    x = MixStyle(p=p, alpha=alpha, name="mixstyle_stage2")(x)

    # Stage 3: conv4 (14x14x1024)
    x = resnet_lib.stack_residual_blocks_v1(x, 256, 6, name="conv4")

    # Stage 4: conv5 (7x7x2048) - Target for Grad-CAM
    x = resnet_lib.stack_residual_blocks_v1(x, 512, 3, name="conv5")

    # Classification Head
    x = keras.layers.GlobalAveragePooling2D(name="avg_pool")(x)
    x = keras.layers.BatchNormalization(name="head_bn")(x)
    x = keras.layers.Dense(512, activation="relu", name="head_dense")(x)
    x = keras.layers.Dropout(dropout_rate, name="head_dropout")(x)
    outputs = keras.layers.Dense(num_classes, activation="softmax", name="classification_head")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="resnet50_mixstyle_dg")

    if pretrained:
        weight_file = WEIGHTS_DIR / "resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5"
        if weight_file.exists():
            model.load_weights(str(weight_file), by_name=True)
        else:
            print("  [WARN] Local ResNet50 weights not found, using uninitialized weights.")

    return model


def build_model(model_name: str, num_classes: int = NUM_CLASSES, **kwargs) -> keras.Model:
    """Unified factory to instantiate benchmark models by key string."""
    name_clean = model_name.lower().replace("-", "_").replace(" ", "_")

    if name_clean in ("cnn", "baseline_cnn", "custom_cnn"):
        return build_custom_cnn(num_classes=num_classes, **kwargs)
    elif name_clean in ("mobilenet", "mobilenet_v2", "mobilenetv2"):
        return build_mobilenet_v2(num_classes=num_classes, **kwargs)
    elif name_clean in ("resnet", "resnet50", "resnet50_erm"):
        return build_resnet50_erm(num_classes=num_classes, **kwargs)
    elif name_clean in ("resnet50_mixstyle", "mixstyle", "dg_resnet"):
        return build_resnet50_mixstyle(num_classes=num_classes, **kwargs)
    else:
        raise ValueError(
            f"Unknown model name '{model_name}'. Available: 'custom_cnn', 'mobilenet_v2', 'resnet50_erm', 'resnet50_mixstyle'"
        )

