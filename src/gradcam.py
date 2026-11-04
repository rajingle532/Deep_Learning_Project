"""
Explainable AI (XAI): Grad-CAM Heatmap Localization Generator.
Domain-Generalized Deep Learning for Real-World Plant Disease Recognition.

Paper Reference:
"Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization"
(Selvaraju et al., ICCV 2017).

Provides:
1. Target convolutional layer extraction (e.g. conv5_block3_out or conv5 in ResNet50).
2. Gradient computation of top predicted class score with respect to feature maps.
3. Alpha channel weighting, ReLU rectification, and spatial upsampling.
4. Jet colormap overlay onto original leaf imagery for agronomic inspection.
"""

from typing import Tuple, Optional
import numpy as np
import cv2
import tensorflow as tf
import keras


def find_target_conv_layer(model: keras.Model) -> str:
    """Automatically find the deepest 4D convolutional layer in the backbone."""
    # Preferred layer names in ResNet50
    for preferred in ["conv5_block3_out", "conv5", "top_conv", "Conv_1", "conv4_2"]:
        for layer in model.layers:
            if preferred in layer.name:
                return layer.name

    # Fallback: search backwards for any Conv2D / 4D output layer
    for layer in reversed(model.layers):
        if len(layer.output.shape) == 4 and "conv" in layer.name.lower():
            return layer.name

    raise ValueError("No suitable 4D convolutional layer found in model for Grad-CAM.")


def compute_gradcam_heatmap(
    model: keras.Model,
    image_tensor: tf.Tensor,
    class_idx: Optional[int] = None,
    target_layer_name: Optional[str] = None,
) -> np.ndarray:
    """
    Generate normalized 2D Grad-CAM activation heatmap [0.0, 1.0].

    Args:
        model: Trained Keras model.
        image_tensor: Normalized input tensor of shape (1, 224, 224, 3).
        class_idx: Target class index. If None, uses top predicted class.
        target_layer_name: Name of target conv layer. If None, auto-detected.

    Returns:
        2D numpy array (224, 224) with values in range [0.0, 1.0].
    """
    if target_layer_name is None:
        target_layer_name = find_target_conv_layer(model)

    target_layer = model.get_layer(target_layer_name)

    # Build gradient model mapping input -> (conv_output, model_output)
    grad_model = keras.Model(
        inputs=model.inputs,
        outputs=[target_layer.output, model.output],
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(image_tensor, training=False)
        if class_idx is None:
            class_idx = tf.argmax(predictions[0])
        loss = predictions[0, class_idx]

    # Gradients of class score w.r.t. conv feature maps
    grads = tape.gradient(loss, conv_outputs)

    # Global average pooling over spatial dimensions -> importance weights alpha_k
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Weighted combination of feature maps
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # ReLU rectification: only features that positively correlate with target class
    heatmap = tf.maximum(heatmap, 0.0)

    # Normalize to [0.0, 1.0]
    max_val = tf.math.reduce_max(heatmap)
    if max_val > 0.0:
        heatmap = heatmap / max_val

    heatmap_np = heatmap.numpy().astype(np.float32)

    # Resize heatmap to match input dimensions (224, 224)
    h, w = image_tensor.shape[1], image_tensor.shape[2]
    heatmap_resized = cv2.resize(heatmap_np, (w, h), interpolation=cv2.INTER_LINEAR)
    heatmap_resized = np.clip(heatmap_resized, 0.0, 1.0)
    return heatmap_resized


def overlay_gradcam(
    original_image: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.45,
    colormap: int = cv2.COLORMAP_JET,
) -> np.ndarray:
    """
    Superimpose colorized Grad-CAM heatmap onto original RGB image.

    Args:
        original_image: uint8 RGB numpy array (H, W, 3) in range [0, 255].
        heatmap: float32 numpy array (H, W) in range [0.0, 1.0].
        alpha: Blend weight for heatmap overlay (0.0 to 1.0).
        colormap: OpenCV colormap constant (default: cv2.COLORMAP_JET).

    Returns:
        Superimposed uint8 RGB numpy array (H, W, 3).
    """
    h, w = original_image.shape[:2]
    heatmap_scaled = np.uint8(255 * cv2.resize(heatmap, (w, h)))

    # Apply Jet color map (returns BGR)
    colored_heatmap_bgr = cv2.applyColorMap(heatmap_scaled, colormap)
    colored_heatmap_rgb = cv2.cvtColor(colored_heatmap_bgr, cv2.COLOR_BGR2RGB)

    # Weighted blend
    blended = (alpha * colored_heatmap_rgb.astype(np.float32)) + ((1.0 - alpha) * original_image.astype(np.float32))
    blended = np.clip(blended, 0, 255).astype(np.uint8)
    return blended

