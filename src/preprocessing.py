"""
Domain-Robust Image Preprocessing & Augmentation Engine.
Domain-Generalized Deep Learning for Real-World Plant Disease Recognition.

Key Components:
1. Core Transforms: Bilinear resizing (224, 224, 3) and ImageNet standardization.
2. Domain-Invariant Augmentations:
   - Photometric Color Jitter (Brightness, Contrast, Saturation, Hue).
   - Weather & Field Perturbations (Synthetic Shadows & Defocus Blur).
   - Geometric Affine (Flips, Random Rotations +/- 20 degrees).
3. Batch-level MixUp Regularization (Beta(0.2, 0.2) interpolation).
4. Un-normalization helper for Grad-CAM and Web UI visualization.
"""

from typing import Tuple
import tensorflow as tf

# Standard ImageNet distribution constants
IMAGENET_MEAN = tf.constant([0.485, 0.456, 0.406], dtype=tf.float32)
IMAGENET_STD = tf.constant([0.229, 0.224, 0.225], dtype=tf.float32)


def normalize_image(image: tf.Tensor, normalization: str = "imagenet") -> tf.Tensor:
    """
    Scale and normalize float32 RGB image tensor [0.0, 255.0] to standard distribution.
    """
    img = tf.cast(image, tf.float32) / 255.0
    if normalization == "imagenet":
        return (img - IMAGENET_MEAN) / IMAGENET_STD
    elif normalization == "tf":
        return (img * 2.0) - 1.0
    return img


def unnormalize_image(image: tf.Tensor, normalization: str = "imagenet") -> tf.Tensor:
    """
    Convert normalized float32 tensor back to uint8 [0, 255] for visual display.
    """
    if normalization == "imagenet":
        img = (image * IMAGENET_STD) + IMAGENET_MEAN
    elif normalization == "tf":
        img = (image + 1.0) / 2.0
    else:
        img = image

    img = tf.clip_by_value(img * 255.0, 0.0, 255.0)
    return tf.cast(img, tf.uint8)


def random_shadow(image: tf.Tensor, shadow_prob: float = 0.35) -> tf.Tensor:
    """
    Simulate outdoor agricultural shadow patterns across the leaf foliage.
    Draws a randomized vertical or diagonal attenuation mask.
    """
    should_apply = tf.random.uniform([]) < shadow_prob

    def apply_shadow():
        h = tf.shape(image)[0]
        w = tf.shape(image)[1]
        # Generate linear ramp
        ramp = tf.linspace(0.55, 1.0, w)
        # Randomly flip ramp left-to-right
        if tf.random.uniform([]) > 0.5:
            ramp = tf.reverse(ramp, axis=[0])
        mask = tf.expand_dims(ramp, axis=0)
        mask = tf.repeat(mask, repeats=h, axis=0)
        mask = tf.expand_dims(mask, axis=-1)
        return image * mask

    return tf.cond(should_apply, apply_shadow, lambda: image)


def random_blur(image: tf.Tensor, blur_prob: float = 0.25) -> tf.Tensor:
    """
    Simulate mobile phone camera defocus and motion blur using 3x3 box blur.
    """
    should_apply = tf.random.uniform([]) < blur_prob

    def apply_blur():
        # 3x3 uniform smoothing kernel
        kernel = tf.ones((3, 3, 3, 1), dtype=tf.float32) / 9.0
        img_expanded = tf.expand_dims(image, axis=0)
        # Depthwise conv for 3 color channels
        blurred = tf.nn.depthwise_conv2d(img_expanded, kernel, strides=[1, 1, 1, 1], padding="SAME")
        return tf.squeeze(blurred, axis=0)

    return tf.cond(should_apply, apply_blur, lambda: image)


def domain_robust_augmentation(image: tf.Tensor) -> tf.Tensor:
    """
    Apply comprehensive domain-invariant augmentations to simulate real field conditions.
    Input: float32 tensor in range [0.0, 255.0].
    """
    # 1. Geometric transforms
    img = tf.image.random_flip_left_right(image)
    img = tf.image.random_flip_up_down(img)

    # Random 90-degree discrete rotations
    k = tf.random.uniform([], minval=0, maxval=4, dtype=tf.int32)
    img = tf.image.rot90(img, k=k)

    # 2. Photometric Color Jitter (Simulate shifting sunlight, cloudy vs bright noon)
    img = tf.image.random_brightness(img, max_delta=25.0)
    img = tf.image.random_contrast(img, lower=0.75, upper=1.25)
    img = tf.image.random_saturation(img, lower=0.75, upper=1.30)
    img = tf.image.random_hue(img, max_delta=0.06)

    # 3. Weather & In-The-Wild Perturbations
    img = random_shadow(img, shadow_prob=0.35)
    img = random_blur(img, blur_prob=0.25)

    # Ensure valid [0, 255] clipping
    img = tf.clip_by_value(img, 0.0, 255.0)
    return img


def mixup(
    images: tf.Tensor,
    labels: tf.Tensor,
    alpha: float = 0.2,
) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Convex combination of paired training samples to prevent memorization of spurious background cues.
    x_tilde = lambda * x_i + (1 - lambda) * x_j
    y_tilde = lambda * y_i + (1 - lambda) * y_j
    where lambda ~ Beta(alpha, alpha).
    """
    batch_size = tf.shape(images)[0]
    # Sample lambda from Beta(alpha, alpha) via two independent Gamma distributions
    g1 = tf.random.gamma(shape=[batch_size, 1, 1, 1], alpha=alpha)
    g2 = tf.random.gamma(shape=[batch_size, 1, 1, 1], alpha=alpha)
    lam_img = g1 / (g1 + g2)
    lam_lbl = tf.reshape(lam_img, [batch_size, 1])

    # Shuffle indices
    indices = tf.random.shuffle(tf.range(batch_size))
    shuffled_images = tf.gather(images, indices)
    shuffled_labels = tf.gather(labels, indices)

    # Convex interpolation
    mixed_images = (lam_img * images) + ((1.0 - lam_img) * shuffled_images)
    mixed_labels = (lam_lbl * labels) + ((1.0 - lam_lbl) * shuffled_labels)

    return mixed_images, mixed_labels
