"""
Domain Generalization (DG) Modules & Custom MixStyle Layer.
Domain-Generalized Deep Learning for Real-World Plant Disease Recognition.

Paper Reference:
"Domain Generalization with MixStyle" (Zhou et al., ICLR 2021).
URL: https://openreview.net/forum?id=6xHJ37MVxxp

Core Mechanism:
MixStyle dynamically mixes feature statistics (mean and standard deviation) across
different instances within intermediate convolutional layers during training.
Because feature statistics represent style and domain information (e.g. background soil,
laboratory lighting, sensor noise) while normalized features preserve biological lesion
content, MixStyle forces the network to learn style-invariant representation.
"""

import tensorflow as tf
import keras


@keras.saving.register_keras_serializable(package="PlantDG", name="MixStyle")
@keras.saving.register_keras_serializable(package="agri_guard", name="MixStyle")
class MixStyle(keras.layers.Layer):
    """
    MixStyle layer for feature-level Domain Generalization.

    Args:
        p: Probability of applying MixStyle during training (default: 0.5).
        alpha: Shape parameter for Beta(alpha, alpha) distribution (default: 0.1).
        eps: Small numerical constant for variance stabilization (default: 1e-6).
    """

    def __init__(self, p: float = 0.5, alpha: float = 0.1, eps: float = 1e-6, **kwargs):
        super().__init__(**kwargs)
        self.p = float(p)
        self.alpha = float(alpha)
        self.eps = float(eps)

    def call(self, x: tf.Tensor, training: bool = None) -> tf.Tensor:
        # Strictly identity passthrough during evaluation/testing
        if training is False:
            return x

        # Probabilistic application in graph mode using tf.cond
        should_apply = tf.random.uniform([]) < self.p

        def mix_features():
            batch_size = tf.shape(x)[0]

            # In channels_last format (B, H, W, C), compute spatial mean and variance
            mu = tf.reduce_mean(x, axis=[1, 2], keepdims=True)
            var = tf.math.reduce_variance(x, axis=[1, 2], keepdims=True)
            sig = tf.sqrt(var + self.eps)

            # Content-normalized feature representation
            norm_x = (x - mu) / sig

            # Sample mixing coefficient lambda from Beta(alpha, alpha) using
            # XLA-compatible symmetric inverse CDF transform (100% GPU / JIT compatible)
            u = tf.random.uniform([batch_size, 1, 1, 1], minval=1e-6, maxval=1.0 - 1e-6, dtype=tf.float32)
            inv_alpha = 1.0 / float(self.alpha)
            half = tf.constant(0.5, dtype=tf.float32)
            lam_f32 = tf.where(
                u < half,
                half * tf.pow(2.0 * u, inv_alpha),
                1.0 - half * tf.pow(2.0 * (1.0 - u), inv_alpha)
            )
            lam = tf.cast(lam_f32, dtype=x.dtype)
            one = tf.cast(1.0, dtype=x.dtype)

            # Shuffle instances across batch to simulate unseen environmental styles
            perm = tf.random.shuffle(tf.range(batch_size))
            mu_shuffled = tf.gather(mu, perm)
            sig_shuffled = tf.gather(sig, perm)

            # Synthesize novel domain style statistics
            mu_mix = (lam * mu) + ((one - lam) * mu_shuffled)
            sig_mix = (lam * sig) + ((one - lam) * sig_shuffled)

            # Re-stylize content with synthetic style
            return (sig_mix * norm_x) + mu_mix

        if training is True:
            return tf.cond(should_apply, mix_features, lambda: x)

        if training is None:
            return tf.cond(should_apply, mix_features, lambda: x)

        # training is a symbolic tf.Tensor
        return tf.cond(
            tf.cast(training, tf.bool),
            lambda: tf.cond(should_apply, mix_features, lambda: x),
            lambda: x,
        )

    def get_config(self):
        config = super().get_config()
        config.update({
            "p": self.p,
            "alpha": self.alpha,
            "eps": self.eps,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
