"""
Unit Test Suite for Two-Stage Hierarchical Inference Engine.
Domain-Generalized Deep Learning for Real-World Plant Disease Recognition.
"""

import unittest
import numpy as np

from src.taxonomy import CANONICAL_CLASSES, NUM_CLASSES
from src.hierarchical import (
    HierarchicalDiagnosticsEngine,
    HIERARCHICAL_CROP_MAP,
    HierarchicalDiagnosis,
)


class TestHierarchicalEngine(unittest.TestCase):
    def setUp(self):
        self.engine = HierarchicalDiagnosticsEngine(
            crop_confidence_threshold=0.65,
            crop_entropy_threshold=0.95,
        )

    def test_crop_partition_coverage(self):
        """Ensure all 28 canonical classes are partitioned across crop families."""
        all_classes = []
        for crop, classes in HIERARCHICAL_CROP_MAP.items():
            all_classes.extend(classes)
        
        self.assertEqual(len(all_classes), NUM_CLASSES)
        self.assertEqual(set(all_classes), set(CANONICAL_CLASSES))

    def test_confident_apple_prediction(self):
        """Simulate a confident Apple leaf prediction."""
        preds = np.zeros(NUM_CLASSES, dtype=np.float32)
        # Put 90% probability on Apple Scab
        scab_idx = CANONICAL_CLASSES.index("apple_apple_scab")
        rust_idx = CANONICAL_CLASSES.index("apple_cedar_apple_rust")
        preds[scab_idx] = 0.85
        preds[rust_idx] = 0.10
        # Remaining 5% spread
        preds /= preds.sum()

        diag = self.engine.diagnose(preds, user_crop_prior="Auto-Detect (Global)")
        self.assertEqual(diag.detected_crop, "Apple")
        self.assertFalse(diag.is_crop_ambiguous)
        self.assertEqual(diag.top_condition_canonical, "apple_apple_scab")
        self.assertAlmostEqual(diag.top_condition_prob, 0.85 / 0.95, places=2)

    def test_ambiguous_prediction_triggers_guard(self):
        """Simulate an ambiguous prediction spread between Corn and Tomato."""
        preds = np.zeros(NUM_CLASSES, dtype=np.float32)
        corn_idx = CANONICAL_CLASSES.index("corn_gray_leaf_spot")
        tomato_idx = CANONICAL_CLASSES.index("tomato_late_blight")
        preds[corn_idx] = 0.40
        preds[tomato_idx] = 0.40
        preds /= preds.sum()

        diag = self.engine.diagnose(preds, user_crop_prior="Auto-Detect (Global)")
        self.assertTrue(diag.is_crop_ambiguous)
        self.assertIn("Ambiguity Detected", diag.status_markdown)

    def test_user_prior_overrides_cross_crop_hallucination(self):
        """
        Simulate an out-of-domain Apple leaf where unconditioned ResNet leaked 70% onto Corn,
        but user explicitly specifies 'Apple'.
        """
        preds = np.zeros(NUM_CLASSES, dtype=np.float32)
        corn_idx = CANONICAL_CLASSES.index("corn_gray_leaf_spot")
        scab_idx = CANONICAL_CLASSES.index("apple_apple_scab")
        rust_idx = CANONICAL_CLASSES.index("apple_cedar_apple_rust")
        
        preds[corn_idx] = 0.70
        preds[scab_idx] = 0.18
        preds[rust_idx] = 0.12
        preds /= preds.sum()

        # User confirms this is Apple
        diag = self.engine.diagnose(preds, user_crop_prior="Apple")
        self.assertEqual(diag.resolved_crop, "Apple")
        self.assertTrue(diag.prior_override_active)
        self.assertEqual(diag.top_condition_canonical, "apple_apple_scab")
        # Apple Scab vs Cedar Rust inside Apple: 0.18 / 0.30 = 60%
        self.assertAlmostEqual(diag.top_condition_prob, 0.60, places=2)
        # Verify Corn is NOT in top conditioned probs!
        for k in diag.top_conditioned_probs.keys():
            self.assertNotIn("Corn", k)


if __name__ == "__main__":
    unittest.main()
