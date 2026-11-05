"""
Hierarchical Two-Stage Inference Engine for Foliar Pathology.
Domain-Generalized Deep Learning for Real-World Plant Disease Recognition.

Solves the Flat Softmax Cross-Crop Hallucination Problem:
Stage 1: Botanical Crop Species Verification (Marginalizes foliar posteriors across species).
Stage 2: Specialist Foliar Pathology Diagnosis (Conditioned strictly on verified or user-confirmed crop).
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np

from src.taxonomy import (
    CANONICAL_CLASSES,
    CANONICAL_DISPLAY_NAMES,
    TREATMENT_ADVISORY,
    NUM_CLASSES,
)

# Comprehensive Disjoint Partitioning of Canonical Classes by Plant Species
HIERARCHICAL_CROP_MAP: Dict[str, List[str]] = {
    "Tomato": [c for c in CANONICAL_CLASSES if c.startswith("tomato_")],
    "Apple": [c for c in CANONICAL_CLASSES if c.startswith("apple_")],
    "Corn": [c for c in CANONICAL_CLASSES if c.startswith("corn_")],
    "Potato": [c for c in CANONICAL_CLASSES if c.startswith("potato_")],
    "Grape": [c for c in CANONICAL_CLASSES if c.startswith("grape_")],
    "Bell Pepper": [c for c in CANONICAL_CLASSES if c.startswith("bell_pepper_")],
    "Cherry": [c for c in CANONICAL_CLASSES if c.startswith("cherry_")],
    "Peach": [c for c in CANONICAL_CLASSES if c.startswith("peach_")],
    "Strawberry": [c for c in CANONICAL_CLASSES if c.startswith("strawberry_")],
    "Other Field Crops": [
        c for c in CANONICAL_CLASSES 
        if not any(c.startswith(p) for p in [
            "tomato_", "apple_", "corn_", "potato_", "grape_", 
            "bell_pepper_", "cherry_", "peach_", "strawberry_"
        ])
    ],
}


@dataclass
class HierarchicalDiagnosis:
    """Encapsulates the complete two-stage hierarchical inference result."""
    # Stage 1: Crop Species Level
    resolved_crop: str
    detected_crop: str
    stage1_confidence: float
    stage1_entropy: float
    is_crop_ambiguous: bool
    candidate_crops: List[Tuple[str, float]]
    prior_override_active: bool

    # Stage 2: Specialist Pathology Level
    top_condition_display: str
    top_condition_canonical: str
    top_condition_prob: float
    top_conditioned_probs: Dict[str, float]
    top_disease_idx: int

    # Clinical & Safety Assessment
    status_markdown: str
    advisory_markdown: str


class HierarchicalDiagnosticsEngine:
    """
    Two-Stage Hierarchical Inference Engine.
    Prevents single-layer flat softmax models from hallucinating cross-crop diseases
    (e.g., misclassifying outdoor Apple leaves as Corn or Tomato).
    """

    def __init__(
        self,
        crop_confidence_threshold: float = 0.65,
        crop_entropy_threshold: float = 0.95,
    ):
        self.crop_map = HIERARCHICAL_CROP_MAP
        self.crop_to_indices = {
            crop: [CANONICAL_CLASSES.index(c) for c in classes if c in CANONICAL_CLASSES]
            for crop, classes in self.crop_map.items()
        }
        self.crop_confidence_threshold = crop_confidence_threshold
        self.crop_entropy_threshold = crop_entropy_threshold

    def evaluate_stage1_crop(
        self, raw_global_preds: np.ndarray
    ) -> Tuple[str, float, float, bool, List[Tuple[str, float]]]:
        """
        Stage 1: Marginalize class posteriors across crop taxonomic families.
        P(Crop = c | X) = sum_{y in Crop_c} P(Y = y | X)
        """
        crop_probs: Dict[str, float] = {}
        for crop, indices in self.crop_to_indices.items():
            if indices:
                crop_probs[crop] = float(np.sum([raw_global_preds[i] for i in indices]))
            else:
                crop_probs[crop] = 0.0

        sorted_crops = sorted(crop_probs.items(), key=lambda item: item[1], reverse=True)
        detected_crop, top_conf = sorted_crops[0]

        # Calculate Shannon entropy across crop distribution
        p_vec = np.array([p for _, p in sorted_crops])
        p_vec = np.maximum(p_vec, 1e-12)
        crop_entropy = float(-np.sum(p_vec * np.log(p_vec)))

        # A crop is considered ambiguous if confidence is low OR entropy across species is high
        is_ambiguous = (top_conf < self.crop_confidence_threshold) or (
            crop_entropy > self.crop_entropy_threshold
        )

        return detected_crop, top_conf, crop_entropy, is_ambiguous, sorted_crops[:4]

    def evaluate_stage2_pathology(
        self,
        raw_global_preds: np.ndarray,
        crop: str,
    ) -> Tuple[str, str, float, Dict[str, float], int]:
        """
        Stage 2: Specialist foliar pathology conditioned on confirmed crop.
        P(Disease = d | Crop = c, X) = P(Disease = d | X) / sum_{k in Crop_c} P(Disease = k | X)
        """
        indices = self.crop_to_indices.get(crop, [])
        if not indices:
            # Fallback to unconditioned top-1 if crop family is unknown
            top_idx = int(np.argmax(raw_global_preds))
            c_name = CANONICAL_CLASSES[top_idx]
            d_name = CANONICAL_DISPLAY_NAMES.get(c_name, c_name)
            return d_name, c_name, float(raw_global_preds[top_idx]), {d_name: float(raw_global_preds[top_idx])}, top_idx

        raw_sub = np.array([raw_global_preds[i] for i in indices])
        sub_sum = float(raw_sub.sum())
        denom = sub_sum if sub_sum > 1e-9 else 1.0
        norm_sub = raw_sub / denom

        sorted_local = np.argsort(norm_sub)[::-1]
        top_local = sorted_local[0]
        top_global_idx = indices[top_local]
        top_prob = float(norm_sub[top_local])

        top_canonical = CANONICAL_CLASSES[top_global_idx]
        top_display = CANONICAL_DISPLAY_NAMES.get(top_canonical, top_canonical)

        top_probs_dict = {}
        for l_idx in sorted_local[:min(4, len(sorted_local))]:
            g_idx = indices[l_idx]
            c_name = CANONICAL_CLASSES[g_idx]
            d_name = CANONICAL_DISPLAY_NAMES.get(c_name, c_name)
            top_probs_dict[d_name] = float(norm_sub[l_idx])

        return top_display, top_canonical, top_prob, top_probs_dict, top_global_idx

    def diagnose(
        self,
        raw_global_preds: np.ndarray,
        user_crop_prior: Optional[str] = None,
        latency_ms: float = 0.0,
    ) -> HierarchicalDiagnosis:
        """
        Execute full Two-Stage Hierarchical Pipeline.
        """
        # Step 1: Stage 1 Crop Family Evaluation
        detected_crop, stage1_conf, stage1_entropy, is_ambiguous, candidates = (
            self.evaluate_stage1_crop(raw_global_preds)
        )

        prior_active = bool(
            user_crop_prior
            and user_crop_prior != "Auto-Detect (Global)"
            and user_crop_prior in self.crop_map
        )

        # Resolved Crop: User prior takes definitive precedence if specified
        resolved_crop = user_crop_prior if prior_active else detected_crop

        # Step 2: Stage 2 Specialist Pathology Evaluation
        top_display, top_canonical, top_prob, conditioned_dict, top_idx = (
            self.evaluate_stage2_pathology(raw_global_preds, resolved_crop)
        )

        # Step 3: Format Clinical Status Markdown
        if prior_active:
            status_md = f"""### 🌿 Stage 1: Verified Crop Species &bull; `{resolved_crop}` (User Prior)
* **Botanical Species:** **`{resolved_crop}`**
* **Stage 2 Pathology:** **`{top_display}`**
* **Conditioned Posterior:** **`{top_prob * 100:.1f}%`**
* **Hierarchical Shield:** *Cross-crop contamination (e.g. Corn or Tomato leakage onto Apple foliage) 100% neutralized via Stage-1 Bayesian gating.*
* **Inference Latency:** `{latency_ms:.1f} ms`"""
        elif is_ambiguous:
            cand_str = ", ".join([f"{c} ({p*100:.0f}%)" for c, p in candidates[:3]])
            status_md = f"""### ⚠️ Stage 1: Crop Species Ambiguity Detected
* **Top Automated Match:** `{detected_crop}` ({stage1_conf * 100:.1f}%)
* **Candidate Species:** `{cand_str}`
* **Crop Entropy:** `{stage1_entropy:.2f} nats` (Foliage pattern spreads across multiple species)
* **Stage 2 Tentative Match:** **`{top_display}`** ({top_prob * 100:.1f}%)
* **Agronomic Safeguard:** *To lock in verified diagnosis and eliminate field clutter error, select the exact crop from the **Botanical Crop Prior** dropdown.*"""
        else:
            status_md = f"""### ✅ Stage 1: Automated Species Identification Confirmed
* **Identified Crop:** **`{detected_crop}`** (Confidence: `{stage1_conf * 100:.1f}%`)
* **Stage 2 Diagnosis:** **`{top_display}`**
* **Diagnostic Confidence:** **`{top_prob * 100:.1f}%`**
* **Inference Mode:** Two-Stage Hierarchical Pipeline &bull; `{latency_ms:.1f} ms`"""

        # Step 4: Agronomic Treatment Advisory
        advisory = TREATMENT_ADVISORY.get(top_canonical, {})
        urgency = advisory.get("urgency", "Moderate")
        desc = advisory.get("description", "Consult local agricultural extension office.")
        chem = advisory.get("chemical", "Standard copper-based fungicide or approved bactericide.")
        bio = advisory.get("organic", "Bacillus subtilis or neem extract spray.")
        prev = advisory.get("prevention", "Ensure drip irrigation and crop rotation.")

        advisory_md = f"""| Regimen Parameter | Certified Agricultural Guidance |
| :--- | :--- |
| **Target Crop Family** | `{resolved_crop}` |
| **Pathogen Urgency** | `{urgency}` |
| **Clinical Description** | {desc} |
| **Chemical Regimen** | {chem} |
| **Biological / Organic Treatment** | {bio} |
| **Preventive Management** | {prev} |"""

        return HierarchicalDiagnosis(
            resolved_crop=resolved_crop,
            detected_crop=detected_crop,
            stage1_confidence=stage1_conf,
            stage1_entropy=stage1_entropy,
            is_crop_ambiguous=is_ambiguous,
            candidate_crops=candidates,
            prior_override_active=prior_active,
            top_condition_display=top_display,
            top_condition_canonical=top_canonical,
            top_condition_prob=top_prob,
            top_conditioned_probs=conditioned_dict,
            top_disease_idx=top_idx,
            status_markdown=status_md,
            advisory_markdown=advisory_md,
        )
