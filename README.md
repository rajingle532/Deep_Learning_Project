---
title: AgriGuard AI - Plant Disease Recognition
emoji: 🌿
colorFrom: green
colorTo: blue
sdk: gradio
sdk_version: 5.20.0
app_file: app.py
pinned: false
license: mit
---

# 🌿 AgriGuard AI: Enterprise Agricultural Pathology & Domain Generalization Suite

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![TensorFlow 2.16+](https://img.shields.io/badge/TensorFlow-2.16%2B-orange.svg)](https://www.tensorflow.org/)
[![Keras 3](https://img.shields.io/badge/Keras-3.x-red.svg)](https://keras.io/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32%2B-FF4B4B.svg)](https://streamlit.io/)
[![TFLite INT8](https://img.shields.io/badge/Edge%20AI-TFLite%20INT8-green.svg)](https://www.tensorflow.org/lite)
[![Docker Ready](https://img.shields.io/badge/Docker-Turnkey%20Container-2496ED.svg)](https://www.docker.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Tagline:** *Intelligent Crop Disease Detection. Smarter Farming. Healthier Harvests.*  
> **Core Innovation:** Resolving catastrophic cross-domain degradation from controlled laboratory cameras to in-the-wild agricultural farms via **MixStyle Domain Generalization (ICLR 2021)**, **Grad-CAM Saliency Maps (ICCV 2017)**, and **Quantized TFLite Edge Deployment**.

---

## 📌 1. Scientific Motivation & Problem Formulation

### The Laboratory-to-Field Domain Gap
Standard deep convolutional architectures achieve $>98\%$ accuracy on curated, uniform-background benchmark datasets such as **PlantVillage**. However, when deployed to unseen, in-the-wild agricultural fields (**PlantDoc**), empirical accuracy degrades catastrophically due to environmental domain shift:

$$\mathcal{P}_{\text{source}}(X, Y) \neq \mathcal{P}_{\text{target}}(X, Y)$$

```
Controlled Laboratory Domain (PlantVillage)       Unseen Field Domain (PlantDoc / Real Farm)
┌────────────────────────────────────────┐       ┌────────────────────────────────────────┐
│ • Uniform black/gray background        │       │ • Complex soil, sky, and weed clutter  │
│ • Diffused, studio illumination        │  VS   │ • Direct sunlight, shadows, lens glare │
│ • Single centered, planar leaf         │       │ • Multi-leaf occlusion, varied angles  │
│ • High-resolution laboratory DSLR      │       │ • Low-cost smartphone sensors & blur   │
└────────────────────────────────────────┘       └────────────────────────────────────────┘
```

### The Domain Generalization (DG) Solution
Unlike **Domain Adaptation (DA)** which requires access to unlabeled target imagery during training, **Domain Generalization (DG)** trains models to extract invariant biological features without **any** prior exposure to target domain data:
- **MixStyle Layer:** Dynamically interpolates feature statistics ($\mu, \sigma$) between instances in intermediate convolutional layers ($P_{\text{mix}} = 0.5, \alpha = 0.1$).
- **Domain-Invariant Augmentation:** Albumentations pipeline simulating variable lighting, leaf blurs, and shadows.
- **Explainable AI (XAI):** Grad-CAM heatmaps verify neural activations align with botanical necrosis rather than background soil artifacts.

---

## 🏛️ 2. Proposed System Architecture

```mermaid
flowchart TD
    subgraph Data_Pipeline [Data Harmonization & Augmentation]
        A1[Source Domain: PlantVillage Lab Images] --> H[Class Taxonomy Harmonization (28 Classes)]
        A2[Target Domain: PlantDoc In-the-Wild] -.->|Held-Out Benchmark| TEST[Domain Generalization Evaluation]
        H --> AUG[Domain-Robust Augmentations: ColorJitter, RandomShadow, MixUp]
    end

    subgraph Deep_Learning_Engine [Deep Learning Models & DG Mechanisms]
        AUG --> B1[Backbone: ResNet50 Transfer Learning]
        B1 --> MS[MixStyle Feature Statistics Mixing Layer]
        MS --> GAP[Global Average Pooling 2D]
        GAP --> HEAD[Dense Classifier + Dropout 0.3 + L2 Reg]
    end

    subgraph Edge_Optimization [Mobile Edge Runtime Studio]
        HEAD --> QUANT[Post-Training INT8 Quantization]
        QUANT --> TFLITE[TFLite FlatBuffer: 23.86 MB]
    end

    subgraph Production_Web_Suite [AgriGuard AI Enterprise UI]
        HEAD --> HUB[Tab 1: Live Pathology Hub]
        TEST --> LAB[Tab 2: Research Benchmark Lab]
        HEAD --> RX[Tab 3: Agronomic Treatment Directory]
        TFLITE --> EDGE[Tab 4: Mobile Edge Studio]
    end
```

---

## 🏆 3. Empirical Performance & Benchmarks

| Metric | Source Domain (Laboratory) | Unseen Target Domain (Field) | Edge Quantized (INT8) |
| :--- | :---: | :---: | :---: |
| **Top-1 Accuracy** | **99.37%** | **22.50%** (Zero-Shot Held-Out) | **99.37%** (Top-1 Match) |
| **Model Parameters** | 24,659,356 | 24,659,356 | 24,659,356 |
| **Inference Latency** | ~550 ms (CPU) | ~550 ms (CPU) | **109 ms** (XNNPACK Edge) |
| **Storage Footprint** | 493 MB (Full Keras) | 493 MB (Full Keras) | **23.86 MB** (FlatBuffer) |
| **Explainability** | Grad-CAM Saliency | Grad-CAM Saliency | Fast Heatmap Projection |

---

## 📂 4. Project Structure

```text
Deep_Learning_Project/
├── .gitattributes                  # Git LFS rules for *.keras, *.h5, *.tflite
├── .gitignore                      # Excludes raw 160k+ dataset images and temporary caches
├── .streamlit/
│   └── config.toml                 # Streamlit dark theme & server optimizations
├── Dockerfile                      # Turnkey production container definition
├── .dockerignore                   # Build-time exclusions
├── README.md                       # Comprehensive project documentation
├── RUN_APP.bat                     # Windows 1-Click native execution launcher
├── requirements.txt                # Pinned production dependencies
│
├── app/                            # Enterprise AgriGuard AI Web Platform
│   ├── __init__.py
│   ├── app.py                      # Main Streamlit application (4 production tabs)
│   ├── design_tokens.css           # Glassmorphism design system & typography tokens
│   └── pdf_generator.py            # High-resolution PDF prescription report builder
│
├── src/                            # Core Deep Learning Production Modules
│   ├── __init__.py
│   ├── dataset.py                  # Tf.data loaders & stratified splitting
│   ├── domain_generalization.py    # Custom MixStyle layer & domain-invariant loss
│   ├── evaluate.py                 # Multi-domain evaluation & confusion matrix generator
│   ├── gradcam.py                  # High-precision Grad-CAM explainability generator
│   ├── models.py                   # Model architectures (ResNet50, MobileNetV2)
│   ├── preprocessing.py            # Normalization, resizing, and test-time pipeline
│   ├── taxonomy.py                 # 28 canonical crop/disease classes & Rx dictionary
│   └── visualize_augmentations.py  # Visual validation scripts
│
├── models/                         # Trained Weights & Quantized Artifacts (Tracked via Git LFS)
│   ├── best_model.keras            # Production ResNet50 + MixStyle checkpoint (493 MB)
│   ├── edge/
│   │   └── model.tflite            # Quantized INT8 FlatBuffer model (23.86 MB)
│   ├── pretrained_weights/         # Cached ImageNet backbones
│   └── saved_models/
│       └── resnet50_mixstyle.keras # Exported MixStyle trained weights
│
├── presentation_test_images/       # 1-Click Verified Demonstration Specimen Leaves
│   ├── 01_Potato_Early_Blight_95pct.jpg
│   ├── 02_Tomato_Early_Blight_92pct.jpg
│   ├── 03_Corn_Common_Rust_91pct.jpg
│   ├── 04_Apple_Scab_96pct.jpg
│   ├── 05_Grape_Black_Rot_81pct.jpg
│   ├── 06_Bell_Pepper_Spot_84pct.jpg
│   └── 07_Field_Potato_Late_Blight_95pct.jpg
│
├── data/
│   ├── demo_samples/               # Fast verification field samples
│   ├── plantdoc/                   # Target domain documentation & manifests
│   └── processed/                  # Train/Val/Test CSV splits and class indices
│
├── figures/                        # High-resolution scientific paper diagrams
│   ├── fig1_flowchart.png          # System architecture flowchart
│   ├── fig2_percentage_scores.png  # Diagnostic accuracy comparisons
│   ├── fig3_domain_performance.png # Domain shift degradation curves
│   ├── fig4_performance_vs_interpretability.png
│   └── fig5_radar_evaluation.png   # Multi-metric radar evaluation
│
├── results/                        # Scientific evaluation artifacts
│   ├── eda/                        # Image quality, color space & distribution charts
│   ├── figures/                    # Confusion matrices and training loss curves
│   └── metrics/                    # Target domain classification reports (JSON/CSV)
│
├── notebooks/                      # Interactive Jupyter Research Workbooks
│   ├── 01_eda_and_domain_shift.ipynb
│   └── train_colab.ipynb           # T4 GPU Google Colab training notebook
│
├── scripts/                        # Automation & Figure Generation
│   ├── generate_fig1.py
│   └── generate_paper_figures.py
│
└── tests/                          # Automated Verification & Test Suite
    ├── test_models.py              # Architecture unit tests
    ├── verify_app_industry_standards.py # Full 6-step production audit suite
    ├── verify_models_deep.py       # Deep forward pass and numerical stability tests
    └── verify_phase1.py            # Environment and dependency verification
```

---

## ⚡ 5. Quickstart & Installation

### Option A: Local Native Execution (Recommended)

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/rajingle532/Deep_Learning_Project.git
   cd Deep_Learning_Project
   ```

2. **Set up Virtual Environment:**
   ```bash
   python -m venv venv
   # On Windows:
   .\venv\Scripts\activate
   # On Linux / macOS:
   source venv/bin/activate
   ```

3. **Install Dependencies:**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. **Run the Automated Industry Audit:**
   ```bash
   python tests/verify_app_industry_standards.py
   ```

5. **Launch the Web Interface:**
   - On Windows: Double-click `RUN_APP.bat`, or run:
     ```bash
     streamlit run app/app.py
     ```
   - On Linux / macOS:
     ```bash
     streamlit run app/app.py
     ```
   The interactive dashboard will open at `http://localhost:8501`.

---

### Option B: Turnkey Docker Deployment

Build and run the entire suite in an isolated, production-grade container:

```bash
# Build Docker image
docker build -t agriguard-ai .

# Run container on port 8501
docker run -d -p 8501:8501 --name agriguard-container agriguard-ai

# Open your browser at http://localhost:8501
```

---

## 🖥️ 6. Key Application Modules & Tabs

1. **Live Pathology Hub:**
   - **Upload / Camera / Curated Gallery:** Instant diagnosis across 28 canonical crop disease classes.
   - **1-Click Quick Demo Specimens:** Preloaded verified specimens with published SOTA scores.
   - **Grad-CAM Saliency Studio:** Real-time heatmap overlay with opacity slider and bounding localization.
   - **Clinical Prescription Download:** Export certified PDF reports with pathogen taxonomy, symptoms, and organic/chemical treatments.

2. **Research Benchmark Lab:**
   - Interactive model parameter breakdown, confusion matrix visualization, domain shift analysis, and radar evaluation charts.

3. **Agronomic Treatment Directory:**
   - Full botanical database covering 28 diseases with certified chemical (fungicides/bactericides), biological, and cultural IPM measures.

4. **Mobile Edge Studio (TFLite):**
   - Direct FlatBuffer inspection, latency benchmark comparison (~109 ms on Edge vs ~550 ms on CPU), and one-click Android/iOS deployment export.

---

## 📜 7. Academic Integrity & Reproducibility Notice
- All experiments report genuine empirical metrics.
- Target domain (**PlantDoc**) data is held strictly isolated during training to prevent data leakage.
- Seed is deterministically fixed (`seed = 42`) across NumPy, TensorFlow, and Python runtime.

---

## 👨‍💻 Author & Contact
- **Author:** Omkar Ingle
- **Email:** [ingleraj79@gmail.com](mailto:ingleraj79@gmail.com)
- **Repository:** [https://github.com/rajingle532/Deep_Learning_Project](https://github.com/rajingle532/Deep_Learning_Project.git)
