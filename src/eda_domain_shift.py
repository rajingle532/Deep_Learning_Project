"""
Phase 2: Exploratory Data Analysis (EDA) & Quantifying Domain Shift.
Domain-Generalized Deep Learning for Real-World Plant Disease Recognition.

Executes:
1. Class distribution & imbalance calculation with inverse-frequency class weights:
   w_c = N_total / (C * N_c)
2. Color space analysis (RGB and HSV channel distributions) proving covariate shift.
3. Low-level image statistics: Brightness, Contrast, and Laplacian Variance (blurriness).
4. Exports publication-quality figures to results/eda/ and class weights to data/processed/.
"""

import sys
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
import pandas as pd
from PIL import Image

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from src.taxonomy import CANONICAL_CLASSES, CANONICAL_DISPLAY_NAMES, NUM_CLASSES


def run_eda_pipeline():
    print("=" * 70)
    print("   PHASE 2: EXPLORATORY DATA ANALYSIS & DOMAIN SHIFT PROFILING    ")
    print("=" * 70)

    # 1. Output directories
    results_dir = PROJECT_ROOT / "results" / "eda"
    results_dir.mkdir(parents=True, exist_ok=True)
    processed_dir = PROJECT_ROOT / "data" / "processed"

    train_csv = processed_dir / "train_manifest.csv"
    val_csv = processed_dir / "val_manifest.csv"
    test_csv = processed_dir / "test_manifest.csv"

    df_train = pd.read_csv(train_csv)
    df_val = pd.read_csv(val_csv)
    df_test = pd.read_csv(test_csv)

    # Combine train + val for source domain profile
    df_source = pd.concat([df_train, df_val], ignore_index=True)
    df_target = df_test

    print(f"\n[1] Source Domain (PlantVillage): {len(df_source):,} images across {NUM_CLASSES} classes")
    print(f"    Target Domain (PlantDoc):     {len(df_target):,} images across {NUM_CLASSES} classes\n")

    # -------------------------------------------------------------
    # 1. CLASS IMBALANCE & CLASS WEIGHTS COMPUTATION
    # -------------------------------------------------------------
    print("[2] Computing Class Frequencies & Inverse-Frequency Class Weights...")
    source_counts = df_source["class_id"].value_counts().to_dict()
    total_samples = len(df_source)

    class_weights = {}
    class_stats = []

    for cid in range(NUM_CLASSES):
        cname = CANONICAL_CLASSES[cid]
        dname = CANONICAL_DISPLAY_NAMES[cname]
        n_c = source_counts.get(cid, 0)
        # Standard balanced class weight formula: w_c = N_total / (C * N_c)
        w_c = total_samples / (NUM_CLASSES * n_c) if n_c > 0 else 1.0
        class_weights[str(cid)] = round(float(w_c), 4)

        t_c = len(df_target[df_target["class_id"] == cid])
        class_stats.append({
            "class_id": cid,
            "canonical_name": cname,
            "display_name": dname,
            "source_count": n_c,
            "target_count": t_c,
            "class_weight": round(float(w_c), 4),
        })

    # Save class weights JSON
    weights_path = processed_dir / "class_weights.json"
    with open(weights_path, "w") as f:
        json.dump(class_weights, f, indent=2)
    print(f"  [SAVED] Class weights exported to: {weights_path}")

    # Plot Class Distribution Comparison (Horizontal Bar Chart)
    fig, ax = plt.subplots(figsize=(14, 11))
    df_stats = pd.DataFrame(class_stats)
    y = np.arange(NUM_CLASSES)
    bar_height = 0.38

    ax.barh(y - bar_height / 2, df_stats["source_count"], height=bar_height, label="PlantVillage (Source Lab)", color="#2E86AB")
    ax.barh(y + bar_height / 2, df_stats["target_count"], height=bar_height, label="PlantDoc (Target Real Field)", color="#E07A5F")

    ax.set_yticks(y)
    ax.set_yticklabels(df_stats["display_name"], fontsize=9)
    ax.set_xlabel("Number of Images (Log Scale)", fontsize=11, fontweight="bold")
    ax.set_xscale("log")
    ax.set_title("Class Frequency Comparison: Laboratory (PlantVillage) vs In-the-Wild (PlantDoc)", fontsize=13, fontweight="bold", pad=15)
    ax.grid(True, linestyle="--", alpha=0.5, which="both")
    ax.legend(loc="lower right", fontsize=11)
    plt.tight_layout()

    dist_chart_path = results_dir / "class_distribution_comparison.png"
    fig.savefig(dist_chart_path, dpi=200)
    plt.close(fig)
    print(f"  [SAVED] Distribution chart saved: {dist_chart_path}")

    # -------------------------------------------------------------
    # 2. COLOR SPACE & COVARIATE SHIFT ANALYSIS (RGB & HSV)
    # -------------------------------------------------------------
    print("\n[3] Profiling Color Space Covariate Shift (Sampling 1,000 images per domain)...")
    np.random.seed(42)

    sample_size = 1000
    source_sample = df_source.sample(n=min(sample_size, len(df_source)), random_state=42)
    target_sample = df_target.sample(n=min(sample_size, len(df_target)), random_state=42)

    def extract_color_profiles(df):
        r_vals, g_vals, b_vals = [], [], []
        h_vals, s_vals, v_vals = [], [], []

        for p in df["filepath"]:
            abs_p = PROJECT_ROOT / Path(p)
            img = cv2.imread(str(abs_p))
            if img is None:
                continue
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

            # Sample 200 random pixels per image for density estimation
            h, w, _ = img_rgb.shape
            ys = np.random.randint(0, h, size=200)
            xs = np.random.randint(0, w, size=200)

            rgb_samples = img_rgb[ys, xs]
            hsv_samples = img_hsv[ys, xs]

            r_vals.extend(rgb_samples[:, 0])
            g_vals.extend(rgb_samples[:, 1])
            b_vals.extend(rgb_samples[:, 2])

            h_vals.extend(hsv_samples[:, 0])
            s_vals.extend(hsv_samples[:, 1])
            v_vals.extend(hsv_samples[:, 2])

        return {
            "r": np.array(r_vals),
            "g": np.array(g_vals),
            "b": np.array(b_vals),
            "h": np.array(h_vals),
            "s": np.array(s_vals),
            "v": np.array(v_vals),
        }

    source_colors = extract_color_profiles(source_sample)
    target_colors = extract_color_profiles(target_sample)

    # Plot 2x3 Color Distribution Grid
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))

    channels = [
        ("r", "Red Channel", axes[0, 0], (0, 255), "red"),
        ("g", "Green Channel", axes[0, 1], (0, 255), "green"),
        ("b", "Blue Channel", axes[0, 2], (0, 255), "blue"),
        ("h", "Hue (0-179)", axes[1, 0], (0, 180), "purple"),
        ("s", "Saturation (0-255)", axes[1, 1], (0, 255), "orange"),
        ("v", "Value/Brightness (0-255)", axes[1, 2], (0, 255), "gray"),
    ]

    for key, title, ax, xlim, col in channels:
        sns.kdeplot(source_colors[key], ax=ax, label="PlantVillage (Lab)", color="#2E86AB", fill=True, alpha=0.35, linewidth=2)
        sns.kdeplot(target_colors[key], ax=ax, label="PlantDoc (Field)", color="#E07A5F", fill=True, alpha=0.35, linewidth=2)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlim(xlim)
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.set_ylabel("Density")
        ax.legend(fontsize=9)

    fig.suptitle("Empirical Proof of Covariate Shift: Color Space Profiles (RGB & HSV)", fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout()

    color_shift_path = results_dir / "color_space_domain_shift.png"
    fig.savefig(color_shift_path, dpi=200)
    plt.close(fig)
    print(f"  [SAVED] Color space shift curves: {color_shift_path}")

    # -------------------------------------------------------------
    # 3. IMAGE QUALITY & TEXTURE NOISE PROFILING (Blur, Contrast)
    # -------------------------------------------------------------
    print("\n[4] Profiling Image Texture, Contrast, and Blurriness (Laplacian Variance)...")

    def extract_image_quality_metrics(df):
        brightness_list = []
        contrast_list = []
        laplacian_list = []

        for p in df["filepath"]:
            abs_p = PROJECT_ROOT / Path(p)
            img = cv2.imread(str(abs_p))
            if img is None:
                continue
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Brightness = Mean grayscale intensity
            b = float(np.mean(gray))
            # Contrast = Standard deviation of pixel intensities
            c = float(np.std(gray))
            # Blurriness = Variance of the Laplacian operator
            lap = float(cv2.Laplacian(gray, cv2.CV_64F).var())

            brightness_list.append(b)
            contrast_list.append(c)
            laplacian_list.append(lap)

        return {
            "brightness": np.array(brightness_list),
            "contrast": np.array(contrast_list),
            "laplacian_var": np.array(laplacian_list),
        }

    source_metrics = extract_image_quality_metrics(source_sample)
    target_metrics = extract_image_quality_metrics(target_sample)

    # Plot 1x3 Image Quality Distributions
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Brightness
    sns.kdeplot(source_metrics["brightness"], ax=axes[0], label="PlantVillage (Lab)", color="#2E86AB", fill=True, alpha=0.4)
    sns.kdeplot(target_metrics["brightness"], ax=axes[0], label="PlantDoc (Field)", color="#E07A5F", fill=True, alpha=0.4)
    axes[0].set_title("Mean Image Brightness", fontsize=11, fontweight="bold")
    axes[0].set_xlabel("Mean Pixel Intensity (0-255)")
    axes[0].grid(True, linestyle="--", alpha=0.5)
    axes[0].legend()

    # Contrast
    sns.kdeplot(source_metrics["contrast"], ax=axes[1], label="PlantVillage (Lab)", color="#2E86AB", fill=True, alpha=0.4)
    sns.kdeplot(target_metrics["contrast"], ax=axes[1], label="PlantDoc (Field)", color="#E07A5F", fill=True, alpha=0.4)
    axes[1].set_title("Image Contrast (Pixel Std Dev)", fontsize=11, fontweight="bold")
    axes[1].set_xlabel("Standard Deviation")
    axes[1].grid(True, linestyle="--", alpha=0.5)
    axes[1].legend()

    # Laplacian Variance (Blurriness) - Use Log Scale
    sns.kdeplot(np.log1p(source_metrics["laplacian_var"]), ax=axes[2], label="PlantVillage (Lab)", color="#2E86AB", fill=True, alpha=0.4)
    sns.kdeplot(np.log1p(target_metrics["laplacian_var"]), ax=axes[2], label="PlantDoc (Field)", color="#E07A5F", fill=True, alpha=0.4)
    axes[2].set_title("Sharpness / Blur (Log Laplacian Variance)", fontsize=11, fontweight="bold")
    axes[2].set_xlabel("Log(1 + Laplacian Variance)")
    axes[2].grid(True, linestyle="--", alpha=0.5)
    axes[2].legend()

    fig.suptitle("Image Quality Comparison: Laboratory Controlled vs Real-World Field Noise", fontsize=13, fontweight="bold")
    plt.tight_layout()

    quality_chart_path = results_dir / "image_quality_shift.png"
    fig.savefig(quality_chart_path, dpi=200)
    plt.close(fig)
    print(f"  [SAVED] Image quality comparison: {quality_chart_path}")

    # -------------------------------------------------------------
    # 4. EXPORT STATISTICAL REPORT JSON & SUMMARY
    # -------------------------------------------------------------
    eda_summary = {
        "dataset_sizes": {
            "source_total": len(df_source),
            "source_train": len(df_train),
            "source_val": len(df_val),
            "target_test": len(df_target),
        },
        "color_profile_means": {
            "source_rgb_mean": [float(np.mean(source_colors["r"])), float(np.mean(source_colors["g"])), float(np.mean(source_colors["b"]))],
            "target_rgb_mean": [float(np.mean(target_colors["r"])), float(np.mean(target_colors["g"])), float(np.mean(target_colors["b"]))],
            "source_hsv_mean": [float(np.mean(source_colors["h"])), float(np.mean(source_colors["s"])), float(np.mean(source_colors["v"]))],
            "target_hsv_mean": [float(np.mean(target_colors["h"])), float(np.mean(target_colors["s"])), float(np.mean(target_colors["v"]))],
        },
        "image_quality_statistics": {
            "source": {
                "brightness_mean": round(float(np.mean(source_metrics["brightness"])), 2),
                "contrast_mean": round(float(np.mean(source_metrics["contrast"])), 2),
                "laplacian_var_mean": round(float(np.mean(source_metrics["laplacian_var"])), 2),
                "laplacian_var_median": round(float(np.median(source_metrics["laplacian_var"])), 2),
            },
            "target": {
                "brightness_mean": round(float(np.mean(target_metrics["brightness"])), 2),
                "contrast_mean": round(float(np.mean(target_metrics["contrast"])), 2),
                "laplacian_var_mean": round(float(np.mean(target_metrics["laplacian_var"])), 2),
                "laplacian_var_median": round(float(np.median(target_metrics["laplacian_var"])), 2),
            },
        },
    }

    summary_json_path = results_dir / "eda_summary.json"
    with open(summary_json_path, "w") as f:
        json.dump(eda_summary, f, indent=2)

    print("\n" + "=" * 70)
    print("                    EDA STATISTICAL SUMMARY                        ")
    print("=" * 70)
    print(f"  Metric                     PlantVillage (Lab)    PlantDoc (Field)")
    print(f"  -----------------------------------------------------------------")
    print(f"  Mean Brightness (0-255):   {eda_summary['image_quality_statistics']['source']['brightness_mean']:<21} {eda_summary['image_quality_statistics']['target']['brightness_mean']}")
    print(f"  Mean Contrast (Std Dev):   {eda_summary['image_quality_statistics']['source']['contrast_mean']:<21} {eda_summary['image_quality_statistics']['target']['contrast_mean']}")
    print(f"  Laplacian Var (Sharpness): {eda_summary['image_quality_statistics']['source']['laplacian_var_mean']:<21} {eda_summary['image_quality_statistics']['target']['laplacian_var_mean']}")
    print("=" * 70)
    print(f"\n>>> PHASE 2 COMPLETE! Figures & Class Weights generated successfully in results/eda/ <<<")


if __name__ == "__main__":
    run_eda_pipeline()

