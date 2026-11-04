import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from matplotlib.lines import Line2D

# Set font and visual parameters for publication-grade IEEE figures
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.edgecolor'] = '#333333'
plt.rcParams['axes.linewidth'] = 1.0

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FIGURES_DIR = PROJECT_ROOT / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
BRAIN_DIR = Path("C:/Users/ingle/.gemini/antigravity/brain/b4c668e0-a6bd-4c39-96bc-c60c655a72cb")


# ==============================================================================
# FIG 2: PERCENTAGE SCORE AMONG MODELS
# ==============================================================================
def generate_fig2():
    models = [
        "Custom CNN\n(4-Block)",
        "MobileNetV2\n(Edge)",
        "ResNet50 ERM\n(Transfer)",
        "Proposed\nResNet50+MixStyle",
        "Quantized\nINT8 TFLite"
    ]
    
    acc = [87.40, 96.25, 98.42, 99.37, 99.12]
    prec = [86.85, 95.80, 98.15, 99.32, 99.05]
    rec = [86.50, 95.92, 98.20, 99.35, 99.10]
    f1 = [86.67, 95.86, 98.17, 99.33, 99.07]
    
    x = np.arange(len(models))
    width = 0.19
    
    fig, ax = plt.subplots(figsize=(8.5, 4.8), dpi=300)
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    
    c_acc = '#1f77b4'   # Blue
    c_prec = '#ff7f0e'  # Orange
    c_rec = '#2ca02c'   # Green
    c_f1 = '#d62728'    # Red
    
    rects1 = ax.bar(x - 1.5*width, acc, width, label='Accuracy (%)', color=c_acc, edgecolor='#111', lw=0.6)
    rects2 = ax.bar(x - 0.5*width, prec, width, label='Precision (%)', color=c_prec, edgecolor='#111', lw=0.6)
    rects3 = ax.bar(x + 0.5*width, rec, width, label='Recall (%)', color=c_rec, edgecolor='#111', lw=0.6)
    rects4 = ax.bar(x + 1.5*width, f1, width, label='F1-Score (%)', color=c_f1, edgecolor='#111', lw=0.6)
    
    ax.set_title("Model Performance Comparison (28 Canonical Classes)", fontsize=11, fontweight='bold', pad=12)
    ax.set_ylabel("Percentage (%)", fontsize=10, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=8.5)
    ax.set_ylim(80, 103.5)
    ax.grid(axis='y', linestyle='--', alpha=0.5, color='#ccc')
    ax.set_axisbelow(True)
    
    # Value annotations on bars
    for rects in [rects1, rects2, rects3, rects4]:
        for rect in rects:
            h = rect.get_height()
            ax.annotate(f"{h:.1f}",
                        xy=(rect.get_x() + rect.get_width() / 2, h),
                        xytext=(0, 2.5), textcoords="offset points",
                        ha='center', va='bottom', fontsize=6.8, rotation=90)
            
    # Position legend in upper left where Custom CNN leaves generous whitespace
    ax.legend(loc='upper left', frameon=True, facecolor='white', edgecolor='#ccc', fontsize=8.5)
    
    plt.tight_layout()
    for d in [FIGURES_DIR]:
        plt.savefig(d / "fig2_percentage_scores.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("[SAVED] Fig 2 updated with zero overlap.")


# ==============================================================================
# FIG 3: PERFORMANCE TRENDS AMONG DIFFERENT DATASETS & SHIFT LEVELS
# ==============================================================================
def generate_fig3():
    datasets = [
        "PlantVillage\n(Train Split)",
        "PlantVillage\n(Validation)",
        "Solanaceae\n(Consensus)",
        "PlantDoc Field\n(Top-3 Transfer)",
        "PlantDoc Field\n(Top-1 Zero-Shot)"
    ]
    
    acc = [99.60, 99.37, 79.88, 36.88, 22.50]
    prec = [99.50, 99.32, 78.50, 32.40, 18.79]
    rec = [99.60, 99.35, 80.20, 38.10, 22.32]
    f1 = [99.50, 99.33, 79.34, 34.98, 16.24]
    
    diff_scores = [1.0, 1.0, 2.0, 3.2, 4.0]
    
    x = np.arange(len(datasets))
    width = 0.18
    
    fig, ax1 = plt.subplots(figsize=(9.2, 4.8), dpi=300)
    fig.patch.set_facecolor('white')
    ax1.set_facecolor('white')
    
    c_acc = '#1f77b4'
    c_prec = '#ff7f0e'
    c_rec = '#2ca02c'
    c_f1 = '#d62728'
    
    ax1.bar(x - 1.5*width, acc, width, label='Accuracy (%)', color=c_acc, edgecolor='#111', lw=0.6)
    ax1.bar(x - 0.5*width, prec, width, label='Precision (%)', color=c_prec, edgecolor='#111', lw=0.6)
    ax1.bar(x + 0.5*width, rec, width, label='Recall (%)', color=c_rec, edgecolor='#111', lw=0.6)
    ax1.bar(x + 1.5*width, f1, width, label='F1-Score (%)', color=c_f1, edgecolor='#111', lw=0.6)
    
    ax1.set_title("Dataset Performance vs. Environmental Domain Shift Level", fontsize=11, fontweight='bold', pad=12)
    ax1.set_ylabel("Performance Score (%)", fontsize=10, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(datasets, fontsize=8.5)
    ax1.set_ylim(0, 115)
    ax1.grid(axis='y', linestyle='--', alpha=0.5, color='#ccc')
    ax1.set_axisbelow(True)
    
    # Secondary axis for Difficulty / Domain Shift
    ax2 = ax1.twinx()
    ax2.plot(x, diff_scores, color='#8b008b', marker='o', markersize=7, lw=2.2, linestyle='--', label='Environmental Shift Level')
    ax2.set_ylabel("Domain Shift Severity", color='#8b008b', fontsize=10, fontweight='bold')
    ax2.set_ylim(0.5, 4.5)
    ax2.set_yticks([1.0, 2.0, 3.0, 4.0])
    ax2.set_yticklabels(['Low (Lab)', 'Moderate', 'High', 'Very High (Field)'], fontsize=8.5, color='#8b008b')
    ax2.tick_params(axis='y', colors='#8b008b')
    
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right', frameon=True, facecolor='white', edgecolor='#ccc', fontsize=8.0)
    
    plt.tight_layout()
    for d in [FIGURES_DIR]:
        plt.savefig(d / "fig3_domain_performance.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("[SAVED] Fig 3 updated.")


# ==============================================================================
# FIG 4: MODEL PERFORMANCE VS. INTERPRETABILITY LANDSCAPE
# ==============================================================================
def generate_fig4():
    fig, ax = plt.subplots(figsize=(8.0, 5.2), dpi=300)
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    
    models = [
        {"name": "Custom CNN", "acc": 87.40, "f1": 86.67, "level": "Moderate", "c": "#ff7f0e", "s": 150},
        {"name": "MobileNetV2", "acc": 96.25, "f1": 95.86, "level": "Moderate", "c": "#ff7f0e", "s": 170},
        {"name": "ResNet50 ERM", "acc": 98.42, "f1": 98.17, "level": "High", "c": "#2ca02c", "s": 210},
        {"name": "Quantized INT8", "acc": 99.12, "f1": 99.07, "level": "High", "c": "#2ca02c", "s": 190},
        {"name": "Proposed: ResNet50 + MixStyle", "acc": 99.37, "f1": 99.33, "level": "Very High", "c": "#d62728", "s": 280},
    ]
    
    # Gray reference trend line (Pareto frontier)
    x_line = np.linspace(85, 100.5, 100)
    y_line = x_line - 0.2
    ax.plot(x_line, y_line, color='#cccccc', lw=2.5, linestyle='-', zorder=1)
    ax.text(91.5, 92.6, "Higher Performance $\\rightarrow$\n(Pareto Frontier)", color='#888888',
            fontsize=8.5, rotation=40, ha='center', va='center', style='italic')
    
    for m in models:
        ax.scatter(m["acc"], m["f1"], color=m["c"], s=m["s"], edgecolors='#111', lw=1.2, zorder=4)
        
    # Uncluttered, perfectly spaced annotations
    ax.annotate("Custom CNN (4-Block)\n(87.4%, 86.7%)", xy=(87.40, 86.67), xytext=(88.4, 85.8),
                fontsize=8, fontweight='bold', arrowprops=dict(arrowstyle="->", color='#444', lw=0.9))
    
    # MobileNetV2 text placed below the line
    ax.annotate("MobileNetV2\n(96.3%, 95.9%)", xy=(96.25, 95.86), xytext=(96.25, 93.6),
                ha='center', fontsize=8, fontweight='bold', arrowprops=dict(arrowstyle="->", color='#444', lw=0.9))
    
    # ResNet50 ERM text placed to the top-left
    ax.annotate("ResNet50 ERM\n(98.4%, 98.2%)", xy=(98.42, 98.17), xytext=(95.2, 98.8),
                ha='right', fontsize=8, fontweight='bold', arrowprops=dict(arrowstyle="->", color='#444', lw=0.9))
    
    # Quantized INT8 text placed to the bottom-right
    ax.annotate("Quantized INT8 TFLite\n(99.1%, 99.1%)", xy=(99.12, 99.07), xytext=(99.6, 96.5),
                ha='left', fontsize=8, fontweight='bold', arrowprops=dict(arrowstyle="->", color='#444', lw=0.9))
    
    # Proposed model highlighted at the top
    ax.annotate("Proposed: ResNet50 + MixStyle\n(99.37%, 99.33%) [Very High]", xy=(99.37, 99.33), xytext=(93.0, 100.4),
                fontsize=8.5, fontweight='bold', color='#b91c1c',
                bbox=dict(boxstyle="round,pad=0.25", fc="#fef2f2", ec="#b91c1c", lw=1.1),
                arrowprops=dict(arrowstyle="->", color='#b91c1c', lw=1.2))
    
    ax.set_title("Model Performance vs. Interpretability Landscape", fontsize=11, fontweight='bold', pad=12)
    ax.set_xlabel("Accuracy (%)", fontsize=10, fontweight='bold')
    ax.set_ylabel("F1-Score (%)", fontsize=10, fontweight='bold')
    ax.set_xlim(85, 101.5)
    ax.set_ylim(85, 101.8)
    ax.grid(True, linestyle='--', alpha=0.5, color='#ccc')
    
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='Very High (Grad-CAM + Prior + OOD)', markerfacecolor='#d62728', markersize=10, markeredgecolor='#111'),
        Line2D([0], [0], marker='o', color='w', label='High (Grad-CAM Saliency)', markerfacecolor='#2ca02c', markersize=9, markeredgecolor='#111'),
        Line2D([0], [0], marker='o', color='w', label='Moderate (Feature Maps)', markerfacecolor='#ff7f0e', markersize=8, markeredgecolor='#111'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', frameon=True, facecolor='white', edgecolor='#ccc', fontsize=8.0, title="Interpretability Level")
    
    plt.tight_layout()
    for d in [FIGURES_DIR]:
        plt.savefig(d / "fig4_performance_vs_interpretability.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("[SAVED] Fig 4 updated with clean layout.")


# ==============================================================================
# FIG 5: COMPREHENSIVE MODEL EVALUATION (RADAR CHART)
# ==============================================================================
def generate_fig5():
    categories = ['Precision', 'Accuracy', 'F1-Score', 'Recall']
    N = len(categories)
    
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(6.5, 6.0), subplot_kw=dict(polar=True), dpi=300)
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    
    plt.xticks(angles[:-1], categories, fontsize=10, fontweight='bold', color='#111')
    ax.set_rlabel_position(0)
    plt.yticks([85, 90, 95, 100], ["85%", "90%", "95%", "100%"], color="#666", size=8)
    plt.ylim(80, 101)
    
    model_data = [
        {"name": "Custom CNN (4-Block)", "vals": [86.85, 87.40, 86.67, 86.50], "color": "#1f77b4", "lw": 1.6},
        {"name": "MobileNetV2 (Edge)", "vals": [95.80, 96.25, 95.86, 95.92], "color": "#ff7f0e", "lw": 1.6},
        {"name": "ResNet50 ERM", "vals": [98.15, 98.42, 98.17, 98.20], "color": "#2ca02c", "lw": 1.8},
        {"name": "Proposed: ResNet50 + MixStyle", "vals": [99.32, 99.37, 99.33, 99.35], "color": "#d62728", "lw": 2.2},
        {"name": "Quantized INT8 TFLite", "vals": [99.05, 99.12, 99.07, 99.10], "color": "#9467bd", "lw": 1.6, "ls": "--"},
    ]
    
    for m in model_data:
        v = m["vals"] + m["vals"][:1]
        ls = m.get("ls", "-")
        ax.plot(angles, v, linewidth=m["lw"], linestyle=ls, label=m["name"], color=m["color"])
        ax.fill(angles, v, color=m["color"], alpha=0.08)
        
    ax.set_title("Comprehensive Model Evaluation (Radar Chart)", fontsize=11, fontweight='bold', pad=18)
    ax.grid(True, linestyle='--', color='#bbb', alpha=0.7)
    
    plt.legend(loc='upper right', bbox_to_anchor=(1.32, 0.18), frameon=True, facecolor='white', edgecolor='#ccc', fontsize=7.8)
    
    plt.tight_layout()
    for d in [FIGURES_DIR]:
        plt.savefig(d / "fig5_radar_evaluation.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("[SAVED] Fig 5 updated.")


if __name__ == '__main__':
    print("Re-generating perfected figures...")
    generate_fig2()
    generate_fig3()
    generate_fig4()
    generate_fig5()
    print("All figures regenerated at 300 DPI!")
