import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch

def create_flowchart(output_path, dpi=300):
    # Dimensions: 11.8 inches wide by 5.6 inches high (~2.1:1 ratio)
    fig, ax = plt.subplots(figsize=(11.8, 5.6), dpi=dpi)
    ax.set_xlim(0, 11.8)
    ax.set_ylim(0, 5.6)
    ax.axis('off')
    
    # Pure white background, NO outer border, NO "Flow chart" label
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    # Styling
    box_edge = '#0F172A'       # Deep slate black
    box_lw = 1.4
    arrow_color = '#0F172A'
    arrow_lw = 1.35
    text_color = '#0F172A'

    # Center Y for the main flow
    cy = 2.80

    # Dimensions
    w_sq = 1.48
    h_sq = 1.85

    # Helper function to draw crisp nodes
    def draw_node(cx, cy_node, w, h, title, subtitle="", highlight=False):
        fc = '#FFFFFF'
        ec = '#0F172A'
        rect = FancyBboxPatch(
            (cx - w/2.0, cy_node - h/2.0), w, h,
            boxstyle="round,pad=0.03,rounding_size=0.08",
            facecolor=fc, edgecolor=ec, linewidth=box_lw, zorder=4
        )
        ax.add_patch(rect)
        
        if subtitle:
            ax.text(cx, cy_node + h*0.16, title, ha='center', va='center',
                    fontsize=8.5, fontweight='bold', color=text_color,
                    linespacing=1.25, zorder=5)
            ax.text(cx, cy_node - h*0.18, subtitle, ha='center', va='center',
                    fontsize=7.5, fontweight='normal', color='#334155',
                    linespacing=1.22, zorder=5)
        else:
            ax.text(cx, cy_node, title, ha='center', va='center',
                    fontsize=8.5, fontweight='bold', color=text_color,
                    linespacing=1.25, zorder=5)

    # 1. Col 1: Foliar Leaf Image
    cx1 = 1.05
    draw_node(cx1, cy, w_sq, h_sq,
              "Foliar Leaf Image",
              "(Field Photo / Camera /\nMobile Upload)")

    # 2. Col 2: Preprocessing
    cx2 = 3.05
    draw_node(cx2, cy, w_sq, h_sq,
              "Preprocessing",
              "(Adaptive Crop &\n224\u00d7224 Resizing,\nImageNet Norm.)")

    # 3. Col 3: Three parallel model comparison boxes (Real project models)
    cx3 = 5.25
    w_m = 1.82
    h_m = 1.40
    gap_y = 1.50

    # Top: Proposed Architecture (ResNet50 + MixStyle)
    draw_node(cx3, cy + gap_y, w_m, h_m,
              "Proposed Architecture:\nResNet50 + MixStyle",
              "(Domain Generalization\nvia Feature Statistics Mixing)")
    
    # Mid: ResNet50 ERM Baseline
    draw_node(cx3, cy, w_m, 1.25,
              "Deep Residual Baseline:\nResNet50 ERM",
              "(Empirical Risk Minimization)")
    
    # Bot: Lightweight Edge CNN
    draw_node(cx3, cy - gap_y, w_m, h_m,
              "Lightweight Edge CNN:\nMobileNetV2 / Custom",
              "(Depthwise Separable &\n4-Block Scratch Baseline)")

    # 4. Col 4: Plant Pathology Diagnosis Result
    cx4 = 7.45
    w_diag = 1.60
    h_diag = 1.95
    draw_node(cx4, cy, w_diag, h_diag,
              "Plant Pathology\nDiagnosis Result",
              "(28 Canonical Classes /\nCrop Prior Conditioning /\nOOD Entropy Gating)")

    # 5. Col 5: Explainable AI Module
    cx5 = 9.35
    w_xai = 1.50
    h_xai = 1.85
    draw_node(cx5, cy, w_xai, h_xai,
              "Explainable AI\nModule",
              "(Grad-CAM Saliency\nHeatmaps on\nconv5_block3_out)")

    # 6. Col 6: Agronomic Prescription & Edge Deployment
    cx6 = 10.95
    w_rx = 1.40
    h_rx = 1.85
    draw_node(cx6, cy, w_rx, h_rx,
              "Agronomic Advisory\n& Deployment",
              "(Certified Treatments\n&\nQuantized INT8 TFLite)")

    # ----------------- ARROWS -----------------
    arrow_style = dict(arrowstyle='->,head_width=0.36,head_length=0.42',
                       color=arrow_color, lw=arrow_lw)

    # Arrow 1: Col 1 -> Col 2
    x1_r = cx1 + w_sq/2.0
    x2_l = cx2 - w_sq/2.0
    ax.annotate('', xy=(x2_l, cy), xytext=(x1_r, cy), arrowprops=arrow_style)

    # Arrow 2: Col 2 -> Col 3 (Branching)
    x2_r = cx2 + w_sq/2.0
    x3_l = cx3 - w_m/2.0
    x_split = (x2_r + x3_l) / 2.0

    # Branch to Top (elbow)
    ax.plot([x2_r, x_split, x_split, x3_l - 0.02],
            [cy, cy, cy + gap_y, cy + gap_y],
            color=arrow_color, lw=arrow_lw, zorder=3)
    ax.annotate('', xy=(x3_l, cy + gap_y), xytext=(x3_l - 0.08, cy + gap_y),
                arrowprops=arrow_style)

    # Branch to Mid (straight)
    ax.annotate('', xy=(x3_l, cy), xytext=(x2_r, cy),
                arrowprops=arrow_style)

    # Branch to Bot (elbow)
    ax.plot([x2_r, x_split, x_split, x3_l - 0.02],
            [cy, cy, cy - gap_y, cy - gap_y],
            color=arrow_color, lw=arrow_lw, zorder=3)
    ax.annotate('', xy=(x3_l, cy - gap_y), xytext=(x3_l - 0.08, cy - gap_y),
                arrowprops=arrow_style)

    # Arrow 3: Col 3 -> Col 4 (Converging)
    x3_r = cx3 + w_m/2.0
    x4_l = cx4 - w_diag/2.0
    x_merge = (x3_r + x4_l) / 2.0

    # Converge from Top (elbow into merge point)
    ax.plot([x3_r, x_merge, x_merge],
            [cy + gap_y, cy + gap_y, cy],
            color=arrow_color, lw=arrow_lw, zorder=3)

    # Middle line to merge point
    ax.plot([x3_r, x_merge], [cy, cy],
            color=arrow_color, lw=arrow_lw, zorder=3)

    # Converge from Bot (elbow into merge point)
    ax.plot([x3_r, x_merge, x_merge],
            [cy - gap_y, cy - gap_y, cy],
            color=arrow_color, lw=arrow_lw, zorder=3)

    # Single arrowhead entering Col 4 from merge point
    ax.annotate('', xy=(x4_l, cy), xytext=(x_merge, cy),
                arrowprops=arrow_style)

    # Arrow 4: Col 4 -> Col 5
    x4_r = cx4 + w_diag/2.0
    x5_l = cx5 - w_xai/2.0
    ax.annotate('', xy=(x5_l, cy), xytext=(x4_r, cy), arrowprops=arrow_style)

    # Arrow 5: Col 5 -> Col 6
    x5_r = cx5 + w_xai/2.0
    x6_l = cx6 - w_rx/2.0
    ax.annotate('', xy=(x6_l, cy), xytext=(x5_r, cy), arrowprops=arrow_style)

    plt.tight_layout(pad=0.2)
    plt.savefig(output_path, dpi=dpi, facecolor='white', bbox_inches='tight')
    plt.close()
    print(f"Generated {output_path} successfully at {dpi} DPI.")

if __name__ == '__main__':
    from pathlib import Path
    proj_root = Path(__file__).resolve().parent.parent
    fig_dir = proj_root / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    out_file = fig_dir / "fig1_flowchart.png"
    create_flowchart(str(out_file), dpi=300)
