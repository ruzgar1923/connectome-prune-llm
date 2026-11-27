"""
Publication-Quality Figure Generator: Fig 1 Layer-Wise Saliency Ratio
=====================================================================
Generates camera-ready vector PDF and PNG for NeurIPS / ICLR submission:
- Top: Absolute Saliency Density (Attention vs MLP)
- Bottom: Relative Ratio psi_l = S_attn / S_mlp (> 1.0 everywhere, peak ~2.28x)

Author: Rüzgar Vatansever
"""

import os
import json
import matplotlib.pyplot as plt
import numpy as np

def main():
    json_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "layer_saliency_analysis.json")
    if not os.path.exists(json_path):
        print(f"[ERROR] {json_path} not found!")
        return

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    layers = [d["layer"] for d in data]
    attn_sal = [d["attn_mean_saliency"] for d in data]
    mlp_sal = [d["mlp_mean_saliency"] for d in data]
    ratios = [d["attn_to_mlp_ratio"] for d in data]

    # Create figure with 2 subplots
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Helvetica", "Arial"]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.2, 5.2), sharex=True, dpi=300)

    # Palette
    c_attn = "#2563EB"   # Vivid Blue
    c_mlp = "#64748B"    # Slate Grey
    c_ratio = "#0D9488"  # Teal / Dark Cyan
    c_parity = "#DC2626" # Crimson parity line

    # Subplot 1: Absolute Saliency Density
    ax1.plot(layers, attn_sal, marker="o", markersize=5, color=c_attn, linewidth=2.0, label="Attention Projections ($W_q, W_k, W_v, W_o$)")
    ax1.plot(layers, mlp_sal, marker="s", markersize=5, color=c_mlp, linewidth=2.0, linestyle="--", label="MLP Projections ($W_{gate}, W_{up}, W_{down}$)")
    ax1.fill_between(layers, attn_sal, mlp_sal, color=c_attn, alpha=0.12, label="Excess Attention Saliency Density")
    ax1.set_ylabel(r"Mean Saliency ($|W| \cdot \|X\|_2$)", fontsize=9.5, fontweight="bold")
    ax1.set_title("(a) Mean Activation-Weighted Saliency Density per Layer (LLaMA-3.2-1B)", fontsize=10.5, fontweight="bold", pad=8)
    ax1.grid(True, linestyle=":", alpha=0.6)
    # Position legend in upper right where curves descend to lowest values, eliminating any overlap
    ax1.set_ylim(0.015, 0.165)
    ax1.legend(loc="upper right", frameon=True, fontsize=8.0, framealpha=0.92, ncol=1)

    # Subplot 2: Relative Ratio
    bars = ax2.bar(layers, ratios, color=c_ratio, width=0.6, alpha=0.85, edgecolor="#042F2E", linewidth=0.8, label=r"$\psi_l = S_{attn} / S_{mlp}$")
    ax2.axhline(1.0, color=c_parity, linestyle="-.", linewidth=1.5, label="Parity Threshold (1.0x)")
    
    # Annotate peak ratio
    max_idx = np.argmax(ratios)
    ax2.annotate(f"Peak: {ratios[max_idx]:.2f}x\n(Layer {layers[max_idx]})",
                 xy=(layers[max_idx], ratios[max_idx]),
                 xytext=(layers[max_idx] + 0.8, ratios[max_idx] + 0.12),
                 arrowprops=dict(facecolor="#0F172A", shrink=0.08, width=1.0, headwidth=5),
                 fontsize=8.0, fontweight="bold", color="#0F172A")

    ax2.set_xlabel(r"Transformer Layer Index $l \in [0, 15]$", fontsize=9.5, fontweight="bold")
    ax2.set_ylabel(r"Saliency Ratio $\psi_l$", fontsize=9.5, fontweight="bold")
    ax2.set_title(r"(b) Information Density Advantage of Routing Heads ($\psi_l > 1.0\times$ Everywhere)", fontsize=10.5, fontweight="bold", pad=8)
    ax2.set_xticks(layers)
    ax2.set_ylim(0.8, 2.6)
    ax2.grid(True, linestyle=":", alpha=0.6, axis="y")
    ax2.legend(loc="upper right", frameon=True, fontsize=8.2, framealpha=0.92)

    plt.tight_layout()

    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
    os.makedirs(out_dir, exist_ok=True)
    pdf_out = os.path.join(out_dir, "fig1_layer_saliency_ratio.pdf")
    png_out = os.path.join(out_dir, "fig1_layer_saliency_ratio.png")

    plt.savefig(pdf_out, bbox_inches="tight")
    plt.savefig(png_out, bbox_inches="tight", dpi=300)
    plt.close()

    print(f"[SUCCESS] Figure 1 generated:")
    print(f"  -> Vector PDF: {pdf_out}")
    print(f"  -> PNG Image:  {png_out}")

if __name__ == "__main__":
    main()
