"""
Builds complete multi-page NeurIPS/ICLR formatted PDF for MAKALELER directory.
Now includes:
- Embedded Figure 1 (Layer-wise Saliency Ratio with non-overlapping legend)
- Table 1 (Main Multi-Method Benchmark with 95% Wilson Score CIs)
- Table 2 (Integration with OWL and honest scaling interaction analysis)
- Cross-Model Generalization on Qwen2.5-1.5B
- High-Sparsity 40% and 50% Stress Test
"""

import os
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable, Image, KeepTogether
from reportlab.lib import colors

def build_pdf(filename):
    doc = SimpleDocTemplate(
        filename,
        pagesize=letter,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36
    )

    styles = getSampleStyleSheet()
    
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=12.5,
        leading=16.0,
        alignment=1,
        textColor=colors.HexColor('#0F172A')
    )

    author_style = ParagraphStyle(
        'AuthorStyle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=9.5,
        leading=12.0,
        alignment=1,
        textColor=colors.HexColor('#0F172A')
    )

    affil_style = ParagraphStyle(
        'AffilStyle',
        parent=styles['Normal'],
        fontName='Helvetica-Oblique',
        fontSize=7.8,
        leading=10.2,
        alignment=1,
        textColor=colors.HexColor('#475569')
    )

    h1_style = ParagraphStyle(
        'Heading1_Custom',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=9.0,
        leading=11.8,
        textColor=colors.HexColor('#0F172A'),
        spaceBefore=4,
        spaceAfter=2
    )

    body_style = ParagraphStyle(
        'Body_Custom',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=7.4,
        leading=9.8,
        textColor=colors.HexColor('#1E293B'),
        spaceAfter=2.0
    )

    cell_style = ParagraphStyle(
        'CellStyle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=6.1,
        leading=8.0,
        textColor=colors.HexColor('#1E293B')
    )

    cell_bold = ParagraphStyle(
        'CellBold',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=6.1,
        leading=8.0,
        textColor=colors.HexColor('#0F172A')
    )

    cell_header = ParagraphStyle(
        'CellHeader',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=6.1,
        leading=8.0,
        textColor=colors.white
    )

    story = []

    # Title
    story.append(Paragraph("ConnectomePrune: Attention-Protective Non-Uniform Sparsification for Compact Language Models", title_style))
    story.append(Spacer(1, 2.5))

    # Author
    story.append(Paragraph("Ruzgar Vatansever", author_style))
    story.append(Paragraph("Independent Researcher, Sehit Ogretmen Necmeddin Kuyucu Anadolu Lisesi, Gebze, Kocaeli, Turkey", affil_style))
    story.append(Paragraph("Email: rzvatansevertr@gmail.com | ORCID: 0009-0004-9477-8422", affil_style))
    story.append(Paragraph("Code Repository: https://github.com/ruzgar1923/connectome-prune-llm", affil_style))
    story.append(Spacer(1, 2.5))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor('#CBD5E1'), spaceAfter=3))

    # Abstract
    story.append(Paragraph("<b>Abstract</b>—Deploying compact foundation language models on edge hardware requires aggressive parameter compression without destroying core reasoning representations. While empirical weight-activation pruning (e.g., Wanda) mitigates the destruction of magnitude outliers, existing implementations enforce uniform sparsity layer-by-layer, disproportionately degrading fragile multi-head self-attention routing channels. Motivated by structural heterogeneity in biological connectomics and homeostatic synaptic scaling, we present <b>ConnectomePrune-LLM</b>. Our framework introduces: (1) <i>Attention-Protective Budget Allocation</i>, formulating an iso-sparsity constrained allocation that shields high-saliency attention projections while shifting sparsity onto feed-forward MLP blocks (alpha<sub>attn</sub> = 18.75%, alpha<sub>mlp</sub> = 28.75% on LLaMA-3.2-1B); (2) <i>Output-Selective Homeostatic Scaling</i> (gamma = 1/sqrt(rho)), applied strictly to linear projection outputs (W<sub>o</sub> and W<sub>down</sub>) while bypassing query/key/gate matrices to preserve softmax temperature and prevent non-linear SiLU distortion; and (3) <i>ConnectomeSculptor</i>, an in-place sequential safetensor carving tool requiring &lt; 3.5 GB host RAM during offline pruning. Empirical evaluations on Meta LLaMA-3.2-1B and Qwen2.5-1.5B confirm that ConnectomePrune preserves 11.48 WikiText-2 perplexity (near dense 11.24), outperforming standard Wanda (12.85), SparseGPT (12.15), and OWL (11.84) while maintaining 100% syntax validity on code generation (95% Wilson CI: [92.9%, 100.0%]). At high sparsity (40%), ConnectomePrune outperforms Wanda by over 10.5 perplexity points (119.03 vs. 129.58). Furthermore, integrating attention protection into OWL's depth allocation yields a 4.43 PPL reduction (105.42 to 100.99), demonstrating that routing protection remains effective across non-uniform depth allocations. Layer-wise analysis indicates that attention routing projections maintain 1.3x to 2.28x higher activation-weighted saliency density than MLP blocks across transformer layers.", body_style))
    story.append(Spacer(1, 2.5))

    # Section I & II
    story.append(Paragraph("I. INTRODUCTION & METHODOLOGY", h1_style))
    story.append(Paragraph("In compact foundation models, linear parameters partition between Multi-Head Attention (17.2% in LLaMA-3.2-1B) and feed-forward MLP blocks (82.8%). Uniform pruning treats both modules identically, inadvertently damaging sensitive token routing channels. ConnectomePrune introduces non-uniform attention protection grounded in structural heterogeneity and homeostatic synaptic scaling.<br/>"
                           "<b>Mathematical Invariants:</b> Multiplicative variance restoration gamma = 1/sqrt(rho) applied exclusively to W<sub>o</sub> and W<sub>down</sub> leaves W<sub>q</sub>, W<sub>k</sub> unscaled, preserving softmax temperature, and leaves W<sub>gate</sub>, W<sub>up</sub> unscaled, avoiding cubic distortion in gated SiLU activations. Because Wanda prunes weights with the smallest saliency, gamma acts as an upper-bound compensatory heuristic.", body_style))

    # Figure 1
    fig_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures", "fig1_layer_saliency_ratio.png")
    if os.path.exists(fig_path):
        story.append(Spacer(1, 2))
        story.append(Image(fig_path, width=470, height=270))
        story.append(Paragraph("<b>Figure 1: Mechanistic Saliency Density Analysis across all 16 Layers of LLaMA-3.2-1B.</b> Top: Absolute activation-weighted saliency density for Attention vs. MLP projections. Bottom: Saliency ratio psi<sub>l</sub> = S<sub>attn</sub> / S<sub>mlp</sub>, showing attention routing channels carry 1.3x to 2.28x higher activation-weighted density across the model. <i>Caveat:</i> W<sub>down</sub> receives intermediate activations while attention receives RMSNorm-stabilized states; functional routing sensitivity is causally demonstrated through the Inverse-Hub experiment.", affil_style))
        story.append(Spacer(1, 2.5))

    # Section III: Main Results Table
    story.append(Paragraph("II. MAIN MULTI-METHOD BENCHMARK (REGIME A: FULL LONG-CONTEXT)", h1_style))
    table_data = [
        [Paragraph("Pruning Method", cell_header), 
         Paragraph("Linear Sparsity", cell_header), 
         Paragraph("WikiText-2 PPL (512 / 2k)", cell_header), 
         Paragraph("Code Syntax (50) [95% CI]", cell_header), 
         Paragraph("Reasoning Parity (200)", cell_header)],
        [Paragraph("Dense Baseline (Meta 1B)", cell_style), Paragraph("0.00%", cell_style), Paragraph("11.24 +/- 0.02 / 10.70", cell_style), Paragraph("100.0% (50/50) [92.9, 100.0]", cell_style), Paragraph("100.0% (200/200)", cell_style)],
        [Paragraph("Magnitude Pruning", cell_style), Paragraph("27.10%", cell_style), Paragraph("184.60 (Collapsed)", cell_style), Paragraph("0.0% (0/50) [0.0, 7.1]", cell_style), Paragraph("0.0% (Collapsed)", cell_style)],
        [Paragraph("Standard Wanda (2023)", cell_style), Paragraph("27.10%", cell_style), Paragraph("12.85 +/- 0.08", cell_style), Paragraph("80.0% (40/50) [67.0, 88.8]", cell_style), Paragraph("95.0% (190/200)", cell_style)],
        [Paragraph("Wanda + Scaling Only", cell_style), Paragraph("27.10%", cell_style), Paragraph("12.31 +/- 0.07", cell_style), Paragraph("88.0% (44/50) [76.2, 94.4]", cell_style), Paragraph("97.0% (194/200)", cell_style)],
        [Paragraph("SparseGPT (2023)", cell_style), Paragraph("27.10%", cell_style), Paragraph("12.15 +/- 0.07", cell_style), Paragraph("92.0% (46/50) [81.2, 96.9]", cell_style), Paragraph("98.0% (196/200)", cell_style)],
        [Paragraph("SparseGPT + Scaling", cell_style), Paragraph("27.10%", cell_style), Paragraph("11.72 +/- 0.06", cell_style), Paragraph("94.0% (47/50) [83.8, 98.2]", cell_style), Paragraph("98.5% (197/200)", cell_style)],
        [Paragraph("OWL (Yin et al., 2024)", cell_style), Paragraph("27.10%", cell_style), Paragraph("11.84 +/- 0.06", cell_style), Paragraph("96.0% (48/50) [86.5, 99.5]", cell_style), Paragraph("98.5% (197/200)", cell_style)],
        [Paragraph("OWL + Scaling", cell_style), Paragraph("27.10%", cell_style), Paragraph("11.56 +/- 0.05", cell_style), Paragraph("98.0% (49/50) [89.5, 99.9]", cell_style), Paragraph("99.0% (198/200)", cell_style)],
        [Paragraph("Wanda + Hub Protection Only", cell_style), Paragraph("27.10%", cell_style), Paragraph("11.92 +/- 0.05", cell_style), Paragraph("96.0% (48/50) [86.5, 99.5]", cell_style), Paragraph("98.5% (197/200)", cell_style)],
        [Paragraph("<b>Inverse-Hub Control (Attn 37.5%, MLP 24.9%)</b>", cell_bold), Paragraph("27.07%", cell_style), Paragraph("<b>66.10 (Severe Drift)</b>", cell_bold), Paragraph("42.0% (21/50) [29.4, 55.8]", cell_style), Paragraph("42.0% (84/200)", cell_style)],
        [Paragraph("<b>ConnectomePrune (Ours: Hub+Scaling)</b>", cell_bold), Paragraph("<b>27.10%</b>", cell_bold), Paragraph("<b>11.48 +/- 0.04 / 10.92</b>", cell_bold), Paragraph("<b>100.0% (50/50) [92.9, 100.0]</b>", cell_bold), Paragraph("<b>100.0% (200/200)</b>", cell_bold)]
    ]

    t1 = Table(table_data, colWidths=[120, 38, 110, 112, 110])
    t1.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#0F172A')),
        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#CBD5E1')),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor('#F8FAFC')]),
        ('TOPPADDING', (0,0), (-1,-1), 1.2),
        ('BOTTOMPADDING', (0,0), (-1,-1), 1.2),
    ]))
    story.append(t1)
    story.append(Spacer(1, 2.5))

    # Section IV: Table 2 OWL Ablation & Qwen
    story.append(Paragraph("III. ORTHOGONAL OWL INTEGRATION (REGIME B: ZERO-SHOT STRESS SUITE)", h1_style))
    
    owl_data = [
        [Paragraph("Configuration (LLaMA-3.2-1B)", cell_header), Paragraph("Linear Sparsity", cell_header), Paragraph("Zero-Shot Stress PPL (512 ctx)", cell_header), Paragraph("Orthogonal Synergy & Analysis", cell_header)],
        [Paragraph("Standard OWL (ICML 2024)", cell_style), Paragraph("27.17%", cell_style), Paragraph("105.42", cell_style), Paragraph("Inter-layer depth allocation alone", cell_style)],
        [Paragraph("OWL + Scaling Only", cell_style), Paragraph("27.17%", cell_style), Paragraph("111.89", cell_style), Paragraph("Degrades PPL (outlier skew interaction)", cell_style)],
        [Paragraph("<b>OWL + Hub Protection Only</b>", cell_bold), Paragraph("<b>27.12%</b>", cell_bold), Paragraph("<b>100.99</b>", cell_bold), Paragraph("<b>-4.43 PPL Drop (Isolated Hub Benefit)</b>", cell_bold)],
        [Paragraph("OWL + Hub + Scaling (Unified)", cell_style), Paragraph("27.12%", cell_style), Paragraph("104.43", cell_style), Paragraph("Depth + Intra-layer allocation", cell_style)],
        [Paragraph("<b>ConnectomePrune (Iso-Sparsity)</b>", cell_bold), Paragraph("<b>27.09%</b>", cell_bold), Paragraph("<b>104.69</b>", cell_bold), Paragraph("<b>Closed-form (no depth sweep required)</b>", cell_bold)]
    ]
    t2 = Table(owl_data, colWidths=[140, 70, 110, 170])
    t2.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#0F172A')),
        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#CBD5E1')),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor('#F8FAFC')]),
        ('TOPPADDING', (0,0), (-1,-1), 1.2),
        ('BOTTOMPADDING', (0,0), (-1,-1), 1.2),
    ]))
    story.append(t2)
    story.append(Spacer(1, 2.5))

    story.append(Paragraph("<b>A. 40% & 50% High-Sparsity Regimes:</b> Under the 512-token stress protocol, Standard Wanda degrades to 129.58 PPL at 40% sparsity, while ConnectomePrune maintains 119.03 PPL (a 10.55 PPL advantage). At 50% sparsity, ConnectomePrune retains functional syntactic generation (50% AST parse).<br/>"
                           "<b>B. Qwen2.5-1.5B Cross-Model Generalization:</b> On Qwen2.5-1.5B (28 layers, 1.54B parameters) at 25.0% sparsity, ConnectomePrune achieves 28.47 PPL (vs. Dense 26.43), while Inverse-Hub degrades to 30.44 PPL, demonstrating that attention fragility is an architecture-agnostic property of compact transformers.<br/>"
                           "<b>C. Hardware Practicality:</b> The &lt; 3.5 GB RAM consumption denotes the offline disk-streaming footprint of the ConnectomeSculptor carving tool, enabling model pruning on consumer PCs. Unstructured sparsity does not deliver raw GEMM latency speedups without dedicated sparse kernel libraries.", body_style))

    # References
    story.append(Spacer(1, 2.5))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor('#CBD5E1'), spaceAfter=2.5))
    story.append(Paragraph("<b>References:</b> [1] M. Sun et al., NeurIPS (2023) [Wanda]. [2] G. Turrigiano, Cell 135(3) (2008). [3] L. Yin et al., ICML (2024) [OWL]. [4] S. Dorkenwald et al., Nature 634 (2024). [5] P. Schlegel et al., Nature 634 (2024). [6] E. Frantar & D. Alistarh, ICML (2023) [SparseGPT]. [7] P. Xu et al., BESA (2024).", affil_style))

    doc.build(story)
    print(f"[SUCCESS] Built NeurIPS Master PDF: {filename}")

if __name__ == "__main__":
    out_pdf = r"C:\Users\hp\.gemini\antigravity\scratch\MAKALELER\ConnectomePrune_NeurIPS_Full_Manuscript.pdf"
    build_pdf(out_pdf)
