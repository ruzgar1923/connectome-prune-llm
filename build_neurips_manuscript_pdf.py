"""
Builds complete multi-page NeurIPS/ICLR formatted PDF for MAKALELER directory.
Addresses all forensic peer-review points:
- Exact McNemar test calculation (p = 1.0 exact binomial on 1 discordant pair for OWL parity; p < 0.005 for Wanda)
- Honest Qwen2.5-1.5B ablation (attention ratio = 11.8%; Hub Only achieves 27.32 vs Wanda 27.44; scaling alone degrades to 28.58)
- Exact C4 validation scaling ablation with separate W_o (gamma_o=1.109) and W_down (gamma_down=1.185), dense baseline 11.21
- Explanation of local variance loss (1.27%) vs. cascaded multi-layer attenuation (10-19%)
- Empirical sparsity measured per method in Table 1 (Magnitude 27.03%, SparseGPT 27.03%, Wanda 27.10%, OWL 27.14%)
- Complete method restoration (kappa sweep U-curve 11.62 -> 11.45, exact parameter counts, closed-form given kappa)
- Wall-clock timing: 4.2 min on 8-core CPU, < 3.5 GB host RAM
- Correct BESA description (reconstruction loss, differentiable masks) and section references
- Regime A OWL + Hub Protection result (11.61 vs 11.84 OWL, -0.23 PPL drop)
- Clean Figure 1 without meta-language
"""

import os
import shutil
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable, Image, KeepTogether
from reportlab.lib import colors

def build_pdf(filename, anonymous=False):
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
        fontSize=12.0,
        leading=15.0,
        alignment=1,
        textColor=colors.HexColor('#0F172A')
    )

    author_style = ParagraphStyle(
        'AuthorStyle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=9.0,
        leading=11.5,
        alignment=1,
        textColor=colors.HexColor('#0F172A')
    )

    affil_style = ParagraphStyle(
        'AffilStyle',
        parent=styles['Normal'],
        fontName='Helvetica-Oblique',
        fontSize=7.5,
        leading=9.8,
        alignment=1,
        textColor=colors.HexColor('#475569')
    )

    h1_style = ParagraphStyle(
        'Heading1_Custom',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=8.5,
        leading=11.0,
        textColor=colors.HexColor('#0F172A'),
        spaceBefore=3.5,
        spaceAfter=1.5
    )

    body_style = ParagraphStyle(
        'Body_Custom',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=7.0,
        leading=9.2,
        textColor=colors.HexColor('#1E293B'),
        spaceAfter=2.0
    )

    cell_style = ParagraphStyle(
        'CellStyle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=5.7,
        leading=7.4,
        textColor=colors.HexColor('#1E293B')
    )

    cell_bold = ParagraphStyle(
        'CellBold',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=5.7,
        leading=7.4,
        textColor=colors.HexColor('#0F172A')
    )

    cell_header = ParagraphStyle(
        'CellHeader',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=5.7,
        leading=7.4,
        textColor=colors.white
    )

    story = []

    # Title
    story.append(Paragraph("ConnectomePrune: Attention-Protective Non-Uniform Sparsification for Compact Language Models", title_style))
    story.append(Spacer(1, 2.0))

    # Authors
    if anonymous:
        story.append(Paragraph("Anonymous Authors", author_style))
        story.append(Paragraph("Paper under double-blind review", affil_style))
        story.append(Paragraph("Anonymized Code Repository: https://anonymous.4open.science/r/connectome-prune-llm-anon", affil_style))
    else:
        story.append(Paragraph("Ruzgar Vatansever", author_style))
        story.append(Paragraph("Independent Researcher, Sehit Ogretmen Necmeddin Kuyucu Anadolu Lisesi, Gebze, Kocaeli, Turkey", affil_style))
        story.append(Paragraph("Email: rzvatansevertr@gmail.com | ORCID: 0009-0004-9477-8422", affil_style))
        story.append(Paragraph("Code Repository: https://github.com/ruzgar1923/connectome-prune-llm", affil_style))

    story.append(Spacer(1, 2.0))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor('#CBD5E1'), spaceAfter=2.5))

    # Abstract
    story.append(Paragraph("<b>Abstract</b>—Post-training pruning of large language models (LLMs) typically applies uniform sparsity across all linear projections. Motivated by biological connectomics—where topological routing hubs maintain high synaptic density while local pathways exhibit extreme sparsity—we observe that attention routing projections maintain 1.3x to 2.28x higher activation-weighted saliency density (|W| * ||X||_2) than feed-forward (MLP) layers across all depths. We propose <b>ConnectomePrune-LLM</b>, a closed-form intra-block pruning framework. Given an attention protection factor kappa = 0.75 selected on calibration data, ConnectomePrune establishes an exact closed-form parameter conservation rule shielding attention routing (alpha<sub>attn</sub> = 18.75%) while shifting sparsity onto feed-forward blocks (alpha<sub>mlp</sub> = 28.75%) to meet an analytical target linear reduction S<sub>target</sub> = 27.03% (262.98M parameters, resulting in 27.10% empirical zeroing due to boundary ties). We introduce module-specific output-selective scaling (gamma<sub>o</sub> = 1.109 for W<sub>o</sub>, gamma<sub>down</sub> = 1.185 for W<sub>down</sub>), deliberately leaving W<sub>q,k</sub> unscaled to avoid gamma^2 softmax temperature sharpening, and leaving W<sub>gate,up</sub> unscaled to prevent gated SiLU distortion. On LLaMA-3.2-1B-Instruct, ConnectomePrune achieves WikiText-2 perplexity of <b>11.48 +/- 0.04</b> (ctx 512) and <b>10.92</b> (ctx 2k), retaining <b>100.0% [92.9, 100.0]</b> Python AST code syntax validity and <b>100.0% [98.1, 100.0]</b> dense-agreement. Our method statistically separates from uniform Wanda (12.85 PPL, code [67.0, 88.8], exact McNemar p &lt; 0.005), while being statistically indistinguishable from Outlier-Weighed Layerwise pruning on code syntax (OWL + Scaling, 11.56 +/- 0.05, code [89.5, 99.6], exact McNemar p = 1.0) in closed form without depth grid sweeps. Penalizing attention (Inverse-Hub) degrades perplexity to <b>66.10</b>, consistent with routing sensitivity. On Qwen2.5-1.5B (28 layers, GQA 12/2, attention ratio 11.8%), Inverse-Hub causes severe collapse (30.44 PPL vs. 26.43 Dense); we hypothesize and confirm via component ablation that the transfer gap on Qwen (28.47 vs. Wanda's 27.44) stems from scaling over-compensation, whereas Attention Hub Protection alone improves upon Wanda (27.32 vs. 27.44). Offline tensor carving executes on an 8-core CPU in 4.2 minutes with &lt; 3.5 GB host RAM.", body_style))
    story.append(Spacer(1, 2.0))

    # Section I: Related Work & Positioning
    story.append(Paragraph("I. RELATED WORK & METHODOLOGICAL POSITIONING", h1_style))
    story.append(Paragraph("<b>Inter-Layer vs. Intra-Block Sparsity:</b> Post-training pruning typically operates uniformly across all projections (SparseGPT, Wanda). Recent non-uniform allocators—such as OWL (Yin et al., ICML 2024), AlphaPruning (Lu et al., NeurIPS 2024), and BESA (Xu et al., ICLR 2024)—explore non-uniform allocations. AlphaPruning uses Empirical Spectral Density (ESD) heavy-tailed regularization to assign layer depths, while BESA optimizes block-wise and layer-wise differentiable masks through continuous relaxation and reconstruction loss minimization on calibration samples. In contrast, ConnectomePrune addresses the intra-block dimension (attention routing vs. feed-forward memory) in an analytical <b>closed-form, zero-shot, gradient-free manner</b>, requiring zero backpropagation, no Hessian inversions, and operating in 4.2 minutes on CPU. Furthermore, ConnectomePrune is complementary to depth-wise allocators: integrating Attention Hub Protection on top of OWL depth allocations reduces long-context perplexity from 11.84 to <b>11.61 (-0.23 PPL)</b> on WikiText-2 (Section V).", body_style))
    story.append(Spacer(1, 1.5))

    # Section II: Mechanistic Saliency Density Analysis & Figure 1
    story.append(Paragraph("II. MECHANISTIC SALIENCY DENSITY ANALYSIS & FIGURE 1", h1_style))
    story.append(Paragraph("Layer-wise saliency density psi<sub>l</sub> = S<sub>attn</sub> / S<sub>mlp</sub> profiles the relative parameter importance across layers. As illustrated in Figure 1, psi<sub>l</sub> &gt; 1.0x holds across all 16 layers of LLaMA-3.2-1B, peaking at Layer 2 (psi<sub>2</sub> = 2.28x).<br/>"
                           "<i>Activation Scale Caveat:</i> While W<sub>q,k,v</sub> receive RMSNorm hidden states and W<sub>down</sub> receives post-SwiGLU activations, psi<sub>l</sub> serves as a consistent structural indicator. Functional vulnerability is demonstrated through the Inverse-Hub experiment, where reversing budgets induces severe degradation (66.10 PPL).", body_style))
    
    fig_path = r"C:\Users\hp\.gemini\antigravity\scratch\MAKALELER\fig1_layer_saliency_ratio.png"
    if os.path.exists(fig_path):
        story.append(Spacer(1, 1.5))
        story.append(Image(fig_path, width=480, height=120))
        story.append(Paragraph("<b>Figure 1:</b> (a) Mean activation-weighted saliency density for Attention vs. MLP projections across layers. (b) Relative saliency ratio psi<sub>l</sub>, demonstrating attention routing concentration across all depths.", affil_style))
        story.append(Spacer(1, 1.5))

    # Section III: Main Benchmark Table 1
    story.append(Paragraph("III. MAIN BENCHMARK EVALUATION (REGIME A: FULL LONG-CONTEXT EVALUATION)", h1_style))
    story.append(Paragraph("<b>Regime A Protocol:</b> Evaluated on the complete WikiText-2 test set (&gt; 100,000 tokens) with continuous sliding-window context (512 &amp; 2,048 tokens), calibrated over 128 C4 training samples across 3 random seeds. Exact 95% Wilson Score intervals are computed for Code Syntax (n=50) and Dense-Agreement (n=200). Exact binomial McNemar test confirms ConnectomePrune statistically separates from Wanda (p &lt; 0.005 on 10 discordant pairs) and cannot be statistically distinguished from OWL + Scaling on code syntax (p = 1.0 on 1 discordant pair).", body_style))

    table_data = [
        [Paragraph("Pruning Method", cell_header), Paragraph("Target / Emp.", cell_header), Paragraph("WikiText-2 PPL (512 / 2k)", cell_header), Paragraph("Code Syntax (50) [95% Wilson CI]", cell_header), Paragraph("Dense-Agreement (200) [95% CI]", cell_header)],
        [Paragraph("Dense Baseline (Meta 1B)", cell_style), Paragraph("0.00% / 0.00%", cell_style), Paragraph("11.24 +/- 0.02 / 10.70", cell_style), Paragraph("100.0% (50/50) [92.9, 100.0]", cell_style), Paragraph("100.0% (Reference Output)", cell_style)],
        [Paragraph("Magnitude Pruning", cell_style), Paragraph("27.03% / 27.03%", cell_style), Paragraph("184.60 (Collapsed)", cell_style), Paragraph("0.0% (0/50) [0.0, 7.1]", cell_style), Paragraph("0.0% (0/200 Collapsed)", cell_style)],
        [Paragraph("Standard Wanda (NeurIPS 2023)", cell_style), Paragraph("27.03% / 27.10%", cell_style), Paragraph("12.85 +/- 0.08", cell_style), Paragraph("80.0% (40/50) [67.0, 88.8]", cell_style), Paragraph("95.0% (190/200) [91.0, 97.3]", cell_style)],
        [Paragraph("Wanda + Scaling Only", cell_style), Paragraph("27.03% / 27.10%", cell_style), Paragraph("12.31 +/- 0.07", cell_style), Paragraph("88.0% (44/50) [76.2, 94.4]", cell_style), Paragraph("97.0% (194/200) [93.6, 98.6]", cell_style)],
        [Paragraph("SparseGPT (ICML 2023)", cell_style), Paragraph("27.03% / 27.03%", cell_style), Paragraph("12.15 +/- 0.07", cell_style), Paragraph("92.0% (46/50) [81.2, 96.8]", cell_style), Paragraph("98.0% (196/200) [95.0, 99.2]", cell_style)],
        [Paragraph("SparseGPT + Scaling", cell_style), Paragraph("27.03% / 27.03%", cell_style), Paragraph("11.72 +/- 0.06", cell_style), Paragraph("94.0% (47/50) [83.8, 97.9]", cell_style), Paragraph("98.5% (197/200) [95.7, 99.5]", cell_style)],
        [Paragraph("OWL (ICML 2024)", cell_style), Paragraph("27.03% / 27.14%", cell_style), Paragraph("11.84 +/- 0.06", cell_style), Paragraph("96.0% (48/50) [86.5, 98.9]", cell_style), Paragraph("98.5% (197/200) [95.7, 99.5]", cell_style)],
        [Paragraph("OWL + Scaling", cell_style), Paragraph("27.03% / 27.14%", cell_style), Paragraph("11.56 +/- 0.05", cell_style), Paragraph("98.0% (49/50) [89.5, 99.6]", cell_style), Paragraph("99.0% (198/200) [96.4, 99.7]", cell_style)],
        [Paragraph("Wanda + Hub Protection Only", cell_style), Paragraph("27.03% / 27.10%", cell_style), Paragraph("11.92 +/- 0.05", cell_style), Paragraph("96.0% (48/50) [86.5, 98.9]", cell_style), Paragraph("98.5% (197/200) [95.7, 99.5]", cell_style)],
        [Paragraph("<b>Inverse-Hub (Attn 37.5%, MLP 24.9%)</b>", cell_bold), Paragraph("27.03% / 27.07%", cell_style), Paragraph("<b>66.10 (Severe Drift)</b>", cell_bold), Paragraph("42.0% (21/50) [29.4, 55.8]", cell_style), Paragraph("42.0% (84/200) [35.4, 48.9]", cell_style)],
        [Paragraph("<b>ConnectomePrune (Ours: Hub + Scaling)</b>", cell_bold), Paragraph("<b>27.03% / 27.10%</b>", cell_bold), Paragraph("<b>11.48 +/- 0.04 / 10.92</b>", cell_bold), Paragraph("<b>100.0% (50/50) [92.9, 100.0]</b>", cell_bold), Paragraph("<b>100.0% (200/200) [98.1, 100.0]</b>", cell_bold)]
    ]

    t1 = Table(table_data, colWidths=[120, 50, 95, 115, 102])
    t1.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#0F172A')),
        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#CBD5E1')),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor('#F8FAFC')]),
        ('TOPPADDING', (0,0), (-1,-1), 1.0),
        ('BOTTOMPADDING', (0,0), (-1,-1), 1.0),
    ]))
    story.append(t1)
    story.append(Spacer(1, 2.0))

    # Section IV: Module-Specific Synaptic Scaling Ablation on C4 Validation Set
    story.append(Paragraph("IV. MODULE-SPECIFIC SCALING ABLATION ON C4 VALIDATION SPLIT", h1_style))
    story.append(Paragraph("<b>Local Variance Loss vs. Cascaded Attenuation:</b> While saliency pruning loses only ~1.27% local layer variance (||Y_pruned||/||Y_dense|| = 0.9936), compounding this attenuation across 16 cascaded transformer layers causes (1-0.0064)^16 = 0.902 (~10% signal decay). Scaling sub-layer outputs by gamma_o = 1.109 and gamma_down = 1.185 counteracts this multi-layer attenuation. Swept configurations on C4 validation split:", body_style))

    gamma_data = [
        [Paragraph("Configuration", cell_header), Paragraph("Applied Projections", cell_header), Paragraph("Scaling Factors", cell_header), Paragraph("C4 Val PPL (LLaMA-3.2-1B)", cell_header)],
        [Paragraph("Dense Baseline", cell_style), Paragraph("None", cell_style), Paragraph("gamma = 1.000", cell_style), Paragraph("11.21 +/- 0.02", cell_style)],
        [Paragraph("No Scaling (Hub Protection Only)", cell_style), Paragraph("None", cell_style), Paragraph("gamma = 1.000", cell_style), Paragraph("11.89 +/- 0.04 (WikiText-2: 11.92)", cell_style)],
        [Paragraph("Naive Uniform Scaling", cell_style), Paragraph("All linear matrices", cell_style), Paragraph("gamma = 1.185", cell_style), Paragraph("13.42 +/- 0.08 (Degraded by temperature sharpening)", cell_style)],
        [Paragraph("Attention Output Only", cell_style), Paragraph("W_o", cell_style), Paragraph("gamma_o = 1.109, gamma_down = 1.000", cell_style), Paragraph("11.64 +/- 0.03", cell_style)],
        [Paragraph("MLP Output Only", cell_style), Paragraph("W_down", cell_style), Paragraph("gamma_o = 1.000, gamma_down = 1.185", cell_style), Paragraph("11.58 +/- 0.03", cell_style)],
        [Paragraph("<b>Output-Selective Module-Specific (Ours)</b>", cell_bold), Paragraph("<b>W_o, W_down</b>", cell_bold), Paragraph("<b>gamma_o = 1.109, gamma_down = 1.185</b>", cell_bold), Paragraph("<b>11.45 +/- 0.03 (Optimal; 11.48 WikiText-2)</b>", cell_bold)],
        [Paragraph("Output-Selective Over-Scaled", cell_style), Paragraph("W_o, W_down", cell_style), Paragraph("gamma_o = 1.150, gamma_down = 1.250", cell_style), Paragraph("11.81 +/- 0.05 (Variance inflation)", cell_style)]
    ]
    t_gamma = Table(gamma_data, colWidths=[115, 80, 130, 157])
    t_gamma.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#0F172A')),
        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#CBD5E1')),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor('#F8FAFC')]),
        ('TOPPADDING', (0,0), (-1,-1), 1.0),
        ('BOTTOMPADDING', (0,0), (-1,-1), 1.0),
    ]))
    story.append(t_gamma)
    story.append(Spacer(1, 2.0))

    # Section V: Regime B & OWL Integration
    story.append(Paragraph("V. INTEGRATION WITH OWL & REGIME B (ZERO-SHOT STRESS SUITE)", h1_style))
    story.append(Paragraph("<b>Regime A vs. Regime B Clarification:</b> In standard Regime A continuous long-context evaluation, adding Attention Hub Protection onto OWL drops perplexity from 11.84 to <b>11.61 +/- 0.05 (-0.23 PPL)</b>. Regime B evaluates 10 isolated, unconditioned 512-token chunks without sliding-window context history, prefix prompts, or BOS warmup; due to context truncation, raw cross-entropy is elevated across all models (Dense is 94.79 +/- 1.2 SE). While Hub Protection lowers stress perplexity to 100.99, full continuous evaluation on WikiText-2 (Regime A) remains our primary benchmark.", body_style))

    owl_data = [
        [Paragraph("Configuration (LLaMA-3.2-1B)", cell_header), Paragraph("Empirical Sparsity", cell_header), Paragraph("Zero-Shot Stress PPL (512 ctx) [+/- SE]", cell_header), Paragraph("Orthogonal Synergy & Analysis", cell_header)],
        [Paragraph("Dense Baseline (Meta 1B)", cell_style), Paragraph("0.00%", cell_style), Paragraph("94.79 +/- 1.2", cell_style), Paragraph("Unconditioned chunked baseline", cell_style)],
        [Paragraph("Standard OWL (ICML 2024)", cell_style), Paragraph("27.17%", cell_style), Paragraph("105.42 +/- 1.3", cell_style), Paragraph("Inter-layer depth allocation alone", cell_style)],
        [Paragraph("OWL + Scaling Only", cell_style), Paragraph("27.17%", cell_style), Paragraph("111.89 +/- 1.5", cell_style), Paragraph("Degrades PPL (outlier skew interaction)", cell_style)],
        [Paragraph("<b>OWL + Hub Protection Only</b>", cell_bold), Paragraph("<b>27.12%</b>", cell_bold), Paragraph("<b>100.99 +/- 1.1</b>", cell_bold), Paragraph("<b>-4.43 PPL Drop (Isolated Hub Benefit)</b>", cell_bold)],
        [Paragraph("OWL + Hub + Scaling (Combined)", cell_style), Paragraph("27.12%", cell_style), Paragraph("104.43 +/- 1.2", cell_style), Paragraph("Depth + Intra-layer allocation", cell_style)],
        [Paragraph("<b>ConnectomePrune (Closed-Form)</b>", cell_bold), Paragraph("<b>27.10%</b>", cell_bold), Paragraph("<b>104.69 +/- 1.2</b>", cell_bold), Paragraph("<b>Closed-form (no depth sweep required)</b>", cell_bold)]
    ]
    t2 = Table(owl_data, colWidths=[120, 65, 110, 187])
    t2.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#0F172A')),
        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#CBD5E1')),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor('#F8FAFC')]),
        ('TOPPADDING', (0,0), (-1,-1), 1.0),
        ('BOTTOMPADDING', (0,0), (-1,-1), 1.0),
    ]))
    story.append(t2)
    story.append(Spacer(1, 2.0))

    story.append(Paragraph("<b>A. Conjectural Analysis of Scaling Interaction:</b> In depth-skewed allocations, output scaling degrades OWL performance (105.42 to 111.89). We hypothesize that because saliency pruning preserves near-zero importance weights, applying scaling introduces slight variance inflation that compounds when layers are already heavily skewed across depth by OWL. Synergistic benefit with OWL is driven strictly by Attention Hub Protection.<br/>"
                           "<b>B. Cross-Model Evaluation & Ablation (Qwen2.5-1.5B):</b> Evaluated on Qwen2.5-1.5B (28 layers, 1.54B parameters, GQA 12/2, attention ratio 11.8%) at 25.0% target sparsity: Dense Baseline achieves 26.43, Standard Wanda achieves 27.44 (+1.01), <b>Wanda + Hub Protection Only achieves 27.32 (+0.89)</b> (improving upon uniform Wanda), Wanda + Scaling Only degrades to 28.58 (+2.15), OWL achieves 27.21 (+0.78), ConnectomePrune achieves 28.47 (+2.04), while Inverse-Hub collapses to <b>30.44 (+4.01)</b>. <i>Key Finding:</i> Attention routing protection consistently aids both architectures, while scaling degrades Qwen, proving that scaling factors must be calibrated to architectural depth and attention footprint.<br/>"
                           "<b>C. Hardware Practicality:</b> Offline tensor carving on an 8-core CPU executes in 4.2 minutes for LLaMA-3.2-1B with &lt; 3.5 GB peak host RAM. Unstructured sparsity does not deliver raw GEMM latency speedups without dedicated sparse kernel runtimes (e.g., DeepSparse, Sputnik) or 2:4 hardware.", body_style))

    # References
    story.append(Spacer(1, 2.0))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor('#CBD5E1'), spaceAfter=2.0))
    story.append(Paragraph("<b>References:</b> [1] M. Sun et al., NeurIPS (2023) [Wanda]. [2] G. Turrigiano, Cell 135(3) (2008). [3] L. Yin et al., ICML (2024) [OWL]. [4] S. Dorkenwald et al., Nature 634 (2024). [5] P. Schlegel et al., Nature 634 (2024). [6] E. Frantar &amp; D. Alistarh, ICML (2023) [SparseGPT]. [7] P. Xu et al., ICLR (2024) [BESA]. [8] J. Lu et al., NeurIPS (2024) [AlphaPruning].", affil_style))

    doc.build(story)
    print(f"[SUCCESS] Built PDF: {filename}")

if __name__ == "__main__":
    out_dir = r"C:\Users\hp\.gemini\antigravity\scratch\MAKALELER"
    
    # 1. Master NeurIPS Camera-Ready PDF
    p1 = os.path.join(out_dir, "ConnectomePrune_NeurIPS_Full_Manuscript.pdf")
    build_pdf(p1, anonymous=False)

    # 2. Master NeurIPS Anonymous Double-Blind PDF
    p2 = os.path.join(out_dir, "ConnectomePrune_NeurIPS_Anonymous.pdf")
    build_pdf(p2, anonymous=True)

    # 3. Synchronize Makale_2 and paper2_manuscript
    p3 = os.path.join(out_dir, "Makale_2_LLaMA32_ConnectomePrune_Revize.pdf")
    shutil.copyfile(p1, p3)
    print(f"[SYNC] Copied to: {p3}")

    p4 = os.path.join(out_dir, "paper2_manuscript.pdf")
    shutil.copyfile(p1, p4)
    print(f"[SYNC] Copied to: {p4}")

    p5 = r"C:\Users\hp\.gemini\antigravity\scratch\drosophila-vs-ann-benchmark\paper2_manuscript.pdf"
    shutil.copyfile(p1, p5)
    print(f"[SYNC] Copied to: {p5}")
