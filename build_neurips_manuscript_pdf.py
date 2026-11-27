"""
Builds complete multi-page NeurIPS/ICLR formatted PDF for MAKALELER directory.
Addresses all forensic peer-review points:
- Honest Qwen2.5-1.5B framing (transferring kappa=0.75 without re-tuning does not beat Wanda due to GQA 12/2 parameter ratio)
- Module-specific scaling formulation (gamma_o = 1.109 for W_o, gamma_down = 1.185 for W_down) and C4 validation sweep
- Honest Regime B explanation (unconditioned short chunks, context truncation, Dense baseline at 94.79 +/- 1.2)
- Accurate citations and positioning for BESA (ICLR 2024) and AlphaPruning (NeurIPS 2024)
- Softened claims (avoiding "proven to preserve softmax temperature" or "rigorously demonstrated")
- Target vs. Empirical Sparsity columns, Dense-Agreement metric name, and McNemar test significance
- Clean Figure 1 caption (no meta-response language)
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
        fontSize=7.1,
        leading=9.3,
        textColor=colors.HexColor('#1E293B'),
        spaceAfter=2.0
    )

    cell_style = ParagraphStyle(
        'CellStyle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=5.7,
        leading=7.5,
        textColor=colors.HexColor('#1E293B')
    )

    cell_bold = ParagraphStyle(
        'CellBold',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=5.7,
        leading=7.5,
        textColor=colors.HexColor('#0F172A')
    )

    cell_header = ParagraphStyle(
        'CellHeader',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=5.7,
        leading=7.5,
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
    story.append(Paragraph("<b>Abstract</b>—Post-training pruning of large language models (LLMs) typically applies uniform sparsity across all linear projections. Motivated by biological connectomics—where topological routing hubs maintain high synaptic density while local pathways exhibit extreme sparsity—we observe that attention routing projections maintain 1.3x to 2.28x higher activation-weighted saliency density (|W| * ||X||_2) than feed-forward (MLP) layers across all depths. We propose <b>ConnectomePrune-LLM</b>, a closed-form intra-block pruning framework. ConnectomePrune establishes an analytical parameter conservation rule shielding attention routing (alpha<sub>attn</sub> = 18.75%) while shifting sparsity onto feed-forward blocks (alpha<sub>mlp</sub> = 28.75%) to meet an analytical target linear reduction S<sub>target</sub> = 27.03% (262.98M parameters, resulting in 27.10% empirical zeroing due to boundary ties). We introduce module-specific output-selective scaling (gamma<sub>o</sub> = 1.109 for W<sub>o</sub>, gamma<sub>down</sub> = 1.185 for W<sub>down</sub>), deliberately leaving W<sub>q,k</sub> unscaled to avoid gamma^2 softmax temperature sharpening, and leaving W<sub>gate,up</sub> unscaled to prevent gated SiLU distortion. On LLaMA-3.2-1B-Instruct, ConnectomePrune achieves WikiText-2 perplexity of <b>11.48 +/- 0.04</b> (ctx 512) and <b>10.92</b> (ctx 2k), retaining <b>100.0% [92.9, 100.0]</b> Python AST code syntax validity and <b>100.0% [98.1, 100.0]</b> dense-agreement. Our method statistically separates from uniform Wanda (12.85 PPL, code [67.0, 88.8], McNemar p &lt; 0.005), while achieving parity with Outlier-Weighed Layerwise pruning (OWL + Scaling, 11.56 +/- 0.05, code [89.5, 99.6], McNemar p = 0.48) in closed form without depth grid sweeps. Penalizing attention (Inverse-Hub) degrades perplexity to <b>66.10</b>, consistent with routing sensitivity. Cross-model evaluation on Qwen2.5-1.5B (28 layers, GQA 12/2) demonstrates that Inverse-Hub causes the largest degradation (30.44 PPL vs. 26.43 Dense); however, directly transferring kappa = 0.75 without architecture-specific re-tuning yields 28.47 PPL, which does not outperform uniform Wanda (27.44) due to Qwen's substantially smaller attention footprint. ConnectomePrune operates zero-shot with &lt; 3.5 GB host RAM.", body_style))
    story.append(Spacer(1, 2.0))

    # Section I: Related Work & Positioning
    story.append(Paragraph("I. RELATED WORK & METHODOLOGICAL POSITIONING", h1_style))
    story.append(Paragraph("<b>Inter-Layer vs. Intra-Block Sparsity:</b> Post-training pruning typically operates uniformly across all projections (SparseGPT, Wanda). Recent non-uniform allocators—such as OWL (Yin et al., ICML 2024), AlphaPruning (Lu et al., NeurIPS 2024), and BESA (Xu et al., ICLR 2024)—explore non-uniform allocations. AlphaPruning uses Empirical Spectral Density (ESD) heavy-tailed regularization to assign layer depths, while BESA learns layer-wise and block-wise masks using differentiable first-order Taylor approximations. In contrast, ConnectomePrune addresses the intra-block dimension (attention routing vs. feed-forward memory) in an analytical <b>closed-form, zero-shot, gradient-free manner</b>, requiring zero backpropagation, no Hessian inversions, and operating in seconds on CPU. Furthermore, ConnectomePrune is complementary to depth-wise allocators: integrating Attention Hub Protection on top of OWL depth allocations reduces zero-shot stress perplexity by <b>-4.43 PPL</b> (Section IV).", body_style))
    story.append(Spacer(1, 1.5))

    # Section II: Mechanistic Saliency Density Analysis & Figure 1
    story.append(Paragraph("II. MECHANISTIC SALIENCY DENSITY ANALYSIS & FIGURE 1", h1_style))
    story.append(Paragraph("Layer-wise saliency density psi<sub>l</sub> = S<sub>attn</sub> / S<sub>mlp</sub> profiles the relative parameter importance across layers. As illustrated in Figure 1, psi<sub>l</sub> &gt; 1.0x holds across all 16 layers of LLaMA-3.2-1B, peaking at Layer 2 (psi<sub>2</sub> = 2.28x).<br/>"
                           "<i>Activation Scale Caveat:</i> While W<sub>q,k,v</sub> receive RMSNorm hidden states and W<sub>down</sub> receives post-SwiGLU activations, psi<sub>l</sub> serves as a consistent structural indicator. Functional vulnerability is demonstrated through the Inverse-Hub experiment, where reversing budgets induces severe degradation (66.10 PPL).", body_style))
    
    fig_path = r"C:\Users\hp\.gemini\antigravity\scratch\MAKALELER\fig1_layer_saliency_ratio.png"
    if os.path.exists(fig_path):
        story.append(Spacer(1, 1.5))
        story.append(Image(fig_path, width=480, height=125))
        story.append(Paragraph("<b>Figure 1:</b> (a) Mean activation-weighted saliency density for Attention vs. MLP projections across layers. (b) Relative saliency ratio psi<sub>l</sub>, demonstrating attention routing concentration across all depths.", affil_style))
        story.append(Spacer(1, 1.5))

    # Section III: Main Benchmark Table 1
    story.append(Paragraph("III. MAIN BENCHMARK EVALUATION (REGIME A: FULL LONG-CONTEXT EVALUATION)", h1_style))
    story.append(Paragraph("<b>Regime A Protocol:</b> Evaluated on the complete WikiText-2 test set (&gt; 100,000 tokens) with continuous sliding-window context (512 &amp; 2,048 tokens), calibrated over 128 C4 training samples across 3 random seeds. Exact 95% Wilson Score intervals are computed for Code Syntax (n=50) and Dense-Agreement (n=200). Paired McNemar test on code syntax confirms ConnectomePrune statistically separates from Wanda (p &lt; 0.005) and achieves parity with OWL + Scaling (p = 0.48).", body_style))

    table_data = [
        [Paragraph("Pruning Method", cell_header), Paragraph("Target / Emp.", cell_header), Paragraph("WikiText-2 PPL (512 / 2k)", cell_header), Paragraph("Code Syntax (50) [95% Wilson CI]", cell_header), Paragraph("Dense-Agreement (200) [95% CI]", cell_header)],
        [Paragraph("Dense Baseline (Meta 1B)", cell_style), Paragraph("0.00% / 0.00%", cell_style), Paragraph("11.24 +/- 0.02 / 10.70", cell_style), Paragraph("100.0% (50/50) [92.9, 100.0]", cell_style), Paragraph("100.0% (Reference Output)", cell_style)],
        [Paragraph("Magnitude Pruning", cell_style), Paragraph("27.03% / 27.10%", cell_style), Paragraph("184.60 (Collapsed)", cell_style), Paragraph("0.0% (0/50) [0.0, 7.1]", cell_style), Paragraph("0.0% (0/200 Collapsed)", cell_style)],
        [Paragraph("Standard Wanda (NeurIPS 2023)", cell_style), Paragraph("27.03% / 27.10%", cell_style), Paragraph("12.85 +/- 0.08", cell_style), Paragraph("80.0% (40/50) [67.0, 88.8]", cell_style), Paragraph("95.0% (190/200) [91.0, 97.3]", cell_style)],
        [Paragraph("Wanda + Scaling Only", cell_style), Paragraph("27.03% / 27.10%", cell_style), Paragraph("12.31 +/- 0.07", cell_style), Paragraph("88.0% (44/50) [76.2, 94.4]", cell_style), Paragraph("97.0% (194/200) [93.6, 98.6]", cell_style)],
        [Paragraph("SparseGPT (ICML 2023)", cell_style), Paragraph("27.03% / 27.10%", cell_style), Paragraph("12.15 +/- 0.07", cell_style), Paragraph("92.0% (46/50) [81.2, 96.8]", cell_style), Paragraph("98.0% (196/200) [95.0, 99.2]", cell_style)],
        [Paragraph("SparseGPT + Scaling", cell_style), Paragraph("27.03% / 27.10%", cell_style), Paragraph("11.72 +/- 0.06", cell_style), Paragraph("94.0% (47/50) [83.8, 97.9]", cell_style), Paragraph("98.5% (197/200) [95.7, 99.5]", cell_style)],
        [Paragraph("OWL (ICML 2024)", cell_style), Paragraph("27.03% / 27.10%", cell_style), Paragraph("11.84 +/- 0.06", cell_style), Paragraph("96.0% (48/50) [86.5, 98.9]", cell_style), Paragraph("98.5% (197/200) [95.7, 99.5]", cell_style)],
        [Paragraph("OWL + Scaling", cell_style), Paragraph("27.03% / 27.10%", cell_style), Paragraph("11.56 +/- 0.05", cell_style), Paragraph("98.0% (49/50) [89.5, 99.6]", cell_style), Paragraph("99.0% (198/200) [96.4, 99.7]", cell_style)],
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
    story.append(Paragraph("Because saliency pruning selectively eliminates near-zero importance weights (losing only ~1.27% layer output variance), theoretical 1/sqrt(1-alpha) acts as a heuristic signal amplification factor offsetting downstream layer attenuation. We swept scaling configurations on the C4 validation set:", body_style))

    gamma_data = [
        [Paragraph("Configuration", cell_header), Paragraph("Applied Projections", cell_header), Paragraph("Scaling Factors", cell_header), Paragraph("C4 Val PPL (LLaMA-3.2-1B)", cell_header)],
        [Paragraph("No Scaling (Hub Protection Only)", cell_style), Paragraph("None", cell_style), Paragraph("gamma = 1.000", cell_style), Paragraph("11.92 +/- 0.05", cell_style)],
        [Paragraph("Naive Uniform Scaling", cell_style), Paragraph("All linear matrices", cell_style), Paragraph("gamma = 1.185", cell_style), Paragraph("13.40 (Degraded by temperature distortion)", cell_style)],
        [Paragraph("Output-Selective Mild", cell_style), Paragraph("W_o, W_down", cell_style), Paragraph("gamma_o = 1.05, gamma_down = 1.05", cell_style), Paragraph("11.68 +/- 0.05", cell_style)],
        [Paragraph("<b>Output-Selective Module-Specific (Ours)</b>", cell_bold), Paragraph("<b>W_o, W_down</b>", cell_bold), Paragraph("<b>gamma_o = 1.109, gamma_down = 1.185</b>", cell_bold), Paragraph("<b>11.45 (Optimal; 11.48 WikiText-2)</b>", cell_bold)],
        [Paragraph("Output-Selective Over-Scaled", cell_style), Paragraph("W_o, W_down", cell_style), Paragraph("gamma_o = 1.15, gamma_down = 1.25", cell_style), Paragraph("11.79 +/- 0.06 (Variance inflation)", cell_style)]
    ]
    t_gamma = Table(gamma_data, colWidths=[120, 85, 120, 157])
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
    story.append(Paragraph("<b>Regime B Protocol:</b> Evaluates 10 isolated, unconditioned 512-token chunks from WikiText-2 without sliding-window context history, prefix prompts, or BOS history warmup. Due to prefix loss and context fragmentation, raw cross-entropy is elevated across all models (Dense is 94.79 +/- 1.2). While Attention Hub Protection reduces stress perplexity from 105.42 +/- 1.3 (OWL) to 100.99 +/- 1.1, full continuous evaluation on long-context WikiText-2 (Regime A) remains our primary statistical benchmark.", body_style))

    owl_data = [
        [Paragraph("Configuration (LLaMA-3.2-1B)", cell_header), Paragraph("Empirical Sparsity", cell_header), Paragraph("Zero-Shot Stress PPL (512 ctx)", cell_header), Paragraph("Orthogonal Synergy & Analysis", cell_header)],
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

    story.append(Paragraph("<b>A. Analysis of Scaling Interaction (Conjectural Hypothesis):</b> In Regime B, output scaling degrades OWL depth allocations (105.42 to 111.89). We hypothesize that because saliency pruning loses only ~1.27% variance, applying upper-bound scaling introduces slight variance inflation, which compounds when layers are already heavily skewed across depth by OWL. Synergistic benefit with OWL is driven strictly by Attention Hub Protection (-4.43 PPL).<br/>"
                           "<b>B. Cross-Model Evaluation (Qwen2.5-1.5B):</b> Evaluated on Qwen2.5-1.5B (28 layers, 1.54B parameters, GQA 12/2) transferring kappa = 0.75 without re-tuning at 25.0% sparsity: Dense Baseline achieves 26.43, Standard Wanda achieves 27.44 (+1.01), OWL achieves 27.21 (+0.78), ConnectomePrune achieves 28.47 (+2.04), while Inverse-Hub severely collapses to <b>30.44 (+4.01)</b>. <i>Honest Architectural Finding:</i> While Inverse-Hub confirms routing fragility across both architectures, directly transferring kappa = 0.75 does not outperform Wanda on Qwen because attention represents only ~11.2% of linear parameters (vs. 17.2% in LLaMA), indicating that kappa must be re-tuned for architectures with divergent attention-to-MLP ratios.<br/>"
                           "<b>C. Hardware Practicality:</b> The &lt; 3.5 GB host RAM denotes the offline disk-streaming footprint of the ConnectomeSculptor carving tool, enabling model pruning on consumer PCs. Unstructured sparsity does not deliver raw GEMM latency speedups without dedicated sparse kernel runtimes or 2:4 hardware.", body_style))

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
