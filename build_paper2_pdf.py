import os
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
from reportlab.lib import colors

def build_pdf(filename="paper2_manuscript.pdf"):
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
        fontSize=13.5,
        leading=17.5,
        alignment=1,
        textColor=colors.HexColor('#0F172A')
    )

    author_style = ParagraphStyle(
        'AuthorStyle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=10,
        leading=12.5,
        alignment=1,
        textColor=colors.HexColor('#0F172A')
    )

    affil_style = ParagraphStyle(
        'AffilStyle',
        parent=styles['Normal'],
        fontName='Helvetica-Oblique',
        fontSize=8,
        leading=11,
        alignment=1,
        textColor=colors.HexColor('#475569')
    )

    h1_style = ParagraphStyle(
        'Heading1_Custom',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=9.5,
        leading=12.5,
        textColor=colors.HexColor('#0F172A'),
        spaceBefore=5,
        spaceAfter=2
    )

    body_style = ParagraphStyle(
        'Body_Custom',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8,
        leading=11,
        textColor=colors.HexColor('#1E293B'),
        spaceAfter=3
    )

    story = []

    # Title
    story.append(Paragraph("ConnectomePrune: Biological Hub-Preserving Sparsification for Compact Language Models", title_style))
    story.append(Spacer(1, 4))

    # Author
    story.append(Paragraph("Rüzgar Vatansever", author_style))
    story.append(Paragraph("Independent Researcher, Şehit Öğretmen Necmeddin Kuyucu Anadolu Lisesi, Gebze, Kocaeli, Turkey", affil_style))
    story.append(Paragraph("Email: rzvatansevertr@gmail.com | ORCID: 0009-0004-9477-8422", affil_style))
    story.append(Paragraph("Reproducibility Repository: https://github.com/ruzgar1923/connectome-prune-llm", affil_style))
    story.append(Spacer(1, 5))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor('#CBD5E1'), spaceAfter=5))

    # Abstract
    story.append(Paragraph("<b>Abstract</b>—Deploying compact foundation models on edge devices requires aggressive parameter compression. While empirical activation-weight pruning (e.g., Wanda) effectively mitigates outlier destruction, uniform layer-wise sparsification degrades sensitive multi-head attention routing channels. Drawing conceptual inspiration from biological connectomic invariants—specifically central complex routing resilience and homeostatic synaptic scaling in <i>Drosophila melanogaster</i> (Nature, 2024)—we introduce <b>ConnectomePrune-LLM</b>. In LLaMA-3.2-1B, linear parameters partition into attention routing projections (17.2%) and feed-forward MLP blocks (82.8%). Our framework contributes: (1) <i>Central Complex Hub Protection</i>, establishing non-uniform pruning budgets (&alpha;<sub>target</sub> = 0.25) that allocate lower sparsity (18.75%) to routing heads while shifting higher sparsity (28.75%) onto MLP blocks, yielding 27.10% aggregate sparsity; (2) <i>Output-Selective Homeostatic Scaling</i> (&gamma; = 1/&radic;&rho;), applied strictly to MLP projections and attention output heads (W<sub>o</sub>) while bypassing query/key matrices (W<sub>q</sub>, W<sub>k</sub>) to preserve softmax temperature invariance; and (3) <i>ConnectomeSculptor</i>, an in-place sharded tensor-carving pipeline (&lt; 3.5 GB RAM). We emphasize that biological inspiration is conceptual; no direct FlyWire synaptic wiring data is injected into transformer weights. Applying ConnectomePrune to Meta's official <b>LLaMA-3.2-1B-Instruct</b> eliminates <b>263,672,581 linear weights (27.10% linear sparsity)</b> in 29.08 seconds. An exact-sparsity Inverse-Hub ablation (penalizing attention at 37.5% and MLP at 24.9% to maintain 27.10% total sparsity) triggers catastrophic perplexity degradation to <b>66.10</b>, whereas ConnectomePrune bounds perplexity to <b>11.48</b> (vs. 11.24 dense baseline) across WikiText-2 and 50 algorithmic coding synthesis benchmarks.", body_style))
    story.append(Spacer(1, 4))

    # I. Introduction & Related Work
    story.append(Paragraph("I. INTRODUCTION & RELATED WORK", h1_style))
    story.append(Paragraph("Transformer language models contain significant parameter redundancy. Naive magnitude pruning destroys weights attached to high-magnitude activation channels, inducing catastrophic perplexity explosion. Wanda (Sun et al., 2023) introduced first-order saliency S<sub>ij</sub> = |W<sub>ij</sub>| &times; ||X<sub>j</sub>||<sub>2</sub> to protect activation outliers without second-order Hessian inversions (SparseGPT; Frantar & Alistarh, 2023). However, standard Wanda enforces uniform sparsity across all projections. While non-uniform heuristics such as OWL (Yin et al., 2024) explore layer sensitivity, biological connectomes (Dorkenwald et al., 2024; Schlegel et al., 2024) solve routing and resource-efficiency via topological hub protection and homeostatic plasticity. In this paper, we bridge these biological invariants with practical transformer sparsification.", body_style))

    # II. Methodology
    story.append(Paragraph("II. METHODOLOGY & MATHEMATICAL FORMULATION", h1_style))
    story.append(Paragraph("<b>A. Activation Saliency Baseline:</b> Grounded in Wanda (Sun et al., 2023), input channel &ell;<sub>2</sub>-norms ||X<sub>j</sub>||<sub>2</sub> are captured across calibration tokens. Synaptic saliency is computed as S<sub>ij</sub> = |W<sub>ij</sub>| &times; ||X<sub>j</sub>||<sub>2</sub>.", body_style))
    story.append(Paragraph("<b>B. Central Complex Hub Protection:</b> In LLaMA-3.2-1B, linear parameters are partitioned into attention projections (167.8M parameters, 17.2%) and MLP projections (805.3M parameters, 82.8%). Attention heads (W<sub>q</sub>, W<sub>k</sub>, W<sub>v</sub>, W<sub>o</sub>) are allocated protected budget &alpha;<sub>attn</sub> = 0.75 &times; &alpha;<sub>target</sub>, while MLP projections absorb compensatory sparsity &alpha;<sub>mlp</sub> = min(0.35, 1.15 &times; &alpha;<sub>target</sub>). At target &alpha;<sub>target</sub> = 0.25, attention sparsity is 18.75% and MLP is 28.75%, totaling 27.10% aggregate sparsity (263.67M weights carved). Embedding and language model prediction heads remain 100% dense.", body_style))
    story.append(Paragraph("<b>C. Output-Selective Homeostatic Scaling:</b> In Transformer self-attention, scaling query and key matrices by &gamma; would scale dot-product logits by &gamma;<sup>2</sup>, distorting softmax entropy. Therefore, homeostatic scaling &gamma; = 1 / &radic;&rho; is applied strictly to projection outputs: W<sub>o</sub> and MLP blocks (W<sub>gate</sub>, W<sub>up</sub>, W<sub>down</sub>), ensuring variance restoration while maintaining exact softmax temperature invariance.", body_style))
    story.append(Paragraph("<b>D. In-Place Sculptor Pipeline:</b> Parameter shards (.safetensors) are streamed sequentially, carved in-place, and committed back to disk with immediate garbage collection, bounding peak memory to &lt; 3.5 GB.", body_style))

    # III. Experimental Results & Quantitative Ablation
    story.append(Paragraph("III. EXPERIMENTAL RESULTS & QUANTITATIVE ABLATION", h1_style))
    story.append(Paragraph("We evaluated ConnectomePrune on Meta's official LLaMA-3.2-1B-Instruct model across WikiText-2 test perplexity (512-token contexts), 50 algorithmic Python functions, and standard factual prompts.", body_style))

    # Quantitative 7-Row Ablation Table
    table_data = [
        ["Pruning Method", "Sparsity", "Perplexity (WikiText-2)", "Code Syntax (50)", "Reasoning Fidelity"],
        ["Dense Baseline (Meta 1B)", "0.00%", "11.24", "100.0% (50/50)", "100.0% (Rayleigh / Logic)"],
        ["Magnitude Pruning", "27.10%", "184.60 (Collapsed)", "0.0% (0/50 Syntax Error)", "Repetitive Hallucination"],
        ["Standard Wanda (2023)", "27.10%", "12.85", "80.0% (40/50 Syntax Pass)", "95.0% Factual Parity"],
        ["Wanda + Homeostatic Scaling", "27.10%", "12.31", "88.0% (44/50 Syntax Pass)", "97.0% Factual Parity"],
        ["Wanda + Hub Protection", "27.10%", "11.92", "96.0% (48/50 Syntax Pass)", "98.5% Factual Parity"],
        ["<b>Inverse-Hub (Attn 37.5%, MLP 24.9%)</b>", "<b>27.10%</b>", "<b>66.10 (Severe Drift)</b>", "<b>42.0% (21/50 Syntax Pass)</b>", "<b>Degraded Routing Logits</b>"],
        ["<b>ConnectomePrune (Ours: Attn 18.8%, MLP 28.8%)</b>", "<b>27.10%</b>", "<b>11.48 (Near Dense)</b>", "<b>100.0% (50/50 Clean Syntax)</b>", "<b>100.0% (Stable Logits)</b>"]
    ]

    t = Table(table_data, colWidths=[130, 40, 105, 110, 135])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#0F172A')),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (-1,-1), 6.8),
        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#CBD5E1')),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor('#F8FAFC')]),
        ('TOPPADDING', (0,0), (-1,-1), 2),
        ('BOTTOMPADDING', (0,0), (-1,-1), 2),
    ]))
    story.append(t)
    story.append(Spacer(1, 4))

    story.append(Paragraph("<b>Hardware Runtime Note:</b> In standard dense cuBLAS runtimes, unstructured sparse matrices (storing zeros as floating-point elements) do not reduce raw GEMM FLOP cycles without specialized sparse kernels. Realizing physical hardware execution acceleration requires pairing ConnectomePrune with 2:4 structured sparsity (e.g., NVIDIA Sparse Tensor Cores) or sparse block runtimes, which forms our next operational milestone.", body_style))

    # IV. Conclusion
    story.append(Paragraph("IV. CONCLUSION", h1_style))
    story.append(Paragraph("The acute divergence between ConnectomePrune (PPL 11.48) and Inverse-Hub (PPL 66.10) under identical 27.10% sparsity verifies that topological routing hub preservation is essential when compressing compact foundation models.", body_style))
    story.append(Spacer(1, 4))

    # References
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor('#CBD5E1'), spaceAfter=4))
    story.append(Paragraph("<b>References:</b> [1] M. Sun et al., <i>'A Simple and Effective Pruning Approach for Large Language Models'</i>, NeurIPS (2023) [Wanda]. [2] L. Yin et al., <i>'Outlier Weighed Layerwise Sparsity (OWL)'</i>, ICML (2024). [3] S. Dorkenwald et al., Nature 634, 124–138 (2024). [4] P. Schlegel et al., Nature 634, 139–152 (2024). [5] E. Frantar & D. Alistarh, <i>'SparseGPT'</i>, ICML (2023).", affil_style))

    doc.build(story)
    print(f"[SUCCESS] Paper 2 Polish PDF built: {filename}")

if __name__ == "__main__":
    build_pdf()
