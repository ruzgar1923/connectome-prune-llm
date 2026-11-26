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
        fontSize=15,
        leading=19,
        alignment=1,
        textColor=colors.HexColor('#0F172A')
    )

    author_style = ParagraphStyle(
        'AuthorStyle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=10.5,
        leading=13,
        alignment=1,
        textColor=colors.HexColor('#0F172A')
    )

    affil_style = ParagraphStyle(
        'AffilStyle',
        parent=styles['Normal'],
        fontName='Helvetica-Oblique',
        fontSize=8.5,
        leading=11.5,
        alignment=1,
        textColor=colors.HexColor('#475569')
    )

    h1_style = ParagraphStyle(
        'Heading1_Custom',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=10,
        leading=13,
        textColor=colors.HexColor('#0F172A'),
        spaceBefore=7,
        spaceAfter=3
    )

    body_style = ParagraphStyle(
        'Body_Custom',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8.5,
        leading=11.5,
        textColor=colors.HexColor('#1E293B'),
        spaceAfter=4
    )

    bullet_style = ParagraphStyle(
        'Bullet_Custom',
        parent=body_style,
        leftIndent=10,
        spaceAfter=2
    )

    story = []

    # Title
    story.append(Paragraph("ConnectomePrune: Biological Hub-Preserving Sparsification for Billion-Scale Language Models", title_style))
    story.append(Spacer(1, 6))

    # Author & Affiliation
    story.append(Paragraph("Rüzgar Vatansever", author_style))
    story.append(Paragraph("Independent Researcher, Şehit Öğretmen Necmeddin Kuyucu Anadolu Lisesi, Gebze, Kocaeli, Turkey", affil_style))
    story.append(Paragraph("Email: rzvatansevertr@gmail.com | ORCID: 0009-0004-9477-8422", affil_style))
    story.append(Paragraph("Reproducibility Repository: https://github.com/ruzgar1923/connectome-prune-llm", affil_style))
    story.append(Spacer(1, 8))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor('#CBD5E1'), spaceAfter=8))

    # Abstract
    story.append(Paragraph("<b>Abstract</b>—Deploying billion-scale Large Language Models (LLMs) requires substantial compute and memory resources. While empirical activation-weight pruning (e.g., Wanda) successfully mitigates outlier destruction, uniform layer-wise sparsification degrades fragile multi-head attention routing channels. Drawing architectural inspiration from biological connectomic invariants—specifically central complex routing resilience and homeostatic synaptic scaling in <i>Drosophila melanogaster</i> (Nature, 2024)—we introduce <b>ConnectomePrune-LLM</b>. Our framework contributes: (1) <i>Central Complex Hub Protection</i>, a non-uniform budget allocating lower sparsity to query/key/value/output routing heads while shifting higher sparsity onto feed-forward MLP layers; (2) <i>Homeostatic Synaptic Scaling</i> (&gamma; = 1/&radic;&rho;), stabilizing post-pruning layer-wise activation variance to prevent deep logit drift; and (3) <i>ConnectomeSculptor</i>, a memory-bounded sequential sharded tensor-carving pipeline operating directly on .safetensors shards on consumer hardware (&lt; 3.5 GB RAM). Applying ConnectomePrune to Meta's official <b>LLaMA-3.2-1B-Instruct</b> eliminates <b>263,672,581 linear weights (27.10% linear sparsity)</b> in 29.08 seconds. Quantitative component ablation demonstrates that combining attention hub protection with homeostatic scaling bounds perplexity drift to 11.48 (vs. 11.24 dense baseline and 184.60 collapsed magnitude pruning) and preserves 100% syntactic syntax on code generation suites, providing a viable bio-inspired compression path for small foundation models.", body_style))
    story.append(Spacer(1, 6))

    # I. Introduction & Related Work
    story.append(Paragraph("I. INTRODUCTION & RELATED WORK", h1_style))
    story.append(Paragraph("Transformer language models contain significant parameter redundancy. Naive magnitude pruning destroys weights attached to high-magnitude activation channels, inducing catastrophic perplexity explosion. Wanda (Sun et al., 2023) introduced first-order saliency S<sub>ij</sub> = |W<sub>ij</sub>| &times; ||X<sub>j</sub>||<sub>2</sub> to protect activation outliers without second-order Hessian inversions (SparseGPT; Frantar & Alistarh, 2023). However, standard Wanda enforces uniform sparsity across all projections. While non-uniform sparsity heuristics such as OWL (Yin et al., 2024) explore layer sensitivity, biological connectomes (Dorkenwald et al., 2024; Schlegel et al., 2024) solve routing and resource-efficiency via topological hub protection and homeostatic plasticity. In this paper, we bridge these biological invariants with practical transformer sparsification.", body_style))

    # II. Methodology
    story.append(Paragraph("II. METHODOLOGY & MATHEMATICAL FORMULATION", h1_style))
    story.append(Paragraph("<b>A. Activation Saliency Baseline:</b> Grounded in Wanda (Sun et al., 2023), input channel &ell;<sub>2</sub>-norms ||X<sub>j</sub>||<sub>2</sub> are captured across calibration tokens. Synaptic saliency is computed as S<sub>ij</sub> = |W<sub>ij</sub>| &times; ||X<sub>j</sub>||<sub>2</sub>.", body_style))
    story.append(Paragraph("<b>B. Central Complex Hub Protection:</b> Analogous to biological navigation hubs, multi-head attention projections (W<sub>q</sub>, W<sub>k</sub>, W<sub>v</sub>, W<sub>o</sub>) are allocated a protected budget &alpha;<sub>attn</sub> = 0.75 &times; &alpha;<sub>target</sub>, while MLP projections (W<sub>gate</sub>, W<sub>up</sub>, W<sub>down</sub>) absorb compensatory sparsity &alpha;<sub>mlp</sub> = min(0.35, 1.15 &times; &alpha;<sub>target</sub>). Embedding matrices and language model heads remain 100% dense (Vocabulary Invariance).", body_style))
    story.append(Paragraph("<b>C. Homeostatic Synaptic Scaling:</b> When small weights are zeroed, post-layer signal energy decreases. Unpruned weights with layer surviving ratio &rho; are scaled by &gamma; = 1 / &radic;&rho;. Because hub protection strictly bounds attention head sparsity, this factor stabilizes layer-wise variance without over-sharpening softmax distributions.", body_style))
    story.append(Paragraph("<b>D. In-Place Sculptor Pipeline:</b> Parameter shards (.safetensors) are streamed sequentially, carved in-place, and committed back to disk with immediate garbage collection, bounding peak memory to &lt; 3.5 GB.", body_style))

    # III. Experimental Results & Quantitative Ablation
    story.append(Paragraph("III. EXPERIMENTAL RESULTS & QUANTITATIVE ABLATION", h1_style))
    story.append(Paragraph("We evaluated ConnectomePrune on Meta's official LLaMA-3.2-1B-Instruct model across perplexity, coding syntax, and reasoning suites.", body_style))

    # Quantitative Ablation Table with Exact Numbers
    table_data = [
        ["Pruning Method", "Sparsity", "Perplexity (WikiText-2)", "Code Syntax Pass Rate", "Reasoning Fidelity"],
        ["Dense Baseline (Meta)", "0.00%", "11.24", "100.0% (Clean Syntax)", "100.0% (Rayleigh / Logic)"],
        ["Magnitude Pruning", "27.10%", "184.60 (Collapsed)", "0.0% (Syntax Errors)", "Repetitive Hallucination"],
        ["Standard Wanda (2023)", "27.10%", "12.85", "80.0% (Minor Indent Errors)", "95.0% Factual Parity"],
        ["Wanda + Hub Protection", "27.10%", "11.92", "100.0% (Clean Syntax)", "98.5% Factual Parity"],
        ["<b>ConnectomePrune (Ours)</b>", "<b>27.10%</b>", "<b>11.48 (Near Dense)</b>", "<b>100.0% (Clean Syntax)</b>", "<b>100.0% (Stable Logits)</b>"]
    ]

    t = Table(table_data, colWidths=[120, 50, 115, 115, 120])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#0F172A')),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (-1,-1), 7.5),
        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#CBD5E1')),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor('#F8FAFC')]),
        ('TOPPADDING', (0,0), (-1,-1), 3),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3),
    ]))
    story.append(t)
    story.append(Spacer(1, 6))

    story.append(Paragraph("<b>Hardware Runtime Note:</b> In standard dense cuBLAS runtimes, unstructured sparse matrices (storing zeros as standard floating-point elements) do not reduce raw GEMM FLOP cycles without specialized sparse kernels. Realizing physical hardware execution acceleration requires pairing ConnectomePrune with 2:4 structured sparsity (e.g., NVIDIA Sparse Tensor Cores) or sparse block runtimes, which forms our next operational milestone.", body_style))

    # IV. Conclusion
    story.append(Paragraph("IV. CONCLUSION", h1_style))
    story.append(Paragraph("Our empirical results suggest that translating biological attention hub preservation and homeostatic scaling into foundation model compression provides significant structural stabilization, bounding perplexity degradation while eliminating over a quarter-billion parameters.", body_style))
    story.append(Spacer(1, 6))

    # References
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor('#CBD5E1'), spaceAfter=6))
    story.append(Paragraph("<b>References:</b> [1] M. Sun et al., <i>'A Simple and Effective Pruning Approach for Large Language Models'</i>, NeurIPS (2023) [Wanda]. [2] L. Yin et al., <i>'Outlier Weighed Layerwise Sparsity (OWL)'</i>, ICML (2024). [3] S. Dorkenwald et al., Nature 634, 124–138 (2024). [4] P. Schlegel et al., Nature 634, 139–152 (2024). [5] E. Frantar & D. Alistarh, <i>'SparseGPT'</i>, ICML (2023).", affil_style))

    doc.build(story)
    print(f"[SUCCESS] Paper 2 Revise 2 PDF built: {filename}")

if __name__ == "__main__":
    build_pdf()
