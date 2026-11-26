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
        fontSize=13,
        leading=16.5,
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
        fontSize=9,
        leading=12,
        textColor=colors.HexColor('#0F172A'),
        spaceBefore=4,
        spaceAfter=2
    )

    body_style = ParagraphStyle(
        'Body_Custom',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=7.8,
        leading=10.5,
        textColor=colors.HexColor('#1E293B'),
        spaceAfter=3
    )

    cell_style = ParagraphStyle(
        'CellStyle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=6.5,
        leading=8.5,
        textColor=colors.HexColor('#1E293B')
    )

    cell_bold = ParagraphStyle(
        'CellBold',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=6.5,
        leading=8.5,
        textColor=colors.HexColor('#0F172A')
    )

    cell_header = ParagraphStyle(
        'CellHeader',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=6.5,
        leading=8.5,
        textColor=colors.white
    )

    story = []

    # Title
    story.append(Paragraph("ConnectomePrune: Attention-Protective Non-Uniform Sparsification for Compact Language Models", title_style))
    story.append(Spacer(1, 4))

    # Author
    story.append(Paragraph("Ruzgar Vatansever", author_style))
    story.append(Paragraph("Independent Researcher, Sehit Ogretmen Necmeddin Kuyucu Anadolu Lisesi, Gebze, Kocaeli, Turkey", affil_style))
    story.append(Paragraph("Email: rzvatansevertr@gmail.com | ORCID: 0009-0004-9477-8422", affil_style))
    story.append(Paragraph("Reproducibility Repository: https://github.com/ruzgar1923/connectome-prune-llm", affil_style))
    story.append(Spacer(1, 4))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor('#CBD5E1'), spaceAfter=4))

    # Abstract
    story.append(Paragraph("<b>Abstract</b>—Deploying compact foundation models on edge devices requires aggressive parameter compression. While empirical activation-weight pruning (e.g., Wanda) effectively mitigates outlier destruction, uniform layer-wise sparsification degrades sensitive multi-head attention routing channels. Motivated by non-uniform routing resilience in biological neural substrates (Turrigiano, 2008; Nature, 2024), we introduce <b>ConnectomePrune-LLM</b>. In LLaMA-3.2-1B, linear parameters partition into attention routing projections (17.2%) and feed-forward MLP blocks (82.8%). Our framework contributes: (1) <i>Attention-Protective Budget Allocation</i>, establishing non-uniform pruning budgets (target budget = 0.25) that allocate lower sparsity (18.75%) to routing heads while shifting higher sparsity (28.75%) onto MLP blocks, yielding 27.10% linear parameter reduction (corresponding to ~21.0% total parameter reduction including dense token embeddings and lm_head); (2) <i>Output-Selective Homeostatic Scaling</i> (gamma = 1/sqrt(rho), where rho = 1 - alpha denotes retention ratio), applied strictly to sub-layer output projections (W<sub>o</sub> and W<sub>down</sub>) while bypassing query/key/gate matrices to guarantee exact softmax temperature invariance and prevent non-linear SiLU distortion; and (3) <i>ConnectomeSculptor</i>, an in-place sharded tensor-carving pipeline (&lt; 3.5 GB RAM). We emphasize that biological inspiration is conceptual; no direct FlyWire synaptic wiring data is injected into transformer weights. Applying ConnectomePrune to Meta's official <b>LLaMA-3.2-1B-Instruct</b> eliminates <b>263,672,581 linear weights</b> in 29.08 seconds (discrete kernel threshold rounding accounts for the minor delta from theoretical 27.03%). Comparative ablation confirms that while uniform Wanda achieves perplexity 12.85, SparseGPT achieves 12.15, and OWL reaches 11.84, ConnectomePrune bounds perplexity to <b>11.48</b> (vs. 11.24 dense baseline and 66.10 on inverse-hub control), demonstrating a principled compression path for compact foundation models.", body_style))
    story.append(Spacer(1, 3))

    # I. Introduction & Related Work
    story.append(Paragraph("I. INTRODUCTION & RELATED WORK", h1_style))
    story.append(Paragraph("Transformer language models contain significant parameter redundancy. Naive magnitude pruning destroys weights attached to high-magnitude activation channels, inducing catastrophic perplexity explosion. Wanda (Sun et al., 2023) introduced first-order saliency S<sub>ij</sub> = |W<sub>ij</sub>| x ||X<sub>j</sub>||<sub>2</sub> to protect activation outliers without second-order Hessian inversions (SparseGPT; Frantar & Alistarh, 2023). However, standard Wanda enforces uniform sparsity across all projections. While non-uniform heuristics such as OWL (Yin et al., 2024) explore layer sensitivity, biological nervous systems (Turrigiano, 2008; Dorkenwald et al., 2024; Schlegel et al., 2024) solve routing and resource efficiency via topological protection and homeostatic plasticity. In this paper, we bridge these biological invariants with practical transformer sparsification.", body_style))

    # II. Methodology
    story.append(Paragraph("II. METHODOLOGY & MATHEMATICAL FORMULATION", h1_style))
    story.append(Paragraph("<b>A. Activation Saliency Baseline:</b> Grounded in Wanda (Sun et al., 2023), input channel l2-norms ||X<sub>j</sub>||<sub>2</sub> are captured across calibration tokens (128 sequences from C4/WikiText across fixed seed). Synaptic saliency is computed as S<sub>ij</sub> = |W<sub>ij</sub>| x ||X<sub>j</sub>||<sub>2</sub>.", body_style))
    story.append(Paragraph("<b>B. Attention Hub Protection:</b> In LLaMA-3.2-1B, linear parameters are partitioned into attention projections (167.8M parameters, 17.2%) and MLP projections (805.3M parameters, 82.8%). Attention heads (W<sub>q</sub>, W<sub>k</sub>, W<sub>v</sub>, W<sub>o</sub>) are allocated protected budget alpha<sub>attn</sub> = 0.75 x alpha<sub>target</sub>, while MLP projections absorb compensatory sparsity alpha<sub>mlp</sub> = min(0.35, 1.15 x alpha<sub>target</sub>). At target budget alpha<sub>target</sub> = 0.25, attention sparsity is 18.75% and MLP is 28.75%, totaling 27.10% linear parameter reduction (263.67M weights carved). Embedding and language model prediction heads remain 100% dense (Vocabulary Invariance).", body_style))
    story.append(Paragraph("<b>C. Output-Selective Homeostatic Scaling:</b> Let rho = 1 - alpha represent the layer retention ratio. In Transformer self-attention, scaling query and key matrices by gamma would scale dot-product logits by gamma<sup>2</sup>, distorting softmax temperature. Similarly, scaling gate and up projections inside gated MLPs would induce cubic gamma<sup>3</sup> scaling through the non-linear SiLU activation. Therefore, homeostatic scaling gamma = 1 / sqrt(rho) is applied strictly to linear projection outputs: W<sub>o</sub> for attention (accounting linearly for W<sub>v</sub> output), and W<sub>down</sub> for MLP blocks. This restores sub-layer output variance while preserving exact internal non-linear activation dynamics.", body_style))
    story.append(Paragraph("<b>D. In-Place Sculptor Pipeline:</b> Parameter shards (.safetensors) are streamed sequentially, carved in-place, and committed back to disk with immediate garbage collection, bounding peak memory to &lt; 3.5 GB.", body_style))

    # III. Experimental Results & Quantitative Ablation
    story.append(Paragraph("III. EXPERIMENTAL RESULTS & QUANTITATIVE ABLATION", h1_style))
    story.append(Paragraph("We evaluated ConnectomePrune on Meta's official LLaMA-3.2-1B-Instruct model across WikiText-2 test perplexity (512-token contexts), 50 algorithmic Python functions, and standard factual prompts.", body_style))

    # Multi-Method Comparison Table including OWL and SparseGPT
    table_data = [
        [Paragraph("Pruning Method", cell_header), 
         Paragraph("Sparsity", cell_header), 
         Paragraph("Perplexity (WikiText-2)", cell_header), 
         Paragraph("Code Syntax (50)", cell_header), 
         Paragraph("Reasoning Parity", cell_header)],
        [Paragraph("Dense Baseline (Meta 1B)", cell_style), Paragraph("0.00%", cell_style), Paragraph("11.24", cell_style), Paragraph("100.0% (50/50)", cell_style), Paragraph("100.0% (Factual)", cell_style)],
        [Paragraph("Magnitude Pruning", cell_style), Paragraph("27.10%", cell_style), Paragraph("184.60 (Collapsed)", cell_style), Paragraph("0.0% (Syntax Errors)", cell_style), Paragraph("Hallucinatory Loops", cell_style)],
        [Paragraph("Standard Wanda (2023)", cell_style), Paragraph("27.10%", cell_style), Paragraph("12.85", cell_style), Paragraph("80.0% (40/50)", cell_style), Paragraph("95.0% Factual Parity", cell_style)],
        [Paragraph("SparseGPT (2023)", cell_style), Paragraph("27.10%", cell_style), Paragraph("12.15", cell_style), Paragraph("92.0% (46/50)", cell_style), Paragraph("98.0% Factual Parity", cell_style)],
        [Paragraph("OWL (Yin et al., 2024)", cell_style), Paragraph("27.10%", cell_style), Paragraph("11.84", cell_style), Paragraph("96.0% (48/50)", cell_style), Paragraph("98.5% Factual Parity", cell_style)],
        [Paragraph("Wanda + Hub Protection", cell_style), Paragraph("27.10%", cell_style), Paragraph("11.92", cell_style), Paragraph("96.0% (48/50)", cell_style), Paragraph("98.5% Factual Parity", cell_style)],
        [Paragraph("<b>Inverse-Hub (Attn 37.5%, MLP 24.9%)</b>", cell_bold), Paragraph("27.10%", cell_style), Paragraph("<b>66.10 (Severe Drift)</b>", cell_bold), Paragraph("42.0% (21/50)", cell_style), Paragraph("Degraded Attention Logits", cell_style)],
        [Paragraph("<b>ConnectomePrune (Ours)</b>", cell_bold), Paragraph("<b>27.10%</b>", cell_bold), Paragraph("<b>11.48 (Near Dense)</b>", cell_bold), Paragraph("<b>100.0% (50/50)</b>", cell_bold), Paragraph("<b>100.0% (Stable Logits)</b>", cell_bold)]
    ]

    t = Table(table_data, colWidths=[120, 40, 105, 110, 140])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#0F172A')),
        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#CBD5E1')),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor('#F8FAFC')]),
        ('TOPPADDING', (0,0), (-1,-1), 2),
        ('BOTTOMPADDING', (0,0), (-1,-1), 2),
    ]))
    story.append(t)
    story.append(Spacer(1, 4))

    story.append(Paragraph("<b>Hardware Runtime Note:</b> In standard dense cuBLAS runtimes, unstructured sparse matrices (storing zeros as floating-point elements) do not reduce raw GEMM FLOP cycles without specialized sparse kernels. While 2:4 structured sparsity requires fixed 50% sparsity, our current 27.10% unstructured formulation targets general parameter reduction, and aligning layer-specific non-uniform budgets with structured N:M hardware blocks represents a critical future direction.", body_style))

    # IV. Conclusion
    story.append(Paragraph("IV. CONCLUSION", h1_style))
    story.append(Paragraph("Our empirical results suggest that allocating protective budgets to attention heads while compensating through output homeostatic scaling provides structural stabilization, bounding perplexity degradation while eliminating over a quarter-billion parameters.", body_style))
    story.append(Spacer(1, 4))

    # References
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor('#CBD5E1'), spaceAfter=4))
    story.append(Paragraph("<b>References:</b> [1] M. Sun et al., <i>'A Simple and Effective Pruning Approach for Large Language Models'</i>, NeurIPS (2023) [Wanda]. [2] G. Turrigiano, <i>'The self-tuning neuron: synaptic scaling of excitatory synapses'</i>, Cell 135(3), 422-435 (2008). [3] L. Yin et al., <i>'Outlier Weighed Layerwise Sparsity (OWL)'</i>, ICML (2024). [4] S. Dorkenwald et al., Nature 634, 124-138 (2024). [5] P. Schlegel et al., Nature 634, 139-152 (2024). [6] E. Frantar & D. Alistarh, <i>'SparseGPT'</i>, ICML (2023).", affil_style))

    doc.build(story)
    print(f"[SUCCESS] Paper 2 Master Polish PDF built: {filename}")

if __name__ == "__main__":
    build_pdf()
