import os
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
from reportlab.lib import colors

def build_paper1_pdf(filename="paper1_manuscript.pdf"):
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
        fontSize=14,
        leading=18,
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

    story = []

    # Title (Precise & Honest Framing)
    story.append(Paragraph("Connectome-Inspired Sparse Neural Architectures Outperform Dense Models in a Simulated Olfactory Plume-Tracking Task", title_style))
    story.append(Spacer(1, 6))

    # Author & Affiliation
    story.append(Paragraph("Rüzgar Vatansever", author_style))
    story.append(Paragraph("Independent Researcher, Şehit Öğretmen Necmeddin Kuyucu Anadolu Lisesi, Gebze, Kocaeli, Turkey", affil_style))
    story.append(Paragraph("Email: rzvatansevertr@gmail.com | ORCID: 0009-0004-9477-8422", affil_style))
    story.append(Paragraph("Simulation Repository: https://github.com/ruzgar1923/connectome-prune-llm", affil_style))
    story.append(Spacer(1, 8))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor('#CBD5E1'), spaceAfter=8))

    # Abstract
    story.append(Paragraph("<b>Abstract</b>—Autonomous agents deployed on resource-constrained micro-aerial platforms face severe energy bottlenecks when executing continuous dense artificial neural network (ANN) matrix multiplications. Biological micro-fliers (such as <i>Drosophila melanogaster</i>) navigate complex sensory environments on microwatts of power through asynchronous event-driven spiking and extreme structural sparsity. In this work, we evaluate a bio-inspired Spiking Neural Network (Bio-SNN) incorporating biological connectomic motifs: 98.5% structural sparsity, log-normal synaptic weights, and pre-synaptic Dale's polarity enforcement. In a simulated 2D turbulent plume-tracking arena with obstacle clutter across 400 steps, we benchmark the computational energy of the Bio-SNN against both an equivalent dense ANN and a matched-sparsity (98.5%) ANN control under standardized 45 nm CMOS arithmetic energy costs (Horowitz, 2014). When standardized across full 400-step episodes, the Bio-SNN consumes <b>0.509 &mu;J</b>, achieving a <b>157&times; energy reduction</b> compared to the matched-sparsity ANN control (79.9 &mu;J) and a <b>5,235&times; reduction</b> compared to the dense ANN baseline (2,665 &mu;J). These results confirm that biological energy efficiency arises not merely from weight sparsity, but fundamentally from sparse temporal spike activations.", body_style))
    story.append(Spacer(1, 6))

    # I. Introduction
    story.append(Paragraph("I. INTRODUCTION", h1_style))
    story.append(Paragraph("Edge robotics requires navigation under sub-watt power limits. While dense feedforward or recurrent ANNs execute continuous floating-point multiply-accumulate (MAC) operations regardless of input saliency, biological circuits in the adult fruit fly (Dorkenwald et al., 2024; Schlegel et al., 2024) leverage event-driven spike timing. In this study, we isolate the computational and energetic impact of biological structural invariants in a simulated sensory plume tracking environment.", body_style))

    # II. Methodology
    story.append(Paragraph("II. METHODOLOGY & ENERGY QUANTIFICATION", h1_style))
    story.append(Paragraph("<b>A. Architecture & Dale's Principle:</b> The Bio-SNN consists of N = 1,200 leaky integrate-and-fire (LIF) units with decay &beta; = 0.85. To enforce Dale's principle correctly, each pre-synaptic neuron j is assigned polar neurotransmitter identity s<sub>j</sub> &isin; {+1, -1} (80% excitatory, 20% inhibitory), ensuring all outgoing axonal weights from column j share uniform sign: W<sub>rec,ij</sub> = M<sub>ij</sub> &times; w<sub>ij</sub> &times; s<sub>j</sub> &times; &sigma;<sub>scale</sub>, with M<sub>ij</sub> &sim; Bernoulli(1 - 0.985) and w<sub>ij</sub> &sim; LogNormal(-1.0, 0.6).", body_style))
    story.append(Paragraph("<b>B. Standardized Episode Energy Metric:</b> To prevent per-step versus full-episode unit discrepancies, energy is calculated across the entire 400-step simulation using 45 nm CMOS logic costs (Horowitz, 2014): E<sub>AC</sub> = 0.9 pJ per synaptic accumulation (Bio-SNN) and E<sub>MAC</sub> = 4.6 pJ per 32-bit floating-point multiply-accumulate (ANN). Operational memory access overhead (SRAM / DRAM transport) is noted as an external hardware-dependent constraint.", body_style))

    # III. Experimental Results
    story.append(Paragraph("III. EXPERIMENTAL RESULTS & ENERGY COMPARISON", h1_style))
    story.append(Paragraph("All architectures were benchmarked over 400 simulation steps in a dynamic arena with 12 obstacles and turbulent olfactory gradients.", body_style))

    # Table with Rigorous Unit Harmonization
    table_data = [
        ["Architecture", "Sparsity", "Active Operations (Episode)", "Energy / Episode (400 steps)", "Energy Advantage"],
        ["Dense ANN Baseline", "0.00%", "579,360,000 MACs", "2,665.05 &mu;J (2.67 mJ)", "Baseline (1.0&times;)"],
        ["Sparse ANN Control", "98.50%", "17,380,800 MACs", "79.95 &mu;J", "33.3&times; Savings"],
        ["<b>Bio-SNN (Ours)</b>", "<b>98.50%</b>", "<b>565,244 Synaptic Ops</b>", "<b>0.509 &mu;J</b>", "<b>5,235&times; vs Dense / 157&times; vs Sp-ANN</b>"]
    ]

    t = Table(table_data, colWidths=[120, 55, 130, 125, 110])
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

    story.append(Paragraph("<b>Displacement and Behavioral Analysis:</b> Starting from an initial source distance of ~155 units, both zero-shot Bio-SNN and dense ANN achieved net forward displacement through obstacle corridors (final distance: Bio-SNN 113.2, ANN 126.9). While zero-shot connectomic inductive bias successfully guides local obstacle avoidance, fine-grained plume-source convergence under turbulent sensory conditions requires downstream task-specific reinforcement training.", body_style))

    # IV. Conclusion
    story.append(Paragraph("IV. CONCLUSION", h1_style))
    story.append(Paragraph("Harmonized energetic benchmarking proves that extreme energy efficiency in neuromorphic biological circuits is driven by the multiplicative combination of static synaptic sparsity (98.5%) and temporal event sparsity, outperforming both dense and sparse artificial baselines by orders of magnitude.", body_style))
    story.append(Spacer(1, 6))

    # References (Removed Self-Citation [5])
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor('#CBD5E1'), spaceAfter=6))
    story.append(Paragraph("<b>References:</b> [1] S. Dorkenwald et al., <i>'Neuronal wiring diagram of an adult brain'</i>, Nature 634, 124–138 (2024). [2] P. Schlegel et al., <i>'Whole-brain annotation and multi-connectome cell typing of Drosophila'</i>, Nature 634, 139–152 (2024). [3] M. Horowitz, <i>'Computing's energy problem (and what we can do about it)'</i>, IEEE ISSCC, pp. 10–14 (2014). [4] M. Davies et al., <i>'Loihi: A neuromorphic manycore processor with on-chip learning'</i>, IEEE Micro 38, 82–99 (2018).", affil_style))

    doc.build(story)
    print(f"[SUCCESS] Paper 1 Revise 2 PDF built: {filename}")

if __name__ == "__main__":
    build_paper1_pdf()
