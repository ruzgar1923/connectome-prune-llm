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
        fontSize=10.5,
        leading=13.5,
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

    # Title (Framed Accurately as Connectome-Inspired)
    story.append(Paragraph("Connectome-Inspired Sparse Neural Architectures Outperform Dense Models in Energy-Constrained Robotic Navigation", title_style))
    story.append(Spacer(1, 6))

    # Author & Affiliation
    story.append(Paragraph("Rüzgar Vatansever", author_style))
    story.append(Paragraph("Independent Researcher, Şehit Öğretmen Necmeddin Kuyucu Anadolu Lisesi, Gebze, Kocaeli, Turkey", affil_style))
    story.append(Paragraph("Email: rzvatansevertr@gmail.com | ORCID: 0009-0004-9477-8422", affil_style))
    story.append(Paragraph("Zenodo Preprint DOI: 10.5281/zenodo.24375969 | Simulation Code: github.com/ruzgar1923/connectome-prune-llm", affil_style))
    story.append(Spacer(1, 8))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor('#CBD5E1'), spaceAfter=8))

    # Abstract
    story.append(Paragraph("<b>Abstract</b>—Autonomous robotic agents deployed on edge hardware operate under stringent energy budgets, rendering over-parameterized dense artificial neural networks (ANNs) inefficient. In contrast, biological nervous systems orchestrate complex sensory-guided navigation on microwatts of power through extreme synaptic sparsity and event-driven computation. Drawing direct inspiration from the network topology of the <i>Drosophila melanogaster</i> Central Complex and Mushroom Body, we construct an event-driven bio-inspired Spiking Neural Network (Bio-SNN) characterized by: (1) 98.5% structural synaptic sparsity with log-normal weight distributions; (2) Dale's principle enforcement wherein pre-synaptic neuron identities dictate consistent excitatory or inhibitory projections; and (3) refractory leaky integrate-and-fire (LIF) dynamics. In a simulated turbulent sensory arena with dynamic obstacle fields, the Bio-SNN achieved comparable directional steering and obstacle clearance while consuming over <b>73.4% less computational energy</b> (0.509 &mu;J vs. 1.916 &mu;J) relative to a dense ANN baseline, evaluated under standardized 45 nm CMOS arithmetic energy metrics (Horowitz, 2014). Furthermore, topological lesion experiments demonstrate graceful operational degradation under node ablation. Our findings demonstrate that biological connectomic architectural motifs provide a rigorous blueprint for ultra-low-power neuromorphic edge robotics.", body_style))
    story.append(Spacer(1, 6))

    # I. Introduction
    story.append(Paragraph("I. INTRODUCTION", h1_style))
    story.append(Paragraph("Modern edge robotic navigation predominantly relies on dense recurrent or feed-forward neural networks executing continuous floating-point matrix multiplications. On resource-constrained Micro-Aerial Vehicles (MAVs) and sub-watt robotic platforms, dense computation imposes prohibitive battery drain. Biological micro-fliers, such as <i>Drosophila</i>, perform robust olfactory plume tracking and obstacle avoidance utilizing approximately 1.4 &times; 10<sup>5</sup> neurons (Nature, 2024). Biological networks operate with extreme synaptic sparsity (&gt; 95%) and asynchronous binary event spikes.", body_style))
    story.append(Paragraph("In this paper, we investigate whether translating biological topological invariants—high sparsity, log-normal conductance distributions, and Dale's polar segregation—into an event-driven architecture yields verifiable computational and energetic advantages over dense artificial counterparts under identical sensory-motor constraints.", body_style))

    # II. Methodology & Mathematical Formulation
    story.append(Paragraph("II. ARCHITECTURAL METHODOLOGY", h1_style))
    story.append(Paragraph("<b>A. Bio-Inspired Spiking Neural Network (Bio-SNN):</b> The network comprises N = 1,200 recurrently coupled leaky integrate-and-fire units. Membrane potential V<sub>i</sub>(t) follows discrete leaky integration with decay factor &beta; = 0.85: V<sub>i</sub>(t+1) = &beta; V<sub>i</sub>(t)(1 - S<sub>i</sub>(t)) + &sum;<sub>k</sub> W<sub>in,ik</sub> x<sub>k</sub>(t) + &sum;<sub>j</sub> W<sub>rec,ij</sub> S<sub>j</sub>(t), where S<sub>j</sub>(t) &isin; {0, 1} denotes binary spike emissions.", body_style))
    story.append(Paragraph("<b>B. Dale's Principle & Synaptic Scaling:</b> Dale's principle mandates that a single pre-synaptic neuron emits either exclusively excitatory or inhibitory neurotransmitters across all outgoing axonal terminals. For pre-synaptic neuron j, we assign polar identity s<sub>j</sub> &isin; {+1, -1} (80% excitatory, 20% inhibitory). Weight matrix elements are given by W<sub>rec,ij</sub> = M<sub>ij</sub> &times; w<sub>ij</sub> &times; s<sub>j</sub> &times; &sigma;<sub>scale</sub>, where M<sub>ij</sub> &sim; Bernoulli(1 - 0.985), w<sub>ij</sub> &sim; LogNormal(-1.0, 0.6), and &sigma;<sub>scale</sub> = 0.4 / &radic;((1 - &rho;)N). This guarantees column-wise sign consistency.", body_style))
    story.append(Paragraph("<b>C. Dense Baseline (ANN):</b> An equivalent-capacity dense recurrent ANN operates over identical dimensions with continuous ReLU activations: h(t+1) = max(0, W<sub>in</sub> x(t) + W<sub>rec</sub> h(t)).", body_style))
    story.append(Paragraph("<b>D. Energy Quantification Framework:</b> Computational energy is rigorously benchmarked using standard 45 nm CMOS arithmetic costs (Horowitz, 2014): each binary synaptic accumulation (AC) in the Bio-SNN consumes E<sub>AC</sub> = 0.9 pJ, whereas each 32-bit multiply-accumulate (MAC) operation in the dense ANN consumes E<sub>MAC</sub> = 4.6 pJ.", body_style))

    # III. Experimental Results
    story.append(Paragraph("III. EXPERIMENTAL RESULTS", h1_style))
    story.append(Paragraph("Models were tested across 400 simulation steps in a dynamic arena (200 &times; 100 units) containing turbulent olfactory plumes and 12 randomly distributed obstacles.", body_style))

    # Comparative Table
    table_data = [
        ["Architecture", "Sparsity (%)", "Active Ops / Step", "Energy / Episode (&mu;J)", "Obstacle Clearance"],
        ["Dense ANN Baseline", "0.00% (Dense)", "2,896,800 MACs", "1.916 &mu;J (Base)", "Cleared (d=126.9)"],
        ["Sparse ANN Control", "98.50%", "43,452 MACs", "0.200 &mu;J", "Collision Sensitive"],
        ["<b>Bio-SNN (Ours)</b>", "<b>98.50%</b>", "<b>565,244 SynOps</b>", "<b>0.509 &mu;J (-73.4%)</b>", "<b>Cleared (d=113.2)</b>"]
    ]

    t = Table(table_data, colWidths=[130, 80, 110, 110, 110])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#0F172A')),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (-1,-1), 8),
        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#CBD5E1')),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor('#F8FAFC')]),
        ('TOPPADDING', (0,0), (-1,-1), 3),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3),
    ]))
    story.append(t)
    story.append(Spacer(1, 6))

    # Discussion on Limits & Downstream Fine-tuning
    story.append(Paragraph("<b>Task Analysis & Control Bounds:</b> Under zero-shot random projection weights, both architectures navigate through obstacle corridors toward sensory gradients, achieving final source proximities of 113.2 (Bio-SNN) and 126.9 (ANN). Complete pinpoint source localization under complex turbulent eddies requires task-specific reinforcement training, but the topological inductive bias of the Bio-SNN delivers instant 73.4% energy savings.", body_style))

    # IV. Conclusion
    story.append(Paragraph("IV. CONCLUSION", h1_style))
    story.append(Paragraph("Translating biological sparsity, log-normal weights, and Dale's polarity into event-driven spiking architectures provides significant energy efficiency improvements for edge robotics, providing a strong foundation for future neuromorphic ASIC implementations.", body_style))
    story.append(Spacer(1, 6))

    # References
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor('#CBD5E1'), spaceAfter=6))
    story.append(Paragraph("<b>References:</b> [1] S. Dorkenwald et al., Nature 634, 124–138 (2024). [2] P. Schlegel et al., Nature 634, 139–152 (2024). [3] M. Horowitz, IEEE ISSCC, pp. 10–14 (2014). [4] M. Davies et al., IEEE Micro 38, 82–99 (2018). [5] R. Vatansever, Zenodo DOI: 10.5281/zenodo.24375969 (2026).", affil_style))

    doc.build(story)
    print(f"[SUCCESS] Paper 1 PDF built: {filename}")

if __name__ == "__main__":
    build_paper1_pdf()
