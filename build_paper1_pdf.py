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
        fontSize=11.8,
        leading=15.0,
        alignment=1,
        textColor=colors.HexColor('#0F172A')
    )

    author_style = ParagraphStyle(
        'AuthorStyle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=9.2,
        leading=11.8,
        alignment=1,
        textColor=colors.HexColor('#0F172A')
    )

    affil_style = ParagraphStyle(
        'AffilStyle',
        parent=styles['Normal'],
        fontName='Helvetica-Oblique',
        fontSize=7.6,
        leading=10.2,
        alignment=1,
        textColor=colors.HexColor('#475569')
    )

    h1_style = ParagraphStyle(
        'Heading1_Custom',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=8.6,
        leading=11.2,
        textColor=colors.HexColor('#0F172A'),
        spaceBefore=3.5,
        spaceAfter=1.8
    )

    body_style = ParagraphStyle(
        'Body_Custom',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=7.4,
        leading=9.8,
        textColor=colors.HexColor('#1E293B'),
        spaceAfter=2.2
    )

    cell_style = ParagraphStyle(
        'CellStyle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=6.4,
        leading=8.3,
        textColor=colors.HexColor('#1E293B')
    )

    cell_bold = ParagraphStyle(
        'CellBold',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=6.4,
        leading=8.3,
        textColor=colors.HexColor('#0F172A')
    )

    cell_header = ParagraphStyle(
        'CellHeader',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=6.4,
        leading=8.3,
        textColor=colors.white
    )

    story = []

    # Title
    story.append(Paragraph("Sparse Spiking Neural Architectures Reduce Arithmetic Energy in a Simulated Olfactory Plume-Tracking Task", title_style))
    story.append(Spacer(1, 3))

    # Author
    story.append(Paragraph("Ruzgar Vatansever", author_style))
    story.append(Paragraph("Independent Researcher, Sehit Ogretmen Necmeddin Kuyucu Anadolu Lisesi, Gebze, Kocaeli, Turkey", affil_style))
    story.append(Paragraph("Email: rzvatansevertr@gmail.com | ORCID: 0009-0004-9477-8422", affil_style))
    story.append(Paragraph("DOI: 10.5281/zenodo.24375969 | TUBITAK ARBIS ID: TBTK-0093-7414 | Repository: https://github.com/ruzgar1923/connectome-snn-benchmark", affil_style))
    story.append(Spacer(1, 3))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor('#CBD5E1'), spaceAfter=3))

    # Abstract
    story.append(Paragraph("<b>Abstract</b>—Autonomous robotic edge hardware operates under stringent sub-watt power budgets. While dense artificial neural networks (ANNs) execute continuous floating-point matrix multiplications, insect connectomes demonstrate sparse, non-uniform synaptic connectivity. In this work, we evaluate a Spiking Neural Network incorporating biologically motivated constraints (Bio-SNN): 98.5% structural sparsity, log-normal synaptic weights, and pre-synaptic Dale's polarity enforcement across an empirical firing rate of 6.5% (93.5% temporal sparsity, beta = 0.95, tau<sub>mem</sub> = 20 ms). Across 400 simulation steps in a 2D turbulent plume arena (n = 25 independent random-weight network seeds), we quantify arithmetic energy consumption against dense, uniform sparse (98.5%), and activation-sparse ANNs under standardized 45 nm CMOS logic costs (Horowitz, 2014). Recurrent SNN transmissions consume exactly 0.509 uJ across 565,244 SynOps. Fully accounting for membrane leak updates and active I/O projections yields a total arithmetic budget of <b>2.16 uJ (dedicated neuromorphic event-driven bit-shift)</b> to <b>18.17 uJ (standard FP32)</b>. At the recurrent core level, Bio-SNN achieves a <b>78.0x reduction</b> over a uniform sparse ANN core (39.74 uJ) and a <b>5.1x reduction</b> (2.58 uJ vs 0.509 uJ) over an activation-sparse ANN core, isolating the CMOS logic ratio E<sub>MAC</sub> / E<sub>AC</sub> ~ 5.11. End-to-end, Bio-SNN delivers <b>4.1x savings</b> over matched activation-sparse ANN with duty-cycle I/O (8.88 uJ), <b>25.5x savings</b> over uniform sparse ANN (55.20 uJ), and <b>1,233x savings</b> over dense ANN (2,665.06 uJ). Untrained Bio-SNN agents achieved source proximity (113.2 +/- 8.4) significantly outperforming a random walk null control (152.4 +/- 14.1, p &lt; 10<sup>-12</sup>), shuffled-input control (149.6 +/- 13.2), and dense ANN (126.9 +/- 9.2, p = 1.6 x 10<sup>-6</sup>).", body_style))
    story.append(Spacer(1, 2))

    # I. Introduction
    story.append(Paragraph("I. INTRODUCTION", h1_style))
    story.append(Paragraph("Edge robotics requires sensorimotor navigation under strict energy constraints. While dense ANNs execute heavy continuous GEMMs, recent connectomic reconstructions (Dorkenwald et al., Nature 2024; Schlegel et al., Nature 2024) provide anatomical wiring maps revealing sparse, heavy-tailed synaptic connectivity. Motivated by these architectural characteristics, we evaluate whether biologically motivated structural sparsity and event-driven accumulation deliver genuine arithmetic energy advantages over matched continuous and activation-sparse ANNs in continuous plume tracking.", body_style))
    story.append(Spacer(1, 2))

    # II. Network Architecture & Simulation Physics
    story.append(Paragraph("II. ARCHITECTURE & ARITHMETIC ENERGY MODELING", h1_style))
    story.append(Paragraph("<b>A. Biological Structural Modeling:</b> The recurrent reservoir comprises N = 1,200 LIF neurons with V<sub>th</sub> = 1.0 and beta = 0.95 (biophysical membrane time constant tau<sub>mem</sub> = 20 ms with integration step dt = 1 ms). In accordance with pre-synaptic Dale's principle, polarity s<sub>j</sub> in {-1, +1} (80% excitatory, 20% inhibitory) is assigned strictly to source neuron j: W<sub>rec,ij</sub> = M<sub>ij</sub> x w<sub>ij</sub> x s<sub>j</sub> x sigma<sub>scale</sub>, where M<sub>ij</sub> ~ Bernoulli(1 - rho), w<sub>ij</sub> ~ LogNormal(-1.0, 0.6), and sigma<sub>scale</sub> = 0.4 / sqrt((1 - rho)N) with structural sparsity rho = 0.985. All metrics are evaluated under this unified biophysical parameterization.", body_style))
    story.append(Paragraph("<b>B. Exact Arithmetic Energy Component Breakdown:</b> Energy is evaluated across 400-step episodes under standardized 45 nm CMOS logic costs (Horowitz, 2014): accumulation E<sub>AC</sub> = 0.9 pJ and multiply-accumulate E<sub>MAC</sub> = 4.6 pJ. Dense ANN incurs 579.36M MACs (2,665.06 uJ). Symmetrically dimensioned uniform sparse ANN (8.64M recurrent MACs + 3.36M dense I/O MACs = 12.0M MACs) totals 55.20 uJ.<br/>"
                           "<i>1) Bio-SNN Recurrent Core:</i> Across 400 steps with mean firing rate 6.5%, the reservoir emits exactly 565,244 SynOps x 0.9 pJ = <b>0.509 uJ</b>.<br/>"
                           "<i>2) FP32 Interface Regime:</i> Adding 480k leak MACs (480k x 4.6 pJ = 2.208 uJ), 1.92M sensory input MACs (1.92M x 4.6 pJ = 8.832 uJ), and 1.44M motor readout MACs (1.44M x 4.6 pJ = 6.624 uJ) totals <b>18.17 uJ</b>. Here, comparable activation-sparse ANN with continuous I/O consumes 18.04 uJ (parity ~ 1.0x).<br/>"
                           "<i>3) Dedicated Event-Driven Neuromorphic Regime:</i> Leak updates utilize single-cycle bit-shift accumulator additions (480,000 x 0.9 pJ = 0.432 uJ). Sensory antenna inputs (4 channels to 1,200 neurons = 1.92M potential connection steps) encounter plume filaments across 263 active steps out of 400 (empirical presence duty cycle of 65.75%: 1,920,000 x 0.6575 = 1,262,400 active input spike events x 0.9 pJ = 1.136 uJ). Motor readout projections (1,200 neurons to 3 outputs = 1.44M potential connections) accumulate strictly upon reservoir spikes (6.5% firing rate: 1,440,000 x 0.065 = 93,600 events) plus 1,200 integrator updates (94,800 operations x 0.9 pJ = 0.085 uJ). Summing all event-driven components yields exactly <b>2.16 uJ</b> (0.509 + 0.432 + 1.136 + 0.085 = 2.162 uJ).", body_style))

    # III. Experimental Results
    story.append(Paragraph("III. EXPERIMENTAL RESULTS", h1_style))
    
    table_data = [
        [Paragraph("Hardware / Architecture Model", cell_header), 
         Paragraph("Active Operations", cell_header), 
         Paragraph("Arithmetic Energy", cell_header), 
         Paragraph("vs. Sparse ANN (55.20 uJ)", cell_header), 
         Paragraph("Core vs Core", cell_header)],
        [Paragraph("Dense ANN Baseline", cell_style), Paragraph("579.36M MACs", cell_style), Paragraph("2,665.06 uJ", cell_style), Paragraph("---", cell_style), Paragraph("Baseline (1.0x)", cell_style)],
        [Paragraph("Uniform Sparse ANN (98.5%)", cell_style), Paragraph("12.00M MACs", cell_style), Paragraph("55.20 uJ", cell_style), Paragraph("Baseline (1.0x)", cell_style), Paragraph("Core: 39.74 uJ", cell_style)],
        [Paragraph("Activation-Sparse ANN (Continuous I/O)", cell_style), Paragraph("561.6k MACs + Dense I/O", cell_style), Paragraph("18.04 uJ", cell_style), Paragraph("3.1x Savings", cell_style), Paragraph("Core: 2.58 uJ", cell_style)],
        [Paragraph("Activation-Sparse ANN (Duty-Cycle I/O)", cell_style), Paragraph("561.6k MACs + Active I/O", cell_style), Paragraph("8.88 uJ", cell_style), Paragraph("6.2x Savings", cell_style), Paragraph("Core: 2.58 uJ", cell_style)],
        [Paragraph("<b>Bio-SNN FP32 (Full Leak+I/O)</b>", cell_bold), Paragraph("SynOps + FP32 Ops", cell_style), Paragraph("<b>18.17 uJ</b>", cell_bold), Paragraph("<b>3.0x Savings</b>", cell_bold), Paragraph("Core: 0.509 uJ", cell_style)],
        [Paragraph("<b>Bio-SNN Event-Driven (Bit-Shift)</b>", cell_bold), Paragraph("SynOps + Bit-Shift", cell_style), Paragraph("<b>2.16 uJ</b>", cell_bold), Paragraph("<b>25.5x Savings (4.1x vs Act-IO)</b>", cell_bold), Paragraph("<b>Core: 0.509 uJ (5.1x)</b>", cell_bold)]
    ]

    t = Table(table_data, colWidths=[135, 95, 95, 95, 100])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#0F172A')),
        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#CBD5E1')),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor('#F8FAFC')]),
        ('TOPPADDING', (0,0), (-1,-1), 1.4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 1.4),
    ]))
    story.append(t)
    story.append(Spacer(1, 3))

    story.append(Paragraph("<b>Behavioral Metrics & Null Baselines:</b> Initialized at distance 155.0 to source, untrained random-weight Bio-SNN achieved final distance <b>113.2 +/- 8.4</b> units. This significantly outperforms a random walk null control (152.4 +/- 14.1, Welch-Satterthwaite t(39.1) = 11.9, p &lt; 10<sup>-12</sup>), a shuffled-input null control (149.6 +/- 13.2, t(40.8) = 11.5, p &lt; 10<sup>-12</sup>), a constant forward drive control (141.8 +/- 11.5, t(44.2) = 10.0, p &lt; 10<sup>-9</sup>), and Dense ANN (126.9 +/- 9.2, t(47.6) = 5.48, p = 1.6 x 10<sup>-6</sup> across n = 25 seeds). These statistical controls support directed sensorimotor gradient-following beyond random topological bias. At the recurrent core level, Bio-SNN (0.509 uJ) outperforms activation-sparse ANN (2.58 uJ) by <b>5.1x</b> (2.583 / 0.509 = 5.07 ~ 5.1x), directly isolating the arithmetic advantage of event-driven accumulation over multiply-accumulation (E<sub>MAC</sub> / E<sub>AC</sub> = 5.11). Memory transport (SRAM/DRAM) represents an external hardware-dependent constraint.", body_style))

    # IV. Conclusion
    story.append(Paragraph("IV. CONCLUSION", h1_style))
    story.append(Paragraph("Evaluating across both FP32 and dedicated event-driven neuromorphic regimes demonstrates that biologically motivated spiking architectures achieve an isolated 5.1x core arithmetic advantage over matched activation-sparse ANNs, yielding 4.1x end-to-end savings over activation-sparse ANN with duty-cycle I/O and 25.5x savings over uniform sparse ANN.", body_style))
    story.append(Spacer(1, 3))

    # References
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor('#CBD5E1'), spaceAfter=3))
    story.append(Paragraph("<b>References:</b> [1] S. Dorkenwald et al., Nature 634, 124-138 (2024). [2] P. Schlegel et al., Nature 634, 139-152 (2024). [3] M. Horowitz, IEEE ISSCC, pp. 10-14 (2014). [4] M. Davies et al., IEEE Micro 38, 82-99 (2018).", affil_style))

    doc.build(story)
    print(f"[SUCCESS] Paper 1 Polished PDF built: {filename}")

if __name__ == "__main__":
    build_paper1_pdf()
