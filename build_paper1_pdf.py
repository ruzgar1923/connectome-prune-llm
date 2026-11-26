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
        fontSize=12.5,
        leading=16,
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
        fontSize=6.8,
        leading=8.8,
        textColor=colors.HexColor('#1E293B')
    )

    cell_bold = ParagraphStyle(
        'CellBold',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=6.8,
        leading=8.8,
        textColor=colors.HexColor('#0F172A')
    )

    cell_header = ParagraphStyle(
        'CellHeader',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=6.8,
        leading=8.8,
        textColor=colors.white
    )

    story = []

    # Title
    story.append(Paragraph("Sparse Spiking Neural Architectures Reduce Arithmetic Energy in a Simulated Olfactory Plume-Tracking Task", title_style))
    story.append(Spacer(1, 4))

    # Author
    story.append(Paragraph("Ruzgar Vatansever", author_style))
    story.append(Paragraph("Independent Researcher, Sehit Ogretmen Necmeddin Kuyucu Anadolu Lisesi, Gebze, Kocaeli, Turkey", affil_style))
    story.append(Paragraph("Email: rzvatansevertr@gmail.com | ORCID: 0009-0004-9477-8422", affil_style))
    story.append(Paragraph("Simulation Code Repository: https://github.com/ruzgar1923/connectome-prune-llm", affil_style))
    story.append(Spacer(1, 4))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor('#CBD5E1'), spaceAfter=4))

    # Abstract
    story.append(Paragraph("<b>Abstract</b>—Autonomous robotic edge hardware operates under stringent sub-watt power budgets. While dense artificial neural networks (ANNs) execute continuous floating-point matrix multiplications, biological micro-fliers (such as <i>Drosophila</i>) navigate complex sensory environments using event-driven spikes and structural sparsity. In this work, we evaluate a Spiking Neural Network incorporating biologically motivated constraints (Bio-SNN): 98.5% structural sparsity, log-normal weights, and pre-synaptic Dale's polarity enforcement across an average firing rate of 6.5% (93.5% temporal sparsity). Across 400 simulation steps in a turbulent plume arena (n = 25 independent seeds), we quantify arithmetic energy consumption against dense and symmetrically dimensioned sparse (98.5%) ANNs under standardized 45 nm CMOS logic costs (Horowitz, 2014). SNN synaptic transmissions account for 0.51 uJ; fully accounting for membrane leak updates and input/output projections yields a total arithmetic budget of <b>3.10 uJ to 18.17 uJ</b>, providing a <b>3.0x to 17.8x energy reduction</b> over the sparse ANN control (55.20 uJ) and a <b>147x to 860x reduction</b> over the dense ANN baseline (2,665.06 uJ). We emphasize that these figures reflect arithmetic operation energy; memory transport (SRAM/DRAM) represents an external hardware-dependent constraint.", body_style))
    story.append(Spacer(1, 3))

    # I. Introduction
    story.append(Paragraph("I. INTRODUCTION", h1_style))
    story.append(Paragraph("Edge robotics requires navigation under strict energy constraints. While dense ANNs compute continuous multiply-accumulate (MAC) operations regardless of input saliency, biological circuits in the adult fruit fly (Dorkenwald et al., Nature 2024; Schlegel et al., Nature 2024) rely on sparse asynchronous event spikes. We isolate the arithmetic energetic impact of biological structural invariants in a simulated 400-step olfactory plume tracking task.", body_style))

    # II. Methodology
    story.append(Paragraph("II. METHODOLOGY & ENERGY ACCOUNTING", h1_style))
    story.append(Paragraph("<b>A. Architecture & Dale's Polarity:</b> The Bio-SNN consists of N = 1,200 leaky integrate-and-fire (LIF) units with decay factor beta = 0.85 and firing threshold V<sub>th</sub> = 1.0. Each pre-synaptic neuron j is assigned polar identity s<sub>j</sub> in {+1, -1} (80% excitatory, 20% inhibitory), ensuring all outgoing axonal projections from column j maintain identical polarity: W<sub>rec,ij</sub> = M<sub>ij</sub> x w<sub>ij</sub> x s<sub>j</sub> x sigma<sub>scale</sub>, where M<sub>ij</sub> ~ Bernoulli(1 - rho), w<sub>ij</sub> ~ LogNormal(-1.0, 0.6), and sigma<sub>scale</sub> = 0.4 / sqrt((1 - rho)N) with sparsity rho = 0.985.", body_style))
    story.append(Paragraph("<b>B. Exact Arithmetic Energy Component Breakdown:</b> Energy is evaluated across full 400-step episodes under 45 nm CMOS logic costs (Horowitz, 2014): E<sub>AC</sub> = 0.9 pJ (accumulation) and E<sub>MAC</sub> = 4.6 pJ. Dense ANN incurs 579.36M MACs (2,665.06 uJ). Symmetrically dimensioned sparse ANN (8.64M recurrent MACs + 3.36M dense I/O MACs = 12.0M MACs) totals 55.20 uJ. For Bio-SNN, synaptic transmission consumes exactly 565,244 SynOps x 0.9 pJ = 0.509 uJ. In the FP32 regime, adding 480k leak MACs (480,000 x 4.6 pJ = 2.21 uJ), 1.92M input MACs (1,920,000 x 4.6 pJ = 8.83 uJ), and 1.44M readout MACs (1,440,000 x 4.6 pJ = 6.62 uJ) yields 18.17 uJ. In the dedicated event-driven regime, leak updates utilize bit-shift accumulation (480,000 x 0.9 pJ = 0.432 uJ), input encoding uses event spikes with an empirical active sensory duty cycle of 44.4% (1.92M x 0.444 = 1,280,000 x 0.9 pJ = 1.152 uJ), and readout accumulates only on active spikes with a 6.5% firing rate along with output integration (1,120,000 x 0.9 pJ = 1.008 uJ), totaling exactly 3.10 uJ (0.509 + 0.432 + 1.152 + 1.008 = 3.101 uJ).", body_style))

    # III. Experimental Results
    story.append(Paragraph("III. EXPERIMENTAL RESULTS", h1_style))
    
    table_data = [
        [Paragraph("Hardware / Architecture Model", cell_header), 
         Paragraph("Active Operations", cell_header), 
         Paragraph("Arithmetic Energy (400 steps)", cell_header), 
         Paragraph("vs. Sparse ANN (55.2 uJ)", cell_header), 
         Paragraph("vs. Dense ANN (2.67 mJ)", cell_header)],
        [Paragraph("Dense ANN Baseline", cell_style), Paragraph("579.36M MACs", cell_style), Paragraph("2,665.06 uJ", cell_style), Paragraph("---", cell_style), Paragraph("Baseline (1.0x)", cell_style)],
        [Paragraph("Sparse ANN Control (98.5%)", cell_style), Paragraph("12.00M MACs", cell_style), Paragraph("55.20 uJ", cell_style), Paragraph("Baseline (1.0x)", cell_style), Paragraph("48.3x Savings", cell_style)],
        [Paragraph("<b>Bio-SNN FP32 (Full Leak+In/Out)</b>", cell_bold), Paragraph("SynOps + FP32 Ops", cell_style), Paragraph("<b>18.17 uJ</b>", cell_bold), Paragraph("<b>3.0x Savings</b>", cell_bold), Paragraph("<b>147x Savings</b>", cell_bold)],
        [Paragraph("<b>Bio-SNN Event-Driven (Bit-Shift)</b>", cell_bold), Paragraph("SynOps + Bit-Shift", cell_style), Paragraph("<b>3.10 uJ</b>", cell_bold), Paragraph("<b>17.8x Savings</b>", cell_bold), Paragraph("<b>860x Savings</b>", cell_bold)]
    ]

    t = Table(table_data, colWidths=[140, 95, 100, 95, 90])
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

    story.append(Paragraph("<b>Behavioral Metrics:</b> Starting at distance 155.0 to source, both untrained (random-weight) agents achieved forward displacement toward sensory gradients (mean final distance: Bio-SNN 113.2 +/- 8.4 vs. Dense ANN 126.9 +/- 9.2, where lower distance denotes superior source proximity; Welch's two-tailed t-test t(48) = 5.48, p &lt; 0.01 across n = 25 independent random network seeds). We acknowledge that continuous ANNs with ReLU activations can also exploit dynamic activation sparsity in specialized sparse accelerators, which may narrow the margin over standard dense GEMMs.", body_style))

    # IV. Conclusion
    story.append(Paragraph("IV. CONCLUSION", h1_style))
    story.append(Paragraph("Evaluating across both general FP32 and dedicated neuromorphic arithmetic regimes suggests that biologically motivated spiking architectures achieve solid 3.0x to 17.8x arithmetic energy advantages over matched sparse ANNs and orders of magnitude over dense baselines.", body_style))
    story.append(Spacer(1, 4))

    # References
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor('#CBD5E1'), spaceAfter=4))
    story.append(Paragraph("<b>References:</b> [1] S. Dorkenwald et al., Nature 634, 124-138 (2024). [2] P. Schlegel et al., Nature 634, 139-152 (2024). [3] M. Horowitz, IEEE ISSCC, pp. 10-14 (2014). [4] M. Davies et al., IEEE Micro 38, 82-99 (2018).", affil_style))

    doc.build(story)
    print(f"[SUCCESS] Paper 1 Polished PDF built: {filename}")

if __name__ == "__main__":
    build_paper1_pdf()
