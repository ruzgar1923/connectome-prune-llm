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
        fontSize=13,
        leading=17,
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
    story.append(Paragraph("Connectome-Inspired Sparse Neural Architectures Reduce Energy Costs in a Simulated Olfactory Plume-Tracking Task", title_style))
    story.append(Spacer(1, 4))

    # Author
    story.append(Paragraph("Rüzgar Vatansever", author_style))
    story.append(Paragraph("Independent Researcher, Şehit Öğretmen Necmeddin Kuyucu Anadolu Lisesi, Gebze, Kocaeli, Turkey", affil_style))
    story.append(Paragraph("Email: rzvatansevertr@gmail.com | ORCID: 0009-0004-9477-8422", affil_style))
    story.append(Spacer(1, 5))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor('#CBD5E1'), spaceAfter=5))

    # Abstract
    story.append(Paragraph("<b>Abstract</b>—Autonomous robotic edge hardware operates under stringent sub-watt power budgets. While dense artificial neural networks (ANNs) execute continuous floating-point matrix multiplications, biological micro-fliers (such as <i>Drosophila</i>) navigate complex sensory environments using event-driven spikes and structural sparsity. In this work, we evaluate a bio-inspired Spiking Neural Network (Bio-SNN) incorporating biological connectomic motifs: 98.5% structural sparsity, log-normal weights, and pre-synaptic Dale's polarity enforcement across an average firing rate of 6.5% (93.5% temporal sparsity). Across 400 simulation steps in a turbulent plume arena (n = 25 independent seeds), we quantify energy consumption against dense and symmetrically dimensioned sparse (98.5%) ANNs under standardized 45 nm CMOS arithmetic costs (Horowitz, 2014). SNN synaptic transmissions account for 0.51 &mu;J; accounting for membrane leak updates and input/output projections yields a total neuromorphic budget of <b>3.10 &mu;J to 18.17 &mu;J</b>, providing a <b>3.0&times; to 17.8&times; energy reduction</b> over the sparse ANN control (55.20 &mu;J) and a <b>147&times; to 860&times; reduction</b> over the dense ANN baseline (2,665.06 &mu;J). These findings suggest that biological energy advantages arise from the structural coupling of static network sparsity with temporal event sparsity.", body_style))
    story.append(Spacer(1, 4))

    # I. Introduction
    story.append(Paragraph("I. INTRODUCTION", h1_style))
    story.append(Paragraph("Edge robotics requires navigation under strict energy constraints. While dense ANNs compute continuous multiply-accumulate (MAC) operations regardless of input saliency, biological circuits in the adult fruit fly (Dorkenwald et al., Nature 2024; Schlegel et al., Nature 2024) rely on sparse asynchronous event spikes. We isolate the energetic impact of biological structural invariants in a simulated 400-step olfactory plume tracking task.", body_style))

    # II. Methodology
    story.append(Paragraph("II. METHODOLOGY & ENERGY ACCOUNTING", h1_style))
    story.append(Paragraph("<b>A. Architecture & Dale's Polarity:</b> The Bio-SNN consists of N = 1,200 leaky integrate-and-fire (LIF) units with decay &beta; = 0.85. Each pre-synaptic neuron j is assigned polar identity s<sub>j</sub> &isin; {+1, -1} (80% excitatory, 20% inhibitory), ensuring all outgoing axonal projections from column j maintain identical polarity: W<sub>rec,ij</sub> = M<sub>ij</sub> &times; w<sub>ij</sub> &times; s<sub>j</sub> &times; &sigma;<sub>scale</sub>, where M<sub>ij</sub> &sim; Bernoulli(1 - &rho;), w<sub>ij</sub> &sim; LogNormal(-1.0, 0.6), and &sigma;<sub>scale</sub> = 0.4 / &radic;((1 - &rho;)N) with sparsity &rho; = 0.985.", body_style))
    story.append(Paragraph("<b>B. Comprehensive Energy Accounting:</b> Energy is evaluated across full 400-step episodes under 45 nm CMOS logic costs (Horowitz, 2014): E<sub>AC</sub> = 0.9 pJ and E<sub>MAC</sub> = 4.6 pJ. Dense ANN incurs 579.36M MACs (2,665.06 &mu;J). Symmetrically dimensioned sparse ANN (recurrent connections sparse at 8.64M MACs, dense input/readout at 3.36M MACs) totals 12.0M MACs (55.20 &mu;J). Bio-SNN synaptic transmissions consume 565,244 &times; 0.9 pJ = 0.51 &mu;J. We evaluate two implementation regimes: (1) Standard FP32 regime (480k leak updates, 1.92M input MACs, 1.44M readout MACs: 18.17 &mu;J); and (2) Dedicated neuromorphic hardware (bit-shift decay and spike-encoded input/readout: 3.10 &mu;J).", body_style))

    # III. Experimental Results
    story.append(Paragraph("III. EXPERIMENTAL RESULTS", h1_style))
    
    table_data = [
        ["Hardware / Architecture Model", "Active Operations", "Total Energy (400 steps)", "vs. Sparse ANN (55.2 &mu;J)", "vs. Dense ANN (2.67 mJ)"],
        ["Dense ANN Baseline", "579.36M MACs", "2,665.06 &mu;J (2.67 mJ)", "---", "Baseline (1.0&times;)"],
        ["Sparse ANN Control (%98.5)", "12.00M MACs", "55.20 &mu;J", "Baseline (1.0&times;)", "48.3&times; Savings"],
        ["<b>Bio-SNN FP32 (Full Leak+In/Out)</b>", "<b>SynOps + FP32 Ops</b>", "<b>18.17 &mu;J</b>", "<b>3.0&times; Savings</b>", "<b>147&times; Savings</b>"],
        ["<b>Bio-SNN Neuromorphic (Loihi Bit-Shift)</b>", "<b>SynOps + Bit-Shift</b>", "<b>3.10 &mu;J</b>", "<b>17.8&times; Savings</b>", "<b>860&times; Savings</b>"]
    ]

    t = Table(table_data, colWidths=[140, 95, 95, 95, 95])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#0F172A')),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (-1,-1), 7),
        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#CBD5E1')),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor('#F8FAFC')]),
        ('TOPPADDING', (0,0), (-1,-1), 2),
        ('BOTTOMPADDING', (0,0), (-1,-1), 2),
    ]))
    story.append(t)
    story.append(Spacer(1, 4))

    story.append(Paragraph("<b>Behavioral Metrics:</b> Over n = 25 independent random network initializations, both Bio-SNN and dense ANN achieved net forward displacement toward sensory gradients (mean final distance: Bio-SNN 113.2 &plusmn; 8.4, ANN 126.9 &plusmn; 9.2). Both agents exhibited comparable obstacle navigation, while complete source localization under turbulence suggests the utility of downstream reinforcement training.", body_style))

    # IV. Conclusion
    story.append(Paragraph("IV. CONCLUSION", h1_style))
    story.append(Paragraph("Evaluating across both general FP32 and dedicated neuromorphic regimes confirms that connectome-inspired spiking networks achieve solid 3.0&times; to 17.8&times; energy advantages over matched sparse ANNs and orders of magnitude over dense baselines.", body_style))
    story.append(Spacer(1, 4))

    # References
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor('#CBD5E1'), spaceAfter=4))
    story.append(Paragraph("<b>References:</b> [1] S. Dorkenwald et al., Nature 634, 124–138 (2024). [2] P. Schlegel et al., Nature 634, 139–152 (2024). [3] M. Horowitz, IEEE ISSCC, pp. 10–14 (2014). [4] M. Davies et al., IEEE Micro 38, 82–99 (2018).", affil_style))

    doc.build(story)
    print(f"[SUCCESS] Paper 1 Final Polish PDF built: {filename}")

if __name__ == "__main__":
    build_paper1_pdf()
