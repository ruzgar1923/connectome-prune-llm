import os
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
from reportlab.lib import colors

def build_turkish_pdf(filename="ConnectomePrune_Ozet_Rapor.pdf"):
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
        fontSize=16,
        leading=20,
        alignment=1,
        textColor=colors.HexColor('#0F172A')
    )

    sub_style = ParagraphStyle(
        'SubTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=11,
        leading=14,
        alignment=1,
        textColor=colors.HexColor('#2563EB')
    )

    author_style = ParagraphStyle(
        'AuthorStyle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=10,
        leading=13,
        alignment=1,
        textColor=colors.HexColor('#1E293B')
    )

    affil_style = ParagraphStyle(
        'AffilStyle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8.5,
        leading=11.5,
        alignment=1,
        textColor=colors.HexColor('#475569')
    )

    h1_style = ParagraphStyle(
        'Heading1_Custom',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=11,
        leading=14,
        textColor=colors.HexColor('#0F172A'),
        spaceBefore=8,
        spaceAfter=3
    )

    body_style = ParagraphStyle(
        'Body_Custom',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9,
        leading=12.5,
        textColor=colors.HexColor('#1E293B'),
        spaceAfter=5
    )

    bullet_style = ParagraphStyle(
        'Bullet_Custom',
        parent=body_style,
        leftIndent=12,
        spaceAfter=2
    )

    story = []

    # Baslik
    story.append(Paragraph("PROJE OZET RAPORU: CONNECTOMEPRUNE-LLM", title_style))
    story.append(Paragraph("Meyve Sinegi Beyin Konnektomu ile Buyuk Dil Modelleri Optimizasyonu", sub_style))
    story.append(Spacer(1, 6))

    # Yazar
    story.append(Paragraph("Ruzgar Vatansever", author_style))
    story.append(Paragraph("Sehit Ogretmen Necmeddin Kuyucu Anadolu Lisesi (11. Sinif) | Gebze, Kocaeli", affil_style))
    story.append(Paragraph("E-posta: rzvatansevertr@gmail.com | GitHub: github.com/ruzgar1923 | Hugging Face: RuzgarVatansever", affil_style))
    story.append(Paragraph("Resmi DOI: 10.5281/zenodo.24375969", affil_style))
    story.append(Spacer(1, 8))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor('#CBD5E1'), spaceAfter=8))

    # 1. Yonetici Ozeti
    story.append(Paragraph("1. YONETICI OZETI", h1_style))
    story.append(Paragraph("ConnectomePrune; biyolojik sinir sistemlerinin (ozellikle Nature 2024'te haritalanan <i>Drosophila melanogaster</i> meyve sinegi beyninin) minimum enerjiyle maksimum verimlilik saglama prensiplerini modern Buyuk Dil Modellerine (LLM) aktaran ozgun bir model seyreltme (pruning) algoritmasidir. Meta'nin en guncel <b>LLaMA-3.2-1B-Instruct</b> modeli uzerinde test edilmis; <b>263.6 Milyon parametre (%27.10 dogrusallik seyreltmesi)</b> sadece 29 saniyede silinmistir. Model mantik, fizik ve kodlama testlerinde sifir basari kaybi yasarken, sembolik cikarimda <b>8.7 kat hizlanma</b> elde etmistir.", body_style))

    # 2. Temel Biyolojik & Muhendislik Mekanizmalari
    story.append(Paragraph("2. BIYOLOJIK ESINLENME VE MUHENDISLIK YONTEMI", h1_style))
    story.append(Paragraph("<b>A. Merkezi Hub Koruma (Central Complex Resilience):</b> Beyindeki kritik yonlendirme merkezleri gibi, LLM icerisindeki dikkat (attention) katmanlari korumaya alinmis, yuksek seyreltme MLP katmanlarina aktarilmistir. Kelime gomuleri (embeddings) %100 yogun tutulmustur.", bullet_style))
    story.append(Paragraph("<b>B. Homeostatik Sinaptik Olcekleme:</b> Baglantilar budandikca hayatta kalan agirliklar katman sinyal gucunu korumak icin gamma = 1 / sqrt(surviving_ratio) carpani ile otomatik dengelenmistir.", bullet_style))
    story.append(Paragraph("<b>C. Heykeltiras (Sculptor) Pipeline:</b> Milyarlarca parametre bellek asimi yasanmadan, dogrudan safetensors dosyalari uzerinde parcali olarak islenmis, siradan bir dizustu bilgisayarda 29 saniyede tamamlanmistir.", bullet_style))

    # 3. Karsilastirmali Test Sonuclari Tablosu
    story.append(Paragraph("3. META LLaMA-3.2-1B DENEYSEL TEST SONUCLARI", h1_style))
    
    table_data = [
        ["Metrik / Test", "Orijinal Meta LLaMA-3.2-1B", "Connectome-Pruned (Budanmis)", "Kazanim / Fark"],
        ["Dogrusal Parametre Sayisi", "973,078,528", "709,405,947", "-263,672,581 Parametre"],
        ["Dogrusallik Seyreltmesi", "%0.00", "%27.10", "%27.10 Hafiza Rahatlamasi"],
        ["Budama Suresi", "---", "29.08 saniye", "Anlik Egitim-Sonrasi Islem"],
        ["Python Kod Sentezi (is_prime)", "Eksiksiz Kod", "Eksiksiz Kod (Tam Esit)", "Sifir Dogruluk Kaybi"],
        ["Fizik Muhakemesi (Rayleigh)", "Eksiksiz Aciklama", "Eksiksiz Aciklama", "Sifir Halusinasyon"],
        ["Aritmetik Cikarim Hizi", "1,698.4 ms/token", "194.3 ms/token", "8.7 Kat Daha Hizli Yanit"]
    ]

    t = Table(table_data, colWidths=[140, 125, 135, 140])
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

    # 4. Sahada Kime ve Neye Yarar? (Endustriyel & Stratejik Deger)
    story.append(Paragraph("4. STRATEJIK VE ENDUSTRIYEL KULLANIM ALANLARI", h1_style))
    story.append(Paragraph("<b>1. Kenar Bilisim & Savunma Sanayii (IHA / SIHA / Kara Araclari):</b> Sahada internetsiz calisan mini bilgisayarlarda (NVIDIA Jetson, Raspberry Pi vb.) buyuk modellerin bataryayi tuketmeden ve 8.7 kat daha dusuk gecikmeyle (194 ms) gercek zamanli karar almasini saglar.", bullet_style))
    story.append(Paragraph("<b>2. Yerel Siber Guvenlik ve Veri Gizliligi:</b> Kritik kurumlarin log ve tehdit analizlerini buluta gondermeden, yerel guvenlik duvarlarinda hafifletilmis modelle sifir veri sizintisi riskiyle calistirmasina olanak tanir.", bullet_style))
    story.append(Paragraph("<b>3. Milli Donanim ve Sunucu Tasarrufu:</b> Yuz binlerce kullaniciya hizmet veren milli dil modeli sunucularinda GPU bellek ve elektrik tuketimini %30'a varan oranda azaltir.", bullet_style))

    # 5. Gelecek Yol Haritasi
    story.append(Paragraph("5. T3 VAKFI VE TEKNOFEST DESTEGI ILE GELECEK HEDEFLER", h1_style))
    story.append(Paragraph("T3 Vakfi ve TEKNOFEST ailesinin rehberliginde; bu biyolojik seyreltme mimarisini ulkemizin milli dil modeli girisimlerine, yerli savunma donanimlarina uyarlamayi ve IEEE/TUBITAK duzeyinde uluslararasi basarilara donusturmeyi hedefliyorum.", body_style))

    doc.build(story)
    print(f"[SUCCESS] Turkish summary PDF built: {filename}")

if __name__ == "__main__":
    build_turkish_pdf()
