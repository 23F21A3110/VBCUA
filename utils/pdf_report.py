from fpdf import FPDF

def create_pdf(text, summary, score, feedback):
    pdf = FPDF()
    pdf.add_page()

    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "Voice-Based Concept Understanding Report", ln=True)

    pdf.ln(10)

    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 10, "Transcribed Text:", ln=True)

    pdf.set_font("Arial", "", 11)
    pdf.multi_cell(0, 8, text)

    pdf.ln(5)

    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 10, "AI Summary:", ln=True)

    pdf.set_font("Arial", "", 11)
    pdf.multi_cell(0, 8, summary)

    pdf.ln(5)

    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 10, f"Concept Score: {score:.2f}%", ln=True)

    pdf.ln(5)

    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 10, "AI Feedback:", ln=True)

    pdf.set_font("Arial", "", 11)
    pdf.multi_cell(0, 8, feedback)

    pdf.output("report.pdf")