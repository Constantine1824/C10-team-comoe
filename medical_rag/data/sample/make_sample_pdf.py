"""One-off script: renders sample_guideline.txt into a real PDF so the demo
exercises the actual PDF loader/heading-heuristic path, not a text stand-in."""
from pathlib import Path
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

SRC = Path(__file__).parent / "sample_guideline.txt"
OUT = Path(__file__).parent / "sample_guideline.pdf"

HEADINGS = {
    "Management of Type 2 Diabetes in Adults",
    "Diagnosis",
    "First-Line Pharmacological Therapy",
    "Renal Considerations",
    "Monitoring",
    "When to Escalate Therapy",
}


def wrap(text, width=95):
    words, line, lines = text.split(), "", []
    for w in words:
        if len(line) + len(w) + 1 > width:
            lines.append(line)
            line = w
        else:
            line = f"{line} {w}".strip()
    if line:
        lines.append(line)
    return lines


c = canvas.Canvas(str(OUT), pagesize=letter)
width, height = letter
y = height - 50

for raw_line in SRC.read_text().split("\n"):
    if not raw_line.strip():
        y -= 10
        continue
    is_heading = raw_line.strip() in HEADINGS
    c.setFont("Helvetica-Bold" if is_heading else "Helvetica", 13 if is_heading else 10)
    for line in wrap(raw_line, width=90 if not is_heading else 60):
        if y < 50:
            c.showPage()
            y = height - 50
        c.drawString(50, y, line)
        y -= 18 if is_heading else 14
    y -= 6

c.save()
print(f"wrote {OUT}")
