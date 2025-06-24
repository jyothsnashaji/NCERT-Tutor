import pytesseract
from pdf2image import convert_from_path
from PIL import Image
import csv
import os

# Set path to tesseract if needed
# pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract'  # or r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def extract_text_from_scanned_pdf(pdf_path):
    images = convert_from_path(pdf_path, dpi=300)
    text = ""
    for img in images:
        text += pytesseract.image_to_string(img, lang='mal') + "\n\n"
    return text

def split_into_paragraphs(text):
    import re
    paragraphs = re.split(r'\n\s*\n', text)
    return [p.strip() for p in paragraphs if p.strip()]

def save_paragraphs_to_csv(paragraphs, csv_path):
    with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['paragraph'])
        for para in paragraphs:
            writer.writerow([para])

# === MAIN ===
pdf_path = "NCERT-Tutor/mal.py"
csv_path = "NCERT-Tutor/output_malayalam_paragraphs.csv"

text = extract_text_from_scanned_pdf(pdf_path)
paragraphs = split_into_paragraphs(text)
save_paragraphs_to_csv(paragraphs, csv_path)

print(f"✅ Extracted {len(paragraphs)} Malayalam paragraphs into {csv_path}")
