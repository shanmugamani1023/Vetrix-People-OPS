"""
Extract text from a scanned/image PDF using pytesseract OCR.
Run: python show_ocr.py
"""

import sys
import pytesseract
from pdf2image import convert_from_path

PDF_PATH = "data/corpus/memo_hybrid_work_update.pdf"

print("=" * 60)
print("OCR — Extracting text from scanned image PDF")
print("=" * 60)
print(f"\nFile: {PDF_PATH}\n")

# Step 1: Convert PDF pages to images
print("STEP 1 — Converting PDF pages to images...")
images = convert_from_path(PDF_PATH, dpi=300)
print(f"  {len(images)} page(s) converted to image\n")

# Step 2: Run OCR on each page
print("STEP 2 — Running Tesseract OCR on each page...\n")
full_text = ""
for i, image in enumerate(images, 1):
    print(f"  Page {i}: ", end="", flush=True)
    text = pytesseract.image_to_string(image, lang="eng")
    print(f"{len(text)} characters extracted")
    full_text += text + "\n"

# Step 3: Show result
print(f"\n{'='*60}")
print("EXTRACTED TEXT:")
print(f"{'='*60}\n")
print(full_text)
print(f"\n{'='*60}")
print(f"Total characters extracted: {len(full_text)}")
print(f"Total words extracted     : {len(full_text.split())}")
