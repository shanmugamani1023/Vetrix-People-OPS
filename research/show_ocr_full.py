"""
Full pipeline for scanned PDF:
  Step 1 - Convert PDF to image
  Step 2 - OCR with pytesseract
  Step 3 - Chunk the extracted text
  Step 4 - Show everything

Run: python show_ocr_full.py
"""

import re
import sys
import warnings
warnings.filterwarnings("ignore")

import pytesseract
import tiktoken
from pdf2image import convert_from_path

PDF_PATH = "data/corpus/memo_hybrid_work_update.pdf"
TOKENIZER = tiktoken.get_encoding("cl100k_base")
MAX_CHUNK_TOKENS = 400


def count_tokens(text):
    return len(TOKENIZER.encode(text))


# ── Chunking (same logic as pipeline.py) ─────────────────────────────────────

HEADING_RE = re.compile(
    r"^(#{1,6}\s+\S.*|"
    r"\d+(?:\.\d+)*\s+[A-Z]\S.*|"
    r"[A-Z][A-Z &/-]{3,}:?\s*)$",
    re.MULTILINE,
)


def split_by_paragraphs(text, locator):
    paras = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
    chunks, buf, buf_tokens = [], [], 0
    for para in paras:
        pt = count_tokens(para)
        if buf and buf_tokens + pt > MAX_CHUNK_TOKENS:
            body = "\n\n".join(buf)
            chunks.append({"locator": locator, "text": body, "tokens": count_tokens(body)})
            buf, buf_tokens = [], 0
        buf.append(para)
        buf_tokens += pt
    if buf:
        body = "\n\n".join(buf)
        chunks.append({"locator": locator, "text": body, "tokens": count_tokens(body)})
    return chunks or [{"locator": locator, "text": text[:2000], "tokens": count_tokens(text[:2000])}]


def chunk_text(text):
    matches = list(HEADING_RE.finditer(text))
    chunks = []
    if not matches:
        return split_by_paragraphs(text, "p. 1")
    pre = text[: matches[0].start()].strip()
    if pre:
        chunks.extend(split_by_paragraphs(pre, "p. 1"))
    for i, m in enumerate(matches):
        heading = m.group().strip()[:80]
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        section = text[start:end].strip()
        if not section:
            continue
        if count_tokens(section) <= MAX_CHUNK_TOKENS:
            chunks.append({"locator": heading, "text": section, "tokens": count_tokens(section)})
        else:
            chunks.extend(split_by_paragraphs(section, heading))
    return chunks


# ── MAIN ──────────────────────────────────────────────────────────────────────

print("\n" + "=" * 65)
print("  SCANNED PDF → OCR → CHUNKS — Full Pipeline Demo")
print("=" * 65)
print(f"\n  File: {PDF_PATH}")

# ── STEP 1: Convert PDF to image ──────────────────────────────────────────────

print("\n" + "─" * 65)
print("  STEP 1 — Convert PDF pages to images (300 DPI)")
print("─" * 65)

images = convert_from_path(PDF_PATH, dpi=300)
print(f"\n  Pages found : {len(images)}")
print(f"  Image size  : {images[0].size[0]} x {images[0].size[1]} pixels")
print(f"  Mode        : {images[0].mode}")
print(f"\n  (Each page is now a high-resolution photo that OCR can read)")

# ── STEP 2: OCR each page ─────────────────────────────────────────────────────

print("\n" + "─" * 65)
print("  STEP 2 — Tesseract OCR: image → text")
print("─" * 65)

full_text = ""
for i, image in enumerate(images, 1):
    page_text = pytesseract.image_to_string(image, lang="eng")
    full_text += page_text + "\n"
    print(f"\n  Page {i} result:")
    print(f"  Characters : {len(page_text)}")
    print(f"  Words      : {len(page_text.split())}")
    print(f"  Tokens     : {count_tokens(page_text)}")

# ── STEP 3: Show raw extracted text ──────────────────────────────────────────

print("\n" + "─" * 65)
print("  STEP 3 — Raw extracted text (exactly what OCR returned)")
print("─" * 65)
print()
for line in full_text.strip().splitlines():
    print(f"  │  {line}")

# ── STEP 4: Chunk the text ────────────────────────────────────────────────────

print("\n" + "─" * 65)
print("  STEP 4 — Chunking the extracted text")
print("─" * 65)

chunks = chunk_text(full_text.strip())
print(f"\n  Total chunks created: {len(chunks)}\n")

for i, chunk in enumerate(chunks, 1):
    print(f"  ┌─ CHUNK {i} of {len(chunks)} {'─' * 45}")
    print(f"  │  Locator : {chunk['locator']}")
    print(f"  │  Tokens  : {chunk['tokens']} / {MAX_CHUNK_TOKENS} max")
    print(f"  │  Chars   : {len(chunk['text'])}")
    print(f"  ├{'─' * 55}")
    for line in chunk["text"].splitlines():
        print(f"  │  {line}")
    print(f"  └{'─' * 55}\n")

# ── STEP 5: Summary ───────────────────────────────────────────────────────────

print("─" * 65)
print("  SUMMARY")
print("─" * 65)
print(f"""
  PDF type       : Scanned image (0 chars with pdfplumber)
  OCR engine     : Tesseract 4.1.1
  Pages          : {len(images)}
  Characters     : {len(full_text.strip())}
  Words          : {len(full_text.split())}
  Tokens         : {count_tokens(full_text.strip())}
  Chunks created : {len(chunks)}

  These chunks can now be:
    → Stored in doc_chunks table in SQLite
    → Converted to vectors (embeddings)
    → Added to FAISS index
    → Searched when employees ask about hybrid work policy
""")
print("=" * 65)
