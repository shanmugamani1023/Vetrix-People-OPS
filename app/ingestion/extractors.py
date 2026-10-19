"""
Document text extractors for PDF, DOCX, HTML, Markdown.
Includes OCR fallback via Claude Vision for scanned PDFs.
"""

from __future__ import annotations

import base64
import io
import os
import re
from pathlib import Path

import structlog

log = structlog.get_logger()

# Threshold: if avg chars/page < this, treat as scanned
SCANNED_THRESHOLD = 50


# ── Helpers ──────────────────────────────────────────────────────────────────

def _table_to_text(table: list[list[str | None]]) -> str:
    """Convert a pdfplumber / docx table to pipe-delimited text."""
    lines = []
    for row in table:
        cells = [str(c).strip() if c else "" for c in row]
        lines.append(" | ".join(cells))
    return "\n".join(lines)


# ── PDF ───────────────────────────────────────────────────────────────────────

def _extract_pdf_pdfplumber(path: str) -> tuple[str, bool]:
    """
    Returns (text, is_scanned).
    is_scanned=True when average chars/page < SCANNED_THRESHOLD.
    """
    import pdfplumber

    pages_text: list[str] = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            parts: list[str] = []

            # Plain text
            text = page.extract_text() or ""
            if text.strip():
                parts.append(text)

            # Tables
            tables = page.extract_tables()
            for table in tables:
                parts.append(_table_to_text(table))

            pages_text.append("\n".join(parts))

    full_text = "\n\n".join(pages_text)
    avg_chars = len(full_text.strip()) / max(len(pages_text), 1)
    is_scanned = avg_chars < SCANNED_THRESHOLD
    return full_text, is_scanned


def _ocr_pdf_tesseract(path: str) -> str:
    """
    Convert each PDF page to image and run Tesseract OCR.
    Used as free local fallback when Claude Vision is not available.
    """
    from pdf2image import convert_from_path
    import pytesseract

    log.info("ocr_tesseract_start", file=os.path.basename(path))
    pages = convert_from_path(path, dpi=300)
    all_text: list[str] = []

    for i, page_img in enumerate(pages, 1):
        page_text = pytesseract.image_to_string(page_img, lang="eng")
        if page_text.strip():
            all_text.append(page_text.strip())
        log.info("ocr_tesseract_page_done", file=os.path.basename(path), page=i, chars=len(page_text))

    return "\n\n".join(all_text)


def _ocr_pdf_claude(path: str, llm_client) -> str:
    """
    Convert each PDF page to image and send to Claude Vision for OCR.
    Returns extracted text.
    """
    from pdf2image import convert_from_path

    log.info("ocr_start", file=os.path.basename(path))
    pages = convert_from_path(path, dpi=200)
    all_text: list[str] = []

    for i, page_img in enumerate(pages, 1):
        buf = io.BytesIO()
        page_img.save(buf, format="PNG")
        img_b64 = base64.standard_b64encode(buf.getvalue()).decode()

        response = llm_client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=2048,
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": img_b64,
                        },
                    },
                    {
                        "type": "text",
                        "text": (
                            "Extract all text from this HR document page. "
                            "Preserve headings, bullet points and table structure. "
                            "Output only the extracted text, nothing else."
                        ),
                    },
                ],
            }],
        )
        page_text = response.content[0].text
        all_text.append(f"[Page {i}]\n{page_text}")
        log.info("ocr_page_done", file=os.path.basename(path), page=i)

    return "\n\n".join(all_text)


def extract_pdf(path: str, llm_client=None) -> str:
    text, is_scanned = _extract_pdf_pdfplumber(path)
    if not is_scanned:
        return text

    # Scanned PDF — try tesseract first (free, local)
    try:
        text = _ocr_pdf_tesseract(path)
        if text.strip():
            return text
    except Exception as exc:
        log.warning("ocr_tesseract_failed", file=os.path.basename(path), error=str(exc))

    # Fallback to Claude Vision if available
    if llm_client is not None:
        try:
            return _ocr_pdf_claude(path, llm_client)
        except Exception as exc:
            log.warning("ocr_claude_failed", file=os.path.basename(path), error=str(exc))

    log.warning("ocr_all_failed", file=os.path.basename(path))
    return text


# ── DOCX ──────────────────────────────────────────────────────────────────────

def extract_docx(path: str) -> str:
    from docx import Document

    doc = Document(path)
    parts: list[str] = []

    for block in doc.element.body:
        tag = block.tag.split("}")[-1] if "}" in block.tag else block.tag

        if tag == "p":
            # Paragraph
            from docx.oxml.ns import qn
            para_text = "".join(
                node.text or "" for node in block.iter()
                if node.tag == qn("w:t")
            )
            if para_text.strip():
                parts.append(para_text.strip())

        elif tag == "tbl":
            # Table
            rows: list[list[str]] = []
            for tr in block.findall(".//{http://schemas.openxmlformats.org/wordprocessingml/2006/main}tr"):
                cells = []
                for tc in tr.findall("{http://schemas.openxmlformats.org/wordprocessingml/2006/main}tc"):
                    cell_text = "".join(
                        n.text or "" for n in tc.iter()
                        if n.tag == "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}t"
                    )
                    cells.append(cell_text.strip())
                if cells:
                    rows.append(cells)
            if rows:
                parts.append(_table_to_text(rows))

    return "\n\n".join(parts)


# ── HTML ──────────────────────────────────────────────────────────────────────

def extract_html(path: str) -> str:
    from bs4 import BeautifulSoup

    with open(path, encoding="utf-8", errors="replace") as f:
        soup = BeautifulSoup(f.read(), "lxml")

    # Remove script/style
    for tag in soup(["script", "style", "nav", "footer"]):
        tag.decompose()

    # Convert tables to pipe-delimited text
    for table in soup.find_all("table"):
        rows: list[list[str]] = []
        for tr in table.find_all("tr"):
            cells = [td.get_text(strip=True) for td in tr.find_all(["td", "th"])]
            if cells:
                rows.append(cells)
        if rows:
            table.replace_with(soup.new_string(_table_to_text(rows) + "\n"))

    text = soup.get_text(separator="\n")
    # Collapse excessive blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# ── Markdown ──────────────────────────────────────────────────────────────────

def extract_md(path: str) -> str:
    with open(path, encoding="utf-8", errors="replace") as f:
        return f.read()


# ── Dispatcher ────────────────────────────────────────────────────────────────

def extract_text(file_path: str, llm_client=None) -> str:
    """
    Auto-detect format from extension and extract text.
    llm_client is required for OCR of scanned PDFs.
    """
    path = Path(file_path)
    ext = path.suffix.lower()

    if ext == ".pdf":
        return extract_pdf(str(path), llm_client)
    elif ext in (".docx", ".doc"):
        return extract_docx(str(path))
    elif ext in (".html", ".htm"):
        return extract_html(str(path))
    elif ext in (".md", ".markdown", ".txt"):
        return extract_md(str(path))
    else:
        log.warning("unknown_extension", file=path.name, ext=ext)
        # Try reading as plain text
        with open(path, encoding="utf-8", errors="replace") as f:
            return f.read()
