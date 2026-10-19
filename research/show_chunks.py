"""
Script to extract and display chunks from a single PDF.
Run: python show_chunks.py
"""

import re
import sys
import tiktoken

TOKENIZER = tiktoken.get_encoding("cl100k_base")
MAX_CHUNK_TOKENS = 400


def count_tokens(text: str) -> int:
    return len(TOKENIZER.encode(text))


# ── Step 1: Read PDF ──────────────────────────────────────────────────────────

def extract_pdf_text(path: str) -> str:
    """Use the project's real extractor (same as ingestion pipeline)."""
    import sys
    sys.path.insert(0, ".")
    from app.ingestion.extractors import extract_text
    return extract_text(path, llm_client=None)


# ── Step 2: Split by headings ─────────────────────────────────────────────────

HEADING_RE = re.compile(
    r"^(#{1,6}\s+\S.*|"           # ## Markdown heading
    r"\d+(?:\.\d+)*\s+[A-Z]\S.*|"  # 1.2 Numbered heading
    r"[A-Z][A-Z &/-]{3,}:?\s*)$",  # ALL-CAPS heading
    re.MULTILINE,
)


def split_by_paragraphs(text: str, locator: str) -> list[dict]:
    """Split a long section into paragraph-grouped chunks."""
    paras = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
    chunks = []
    buf = []
    buf_tokens = 0

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


def chunk_text(text: str) -> list[dict]:
    """Heading-aware chunking."""
    matches = list(HEADING_RE.finditer(text))
    chunks = []

    if not matches:
        return split_by_paragraphs(text, "p. 1")

    # Text before the first heading
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


# ── Step 3: Display ───────────────────────────────────────────────────────────

def display_chunks(pdf_path: str):
    print(f"\n{'='*60}")
    print(f"PDF: {pdf_path}")
    print(f"{'='*60}\n")

    print("STEP 1 — Extracting text from PDF...")
    raw_text = extract_pdf_text(pdf_path)
    total_tokens = count_tokens(raw_text)
    print(f"  Raw text: {len(raw_text)} characters | {total_tokens} tokens\n")

    print("STEP 2 — Splitting into chunks...\n")
    chunks = chunk_text(raw_text)

    print(f"STEP 3 — Result: {len(chunks)} chunks\n")
    print(f"{'='*60}\n")

    for i, chunk in enumerate(chunks, 1):
        print(f"┌─ CHUNK {i} of {len(chunks)} {'─'*40}")
        print(f"│  Locator : {chunk['locator']}")
        print(f"│  Tokens  : {chunk['tokens']} / {MAX_CHUNK_TOKENS} max")
        print(f"│  Chars   : {len(chunk['text'])}")
        print(f"├{'─'*50}")
        # Show full text
        for line in chunk["text"].splitlines():
            print(f"│  {line}")
        print(f"└{'─'*50}\n")

    print(f"\nSUMMARY")
    print(f"  Total chunks : {len(chunks)}")
    print(f"  Total tokens : {sum(c['tokens'] for c in chunks)}")
    print(f"  Avg tokens   : {sum(c['tokens'] for c in chunks) // len(chunks)}")
    print(f"  Max tokens   : {max(c['tokens'] for c in chunks)}")
    print(f"  Min tokens   : {min(c['tokens'] for c in chunks)}")


# ── Run ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    pdf = sys.argv[1] if len(sys.argv) > 1 else "data/corpus/global_leave_policy_v2.pdf"
    display_chunks(pdf)
