# PDF → raw text extraction (Phase 2)

import re
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

from pypdf import PdfReader  # lightweight PDF library — reads text layer only (not OCR for scanned PDFs)


@dataclass
class PageText:
    page: int  # 1-based page number
    text: str  # extracted text from that page


def extract_pdf_text(file_bytes: bytes) -> list[PageText]:
    reader = PdfReader(BytesIO(file_bytes))  # read PDF from uploaded bytes in memory
    pages: list[PageText] = []

    for i, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""  # empty string if page has no text layer (e.g. scanned image)
        text = _clean_text(text)  # normalize whitespace
        if text.strip():
            pages.append(PageText(page=i, text=text))

    return pages  # list of {page, text} — one entry per page with content


def pages_to_full_text(pages: list[PageText]) -> str:
    parts = [f"[Page {p.page}]\n{p.text}" for p in pages]  # mark page boundaries for later citation
    return "\n\n".join(parts)


def _clean_text(text: str) -> str:
    text = text.replace("\x00", "")  # remove null bytes that break some databases
    text = re.sub(r"[ \t]+", " ", text)  # collapse horizontal whitespace
    text = re.sub(r"\n{3,}", "\n\n", text)  # max two newlines in a row
    return text.strip()


def save_upload_copy(document_id: str, filename: str, file_bytes: bytes, uploads_dir: Path) -> Path:
    uploads_dir.mkdir(parents=True, exist_ok=True)
    safe_name = re.sub(r"[^\w.\-]", "_", filename)  # sanitize filename for filesystem
    dest = uploads_dir / f"{document_id}_{safe_name}"
    dest.write_bytes(file_bytes)  # keep original PDF on disk for reference
    return dest
