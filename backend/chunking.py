# Document → chunks (Phase 3)

import re
from dataclasses import dataclass


@dataclass
class Chunk:
    index: int  # chunk number within document (0, 1, 2...)
    text: str  # the actual text that gets embedded and retrieved
    page: int  # which page this chunk came from (for citations)


CHUNK_SIZE = 1500  # larger chunks = fewer embedding API calls (free tier ~100/min)
CHUNK_OVERLAP = 150


def chunk_pages(pages: list, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[Chunk]:
    full_parts: list[tuple[int, str]] = []  # (page_number, text_on_page)
    for p in pages:
        full_parts.append((p.page, p.text))

    combined = ""  # build one string with page markers embedded
    page_map: list[tuple[int, int, int]] = []  # (char_start, char_end, page_num)
    cursor = 0
    for page_num, text in full_parts:
        marker = f"[Page {page_num}]\n"
        combined += marker + text + "\n\n"
        start = cursor + len(marker)
        end = cursor + len(marker) + len(text)
        page_map.append((start, end, page_num))
        cursor = len(combined)

    if not combined.strip():
        return []

    chunks: list[Chunk] = []
    start = 0
    index = 0

    while start < len(combined):
        end = min(start + chunk_size, len(combined))  # slice a window of CHUNK_SIZE chars

        if end < len(combined):
            boundary = _find_break(combined, start, end)  # try to break at paragraph/sentence
            if boundary > start:
                end = boundary

        text = combined[start:end].strip()
        if text:
            page = _page_for_position(start, page_map)  # figure out which page this chunk belongs to
            chunks.append(Chunk(index=index, text=text, page=page))
            index += 1

        if end >= len(combined):
            break
        start = max(end - overlap, start + 1)  # step forward with overlap

    return chunks


def _find_break(text: str, start: int, end: int) -> int:
    window = text[start:end]
    for sep in ["\n\n", "\n", ". ", "? ", "! "]:  # prefer breaking at natural boundaries
        pos = window.rfind(sep)
        if pos > len(window) * 0.4:  # only break if we're past 40% of chunk (avoid tiny chunks)
            return start + pos + len(sep)
    return end


def _page_for_position(pos: int, page_map: list[tuple[int, int, int]]) -> int:
    for start, end, page in page_map:
        if start <= pos <= end:
            return page
    return page_map[-1][2] if page_map else 1
