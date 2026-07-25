from __future__ import annotations

import logging
import re
import unicodedata
from collections import Counter, defaultdict
from typing import Iterable

from langchain_core.documents import Document

from documents import load_documents


LOGGER = logging.getLogger("nutrischool.preprocessing")

ZERO_WIDTH_RE = re.compile(r"[\u200B-\u200D\uFEFF]")
SOFT_HYPHEN_RE = re.compile(r"\u00AD")
MULTISPACE_RE = re.compile(r"[ \t]+")
EXCESSIVE_NEWLINES_RE = re.compile(r"\n{3,}")
BROKEN_ENGLISH_WORD_RE = re.compile(r"(?<=[A-Za-z])[-\u2010\u2011]\s*\n\s*(?=[A-Za-z])")
BROKEN_BODY_LINE_RE = re.compile(r"(?<![.!?:;\]])\n(?=[a-z])")
NOISE_SYMBOL_RE = re.compile(r"[\uFFFD\u25A1\u25A0\u25C6]+")
ONLY_PAGE_NUMBER_RE = re.compile(r"^\s*(?:page\s*)?\d+\s*$", re.IGNORECASE)

DOWNLOAD_FOOTER_RE = re.compile(
    r"Downloaded\s+from\s+https?://.*?"
    r"(?:on\s+\d{1,2}\s+[A-Za-z]+\s+\d{4}|by\s+guest\s+on\s+\d{1,2}\s+[A-Za-z]+\s+\d{4})",
    re.IGNORECASE | re.DOTALL,
)
INDD_NOISE_RE = re.compile(
    r"\b\S+\.indd\s+\d+\s+"
    r"\d{1,2}/\d{1,2}/\d{2,4}\s+"
    r"\d{1,2}:\d{2}\s*(?:AM|PM)?",
    re.IGNORECASE,
)
URL_LINE_RE = re.compile(r"^\s*(?:https?://|www\.)\S+\s*$", re.IGNORECASE)
CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F]")


def normalize_unicode(text: str) -> str:
    value = unicodedata.normalize("NFKC", str(text or ""))
    value = value.replace("\r\n", "\n").replace("\r", "\n")
    value = value.replace("\u00A0", " ")
    value = SOFT_HYPHEN_RE.sub("", value)
    value = ZERO_WIDTH_RE.sub("", value)
    return CONTROL_CHAR_RE.sub("", value)


def normalize_line_for_detection(line: str) -> str:
    value = normalize_unicode(line).strip().lower()
    value = re.sub(r"\d+", "#", value)
    value = MULTISPACE_RE.sub(" ", value)
    return value


def collect_repeated_edge_lines(
    documents: Iterable[Document],
    edge_line_count: int = 5,
    minimum_pages: int = 12,
    minimum_ratio: float = 0.12,
) -> dict[str, set[str]]:
    book_pages: dict[str, list[list[str]]] = defaultdict(list)

    for document in documents:
        book_id = str(document.metadata.get("book_id", "unknown"))
        lines = [
            line.strip()
            for line in normalize_unicode(document.page_content).splitlines()
            if line.strip()
        ]
        book_pages[book_id].append(lines)

    repeated_by_book: dict[str, set[str]] = {}

    for book_id, pages in book_pages.items():
        counter: Counter[str] = Counter()

        for lines in pages:
            edge_lines = lines[:edge_line_count] + lines[-edge_line_count:]
            page_seen = {
                normalize_line_for_detection(line)
                for line in edge_lines
                if normalize_line_for_detection(line)
            }
            counter.update(page_seen)

        required_count = max(minimum_pages, round(len(pages) * minimum_ratio))
        repeated_by_book[book_id] = {
            line
            for line, count in counter.items()
            if count >= required_count and len(line) >= 4
        }

    return repeated_by_book


def remove_repeated_edges(
    text: str,
    repeated_lines: set[str],
    edge_line_count: int = 5,
) -> str:
    lines = normalize_unicode(text).splitlines()
    nonempty_indexes = [index for index, line in enumerate(lines) if line.strip()]
    edge_indexes = set(
        nonempty_indexes[:edge_line_count] + nonempty_indexes[-edge_line_count:]
    )
    cleaned_lines: list[str] = []

    for index, line in enumerate(lines):
        normalized = normalize_line_for_detection(line)

        if index in edge_indexes:
            if normalized in repeated_lines:
                continue
            if ONLY_PAGE_NUMBER_RE.fullmatch(line):
                continue

        cleaned_lines.append(line)

    return "\n".join(cleaned_lines)


def normalize_pdf_artifacts(value: str) -> str:
    value = DOWNLOAD_FOOTER_RE.sub("\n", value)
    value = INDD_NOISE_RE.sub("\n", value)
    value = re.sub(r"(?<=\d{4})(?=\d{2,4}\s+Chapter\s+\d+)", "\n", value)
    value = re.sub(r"(?<=\d)(?=Section\s+[IVXLC]+:)", "\n", value)
    value = re.sub(r"(?<=\d)(?=Chapter\s+\d+)", "\n", value)
    return value


def clean_page_text(text: str, repeated_lines: set[str] | None = None) -> str:
    value = normalize_unicode(text)
    value = normalize_pdf_artifacts(value)

    if repeated_lines:
        value = remove_repeated_edges(value, repeated_lines)

    value = BROKEN_ENGLISH_WORD_RE.sub("", value)
    value = NOISE_SYMBOL_RE.sub(" ", value)
    value = MULTISPACE_RE.sub(" ", value)

    cleaned_lines: list[str] = []
    for raw_line in value.splitlines():
        line = raw_line.strip()
        if not line:
            cleaned_lines.append("")
            continue
        if URL_LINE_RE.fullmatch(line):
            continue
        cleaned_lines.append(line)

    value = "\n".join(cleaned_lines)
    value = BROKEN_BODY_LINE_RE.sub(" ", value)
    value = EXCESSIVE_NEWLINES_RE.sub("\n\n", value)
    return value.strip()


def preprocess_documents(
    documents: list[Document],
    minimum_clean_characters: int = 60,
) -> list[Document]:
    repeated_by_book = collect_repeated_edge_lines(documents)
    cleaned_documents: list[Document] = []
    skipped = 0

    for document in documents:
        metadata = dict(document.metadata or {})
        book_id = str(metadata.get("book_id", "unknown"))
        original_text = str(document.page_content or "")
        cleaned_text = clean_page_text(
            original_text,
            repeated_by_book.get(book_id, set()),
        )

        if len(cleaned_text) < minimum_clean_characters:
            skipped += 1
            continue

        metadata.update(
            {
                "preprocessed": True,
                "preprocessing_version": "school_medical_safe_v5",
                "original_character_count": len(original_text),
                "clean_character_count": len(cleaned_text),
            }
        )
        cleaned_documents.append(Document(page_content=cleaned_text, metadata=metadata))

    if not cleaned_documents:
        raise ValueError("No pages remained after preprocessing.")

    LOGGER.info(
        "Preprocessed %s of %s pages; skipped %s pages after cleaning.",
        len(cleaned_documents),
        len(documents),
        skipped,
    )
    return cleaned_documents


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    raw_documents = load_documents()
    cleaned_documents = preprocess_documents(raw_documents)
    original_characters = sum(len(document.page_content) for document in raw_documents)
    cleaned_characters = sum(len(document.page_content) for document in cleaned_documents)

    print("\nPREPROCESSING REPORT")
    print("=" * 72)
    print(f"Input pages: {len(raw_documents)}")
    print(f"Output pages: {len(cleaned_documents)}")
    print(f"Original characters: {original_characters:,}")
    print(f"Cleaned characters: {cleaned_characters:,}")
    print(f"Characters removed: {original_characters - cleaned_characters:,}")


if __name__ == "__main__":
    main()