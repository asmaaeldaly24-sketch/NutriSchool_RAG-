from __future__ import annotations

import hashlib
import logging
import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Iterable

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from book_catalog import (
    AAP_CHAPTER_TITLES,
    PNP_CANONICAL_SECTIONS,
    canonical_aap_chapter,
    canonical_pnp_section,
    school_scope_allows,
)
from documents import load_documents
from preprocessing import preprocess_documents


LOGGER = logging.getLogger("nutrischool.chunking")

CHUNK_SIZE = max(700, int(os.getenv("RAG_CHUNK_SIZE", "1200")))
CHUNK_OVERLAP = max(80, int(os.getenv("RAG_CHUNK_OVERLAP", "180")))
SCHOOL_SCOPE_ONLY = os.getenv("RAG_SCHOOL_SCOPE_ONLY", "true").strip().lower() not in {
    "0",
    "false",
    "no",
    "off",
}

AAP_BODY_START_PAGE = max(1, int(os.getenv("RAG_AAP_BODY_START_PAGE", "13")))
PNP_BODY_START_PAGE = max(1, int(os.getenv("RAG_PNP_BODY_START_PAGE", "17")))

AAP_CHAPTER_LINE_RE = re.compile(
    r"^\s*(?:(?:\d{1,4})\s+)?Chapter\s+(?P<number>\d{1,3})(?:\s+.*)?$",
    re.IGNORECASE,
)
AAP_SECTION_RE = re.compile(
    r"^\s*(?:Section\s+)?(?P<number>[IVXLC]+)\s*[:.-]\s*(?P<title>[^\n]{3,120})$",
    re.IGNORECASE,
)
PNP_HEADING_RE = re.compile(
    r"^\s*(?P<number>\d{1,2}(?:\.\d{1,2}){1,2})\s+"
    r"(?P<title>[A-Z][A-Za-z0-9 ,/&()'’\-–—:]{1,180})\s*$"
)
REFERENCE_ENTRY_RE = re.compile(
    r"^\s*\d{1,3}[.)]?\s+[A-Z][A-Za-z'’\-]+(?:\s+[A-Z][A-Za-z'’\-]+)?(?:,|\s+[A-Z]{1,4},)",
)
YEAR_RE = re.compile(r"\b(?:19|20)\d{2}\b")


@dataclass(frozen=True)
class HierarchyState:
    part_title: str = ""
    section_title: str = ""
    chapter_title: str = ""
    chapter_number: str = ""


def compact(value: object) -> str:
    return " ".join(str(value or "").split()).strip()


def page_lines(text: str, maximum_lines: int = 70) -> list[str]:
    return [
        compact(line)
        for line in str(text or "").splitlines()[:maximum_lines]
        if compact(line)
    ]


def identify_book_layout(document: Document) -> str:
    book_id = str(document.metadata.get("book_id", "")).lower()
    if book_id == "aap8":
        return "aap_chapter"
    if book_id == "pnp3":
        return "pnp_numbered"
    return "unknown"


def classify_page(document: Document) -> str:
    book_id = str(document.metadata.get("book_id", ""))
    page_number = int(document.metadata.get("page_number", 0) or 0)
    text = compact(document.page_content).lower()
    first_lines = " ".join(page_lines(document.page_content, 12)).lower()

    if book_id == "aap8" and page_number < AAP_BODY_START_PAGE:
        return "front_matter"
    if book_id == "pnp3" and page_number < PNP_BODY_START_PAGE:
        return "front_matter"

    index_markers = (
        "subject index",
        "author index",
        "index of terms",
    )
    if any(marker in first_lines for marker in index_markers):
        return "index"

    reference_lines = sum(
        bool(REFERENCE_ENTRY_RE.match(line)) and bool(YEAR_RE.search(line))
        for line in page_lines(document.page_content, 80)
    )
    if first_lines.startswith("references") or reference_lines >= 5:
        return "references"

    if len(text) < 60:
        return "low_text"

    return "body"


def detect_aap_hierarchy(document: Document) -> HierarchyState:
    lines = page_lines(document.page_content, 35)
    section_title = ""

    for line in lines[:12]:
        match = AAP_SECTION_RE.fullmatch(line)
        if match and "section" in line.lower():
            title = re.sub(r"\s+\d+$", "", compact(match.group("title"))).strip()
            section_title = f"Section {match.group('number').upper()}: {title}"
            break

    for line in lines[:25]:
        match = AAP_CHAPTER_LINE_RE.fullmatch(line)
        if not match:
            continue

        number = int(match.group("number"))
        if number not in AAP_CHAPTER_TITLES:
            continue

        return HierarchyState(
            section_title=section_title,
            chapter_title=canonical_aap_chapter(number),
            chapter_number=str(number),
        )

    return HierarchyState(section_title=section_title)


def valid_pnp_heading(number: str, title: str) -> bool:
    title = compact(title)
    if not title or len(title) > 180:
        return False
    if title.count(",") > 3:
        return False
    lowered = title.lower()
    blocked = (
        "et al",
        "doi",
        "journal",
        "university",
        "published online",
        "references",
        "table ",
        "figure ",
        "fig ",
    )
    if any(marker in lowered for marker in blocked):
        return False
    if YEAR_RE.search(title):
        return False
    return number.count(".") in {1, 2}


def detect_pnp_hierarchy(document: Document) -> HierarchyState:
    candidates: list[tuple[str, str]] = []

    for line in page_lines(document.page_content, 55):
        match = PNP_HEADING_RE.fullmatch(line)
        if not match:
            continue
        number = match.group("number")
        title = compact(match.group("title"))
        if valid_pnp_heading(number, title):
            candidates.append((number, title))

    if not candidates:
        return HierarchyState()

    # The most specific heading near the top of the page is the best chapter label.
    number, extracted_title = max(candidates[:4], key=lambda item: item[0].count("."))
    chapter_title = canonical_pnp_section(number, extracted_title)

    parts = number.split(".")
    part_number = parts[0]
    section_number = ".".join(parts[:2])
    part_title = canonical_pnp_section(part_number, "") if part_number in PNP_CANONICAL_SECTIONS else part_number
    section_title = canonical_pnp_section(
        section_number,
        PNP_CANONICAL_SECTIONS.get(section_number, ""),
    )

    return HierarchyState(
        part_title=part_title,
        section_title=section_title,
        chapter_title=chapter_title,
        chapter_number=number,
    )


def hierarchy_path(metadata: dict[str, object]) -> str:
    values = [
        compact(metadata.get("part_title")),
        compact(metadata.get("section_title")),
        compact(metadata.get("chapter_title")),
    ]
    output: list[str] = []
    for value in values:
        if value and value not in output and value != "Front Matter":
            output.append(value)
    return " > ".join(output)


def attach_hierarchy(documents: Iterable[Document]) -> list[Document]:
    pages_by_book: dict[str, list[Document]] = defaultdict(list)
    for document in documents:
        pages_by_book[str(document.metadata.get("book_id", "unknown"))].append(document)

    output: list[Document] = []

    for book_id, pages in pages_by_book.items():
        pages.sort(key=lambda item: int(item.metadata.get("page_number", 0) or 0))
        state = HierarchyState()

        for page in pages:
            content_type = classify_page(page)
            layout = identify_book_layout(page)

            if content_type != "body":
                metadata = dict(page.metadata or {})
                metadata.update(
                    {
                        "book_layout": layout,
                        "content_type": content_type,
                        "indexable": False,
                        "hierarchy_version": "canonical_school_v10",
                    }
                )
                output.append(Document(page_content=page.page_content, metadata=metadata))
                continue

            detected = (
                detect_aap_hierarchy(page)
                if layout == "aap_chapter"
                else detect_pnp_hierarchy(page)
                if layout == "pnp_numbered"
                else HierarchyState()
            )

            state = HierarchyState(
                part_title=detected.part_title or state.part_title,
                section_title=detected.section_title or state.section_title,
                chapter_title=detected.chapter_title or state.chapter_title,
                chapter_number=detected.chapter_number or state.chapter_number,
            )

            metadata = dict(page.metadata or {})
            metadata.update(
                {
                    "book_layout": layout,
                    "part_title": state.part_title,
                    "section_title": state.section_title,
                    "chapter_title": state.chapter_title,
                    "chapter_number": state.chapter_number,
                    "content_type": "body",
                    "hierarchy_version": "canonical_school_v10",
                }
            )
            metadata["hierarchy_path"] = hierarchy_path(metadata)
            has_hierarchy = bool(metadata["hierarchy_path"] and state.chapter_number)
            in_scope = school_scope_allows(
                book_id=book_id,
                chapter_number=state.chapter_number,
                hierarchy=str(metadata["hierarchy_path"]),
            )
            metadata["school_scope"] = bool(in_scope)
            metadata["indexable"] = bool(has_hierarchy and (in_scope or not SCHOOL_SCOPE_ONLY))
            output.append(Document(page_content=page.page_content, metadata=metadata))

    return output


def is_reference_like_text(text: str) -> bool:
    lines = page_lines(text, 80)
    if not lines:
        return True

    citation_lines = sum(
        bool(REFERENCE_ENTRY_RE.match(line)) or " et al" in line.lower()
        for line in lines
    )
    year_lines = sum(bool(YEAR_RE.search(line)) for line in lines)
    return citation_lines >= 4 and year_lines >= 4 and citation_lines / len(lines) >= 0.20


def stable_chunk_id(document: Document, chunk_index: int) -> str:
    metadata = document.metadata
    payload = "|".join(
        [
            str(metadata.get("book_id", "unknown")),
            str(metadata.get("page_number", "0")),
            str(metadata.get("hierarchy_path", "")),
            str(chunk_index),
            compact(document.page_content)[:700],
        ]
    )
    digest = hashlib.sha256(payload.encode("utf-8", errors="ignore")).hexdigest()[:18]
    return (
        f"{metadata.get('book_id', 'book')}"
        f"_p{metadata.get('page_number', 0)}"
        f"_c{chunk_index}_{digest}"
    )


def create_chunks(
    documents: list[Document],
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> list[Document]:
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size.")

    hierarchical_documents = attach_hierarchy(documents)
    indexable_pages = [
        page
        for page in hierarchical_documents
        if bool(page.metadata.get("indexable"))
    ]

    if not indexable_pages:
        raise RuntimeError("No school-scope body pages were eligible for indexing.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", "; ", ": ", ", ", " ", ""],
        length_function=len,
        is_separator_regex=False,
    )

    chunks: list[Document] = []
    seen_keys: set[str] = set()

    for page in indexable_pages:
        page_chunks = splitter.split_documents([page])

        for chunk_index, chunk in enumerate(page_chunks, start=1):
            normalized_text = compact(chunk.page_content)
            if len(normalized_text) < 120:
                continue
            if is_reference_like_text(chunk.page_content):
                continue

            duplicate_payload = (
                f"{chunk.metadata.get('book_id', 'unknown')}|"
                f"{normalized_text.lower()}"
            )
            duplicate_key = hashlib.sha256(
                duplicate_payload.encode("utf-8", errors="ignore")
            ).hexdigest()
            if duplicate_key in seen_keys:
                continue
            seen_keys.add(duplicate_key)

            metadata = dict(chunk.metadata or {})
            metadata.update(
                {
                    "chunk_index_on_page": chunk_index,
                    "chunk_size_characters": len(chunk.page_content),
                    "chunking_version": "canonical_page_recursive_v10",
                    "content_type": "body",
                    "indexable": True,
                }
            )
            enriched = Document(page_content=chunk.page_content.strip(), metadata=metadata)
            enriched.metadata["chunk_id"] = stable_chunk_id(enriched, chunk_index)
            chunks.append(enriched)

    if not chunks:
        raise ValueError("No chunks were produced.")

    LOGGER.info(
        "Created %s school-scope chunks from %s cleaned pages.",
        len(chunks),
        len(documents),
    )
    return chunks


def audit_chunks(chunks: list[Document]) -> dict[str, object]:
    suspicious: list[str] = []
    book_counts: Counter[str] = Counter()
    chapter_counts: Counter[str] = Counter()

    for chunk in chunks:
        metadata = chunk.metadata
        book_counts[str(metadata.get("book_title", "Unknown"))] += 1
        chapter = compact(metadata.get("chapter_title"))
        chapter_counts[chapter or "Missing"] += 1

        if not chapter or chapter == "Front Matter":
            suspicious.append(str(metadata.get("chunk_id")))
        if bool(REFERENCE_ENTRY_RE.match(chapter)) or " et al" in chapter.lower():
            suspicious.append(str(metadata.get("chunk_id")))
        if metadata.get("content_type") != "body":
            suspicious.append(str(metadata.get("chunk_id")))

    report = {
        "chunk_count": len(chunks),
        "book_counts": dict(book_counts),
        "unique_chapters": len(chapter_counts),
        "suspicious_chunk_count": len(set(suspicious)),
        "top_chapters": chapter_counts.most_common(20),
    }

    if set(book_counts) != {"Pediatric-Nutrition", "PNP 3rd Edition Book"}:
        raise RuntimeError(f"Index must contain both books. Found: {dict(book_counts)}")
    if suspicious:
        raise RuntimeError(
            f"Chunk audit failed: {len(set(suspicious))} suspicious chunks were detected."
        )

    return report


def print_chunking_report(documents: list[Document], chunks: list[Document]) -> None:
    report = audit_chunks(chunks)
    print("\nCHUNKING AND METADATA AUDIT")
    print("=" * 72)
    print(f"Input cleaned pages: {len(documents)}")
    print(f"Created chunks: {report['chunk_count']}")
    print(f"Chunk size: {CHUNK_SIZE} characters")
    print(f"Chunk overlap: {CHUNK_OVERLAP} characters")
    print(f"School scope only: {SCHOOL_SCOPE_ONLY}")
    print(f"Unique chapter paths: {report['unique_chapters']}")
    print(f"Suspicious chunks: {report['suspicious_chunk_count']}")
    print("\nChunks by book:")
    for book_title, count in sorted(dict(report["book_counts"]).items()):
        print(f"- {book_title}: {count}")
    print("\nMost frequent chapters:")
    for chapter, count in report["top_chapters"]:
        print(f"- {chapter}: {count}")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    raw_documents = load_documents()
    cleaned_documents = preprocess_documents(raw_documents)
    chunks = create_chunks(cleaned_documents)
    print_chunking_report(cleaned_documents, chunks)


if __name__ == "__main__":
    main()