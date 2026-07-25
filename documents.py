from __future__ import annotations

import hashlib
import logging
import os
import re
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document


LOGGER = logging.getLogger("nutrischool.documents")
PROJECT_ROOT = Path(__file__).resolve().parent


def resolve_books_directory() -> Path:
    configured = os.getenv("RAG_BOOKS_DIR", "").strip()

    if configured:
        return Path(configured).expanduser().resolve()

    candidates = (
        PROJECT_ROOT / "books",
        PROJECT_ROOT / "data" / "books",
    )

    for candidate in candidates:
        if candidate.is_dir():
            return candidate.resolve()

    return candidates[0].resolve()


BOOKS_DIRECTORY = resolve_books_directory()
MIN_PAGE_CHARACTERS = max(20, int(os.getenv("RAG_MIN_PAGE_CHARACTERS", "40")))


def canonical_book_metadata(pdf_path: Path) -> tuple[str, str]:
    normalized = re.sub(r"[^a-z0-9]+", " ", pdf_path.stem.lower()).strip()

    if "pnp 3rd edition" in normalized:
        return "pnp3", "PNP 3rd Edition Book"

    if "pediatric nutrition" in normalized:
        return "aap8", "Pediatric-Nutrition"

    safe_id = re.sub(r"[^a-z0-9]+", "_", normalized).strip("_") or "book"
    return safe_id, pdf_path.stem.replace("_", " ").strip()


def discover_pdf_files(books_directory: Path = BOOKS_DIRECTORY) -> list[Path]:
    if not books_directory.exists():
        raise FileNotFoundError(
            f"Books directory was not found: {books_directory}\n"
            "Expected the two PDF files inside a folder named 'books' beside documents.py."
        )

    if not books_directory.is_dir():
        raise NotADirectoryError(f"Expected a directory but found: {books_directory}")

    pdf_files = sorted(path for path in books_directory.rglob("*.pdf") if path.is_file())

    if not pdf_files:
        raise FileNotFoundError(f"No PDF files were found inside: {books_directory}")

    known_ids = [canonical_book_metadata(path)[0] for path in pdf_files]
    required = {"aap8", "pnp3"}
    missing = required - set(known_ids)

    if missing:
        raise FileNotFoundError(
            "Both source books are required. Missing canonical book IDs: "
            + ", ".join(sorted(missing))
        )

    return pdf_files


def file_fingerprint(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while block := file.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def load_pdf_pages(pdf_path: Path) -> list[Document]:
    loader = PyPDFLoader(str(pdf_path))
    raw_pages = loader.load()
    book_id, book_title = canonical_book_metadata(pdf_path)
    fingerprint = file_fingerprint(pdf_path)
    pages: list[Document] = []

    for pdf_page_index, page in enumerate(raw_pages):
        metadata = dict(page.metadata or {})
        metadata.update(
            {
                "book_id": book_id,
                "book_key": book_id,
                "book_title": book_title,
                "source": pdf_path.name,
                "source_path": str(pdf_path.resolve()),
                "source_sha256": fingerprint,
                "pdf_page_index": pdf_page_index,
                "page_number": pdf_page_index + 1,
                "page_label": str(pdf_page_index + 1),
                "document_type": "pdf",
            }
        )
        pages.append(
            Document(
                page_content=str(page.page_content or ""),
                metadata=metadata,
            )
        )

    return pages


def load_documents(
    books_directory: Path = BOOKS_DIRECTORY,
    minimum_page_characters: int = MIN_PAGE_CHARACTERS,
) -> list[Document]:
    pdf_files = discover_pdf_files(books_directory)
    documents: list[Document] = []
    skipped_pages = 0

    for pdf_path in pdf_files:
        LOGGER.info("Loading PDF: %s", pdf_path.name)

        for page in load_pdf_pages(pdf_path):
            text = str(page.page_content or "").strip()

            if len(text) < minimum_page_characters:
                skipped_pages += 1
                LOGGER.debug(
                    "Skipped low-text page: %s page %s (%s characters)",
                    page.metadata.get("source"),
                    page.metadata.get("page_number"),
                    len(text),
                )
                continue

            documents.append(page)

    if not documents:
        raise ValueError("No usable text pages were extracted from the PDF files.")

    found_books = {str(document.metadata.get("book_id")) for document in documents}
    if found_books != {"aap8", "pnp3"}:
        raise RuntimeError(f"Unexpected extracted books: {sorted(found_books)}")

    LOGGER.info(
        "Loaded %s usable pages from %s books; skipped %s low-text pages.",
        len(documents),
        len(pdf_files),
        skipped_pages,
    )
    return documents


def print_document_report(documents: list[Document]) -> None:
    totals: dict[str, dict[str, int]] = {}

    for document in documents:
        book_title = str(document.metadata.get("book_title", "Unknown Book"))
        stats = totals.setdefault(book_title, {"pages": 0, "characters": 0})
        stats["pages"] += 1
        stats["characters"] += len(document.page_content)

    print("\nDOCUMENT EXTRACTION REPORT")
    print("=" * 72)
    print(f"Books directory: {BOOKS_DIRECTORY}")

    for book_title, stats in sorted(totals.items()):
        print(
            f"- {book_title}: {stats['pages']} usable pages, "
            f"{stats['characters']:,} extracted characters"
        )

    print(f"Total usable pages: {len(documents)}")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    documents = load_documents()
    print_document_report(documents)


if __name__ == "__main__":
    main()