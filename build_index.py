from __future__ import annotations

import pickle
import re
from pathlib import Path

import faiss
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

BOOKS_DIR = Path("books")
INDEX_DIR = Path("rag_index")
EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"


def clean_text(text: str) -> str:
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[�□■◆]", " ", text)
    return text.strip()


def load_books() -> list:
    pages = []
    for pdf in sorted(BOOKS_DIR.glob("*.pdf")):
        for page in PyPDFLoader(str(pdf)).load():
            text = clean_text(page.page_content)
            if len(text) < 80:
                continue
            page.page_content = text
            page.metadata.update(
                book=pdf.stem,
                source=pdf.name,
                page=int(page.metadata.get("page", 0)) + 1,
            )
            pages.append(page)
    return pages


def main() -> None:
    pages = load_books()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(pages)

    unique, seen = [], set()
    for chunk in chunks:
        key = re.sub(r"\s+", " ", chunk.page_content).lower()
        if key in seen:
            continue
        seen.add(key)
        chunk.metadata["chunk_id"] = len(unique)
        unique.append(chunk)

    model = SentenceTransformer(EMBED_MODEL)
    vectors = model.encode(
        [doc.page_content for doc in unique],
        normalize_embeddings=True,
        show_progress_bar=True,
    ).astype("float32")

    index = faiss.IndexHNSWFlat(vectors.shape[1], 32, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = 200
    index.hnsw.efSearch = 80
    index.add(vectors)

    INDEX_DIR.mkdir(exist_ok=True)
    faiss.write_index(index, str(INDEX_DIR / "hnsw.faiss"))
    with open(INDEX_DIR / "chunks.pkl", "wb") as file:
        pickle.dump(unique, file)

    print(f"Indexed {len(unique)} chunks from {len(pages)} pages.")


if __name__ == "__main__":
    main()