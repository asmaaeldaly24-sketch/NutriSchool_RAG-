from __future__ import annotations

import logging
import os
from functools import lru_cache

import numpy as np
from langchain_core.documents import Document
from sentence_transformers import SentenceTransformer

from chunking import create_chunks
from documents import load_documents
from preprocessing import preprocess_documents


LOGGER = logging.getLogger("nutrischool.embeddings")
EMBEDDING_MODEL_NAME = os.getenv(
    "RAG_EMBEDDING_MODEL",
    "intfloat/multilingual-e5-base",
)
EMBEDDING_BATCH_SIZE = max(1, int(os.getenv("RAG_EMBEDDING_BATCH_SIZE", "24")))
EMBEDDING_DEVICE = os.getenv("RAG_EMBEDDING_DEVICE", "").strip() or None


@lru_cache(maxsize=1)
def load_embedding_model() -> SentenceTransformer:
    LOGGER.info("Loading embedding model: %s", EMBEDDING_MODEL_NAME)
    model = SentenceTransformer(EMBEDDING_MODEL_NAME, device=EMBEDDING_DEVICE)
    model.max_seq_length = min(max(int(getattr(model, "max_seq_length", 512)), 384), 512)
    return model


def build_passage_text(chunk: Document) -> str:
    metadata = dict(chunk.metadata or {})
    fields = [
        f"Book: {metadata.get('book_title', '')}",
        f"Hierarchy: {metadata.get('hierarchy_path', '')}",
        f"Chapter: {metadata.get('chapter_title', '')}",
        f"Section: {metadata.get('section_title', '')}",
        f"Content: {chunk.page_content}",
    ]
    payload = "\n".join(field for field in fields if not field.endswith(": "))
    return f"passage: {payload}"


def build_query_text(query: str) -> str:
    clean_query = " ".join(str(query or "").split()).strip()
    if not clean_query:
        raise ValueError("The query cannot be empty.")
    return f"query: {clean_query}"


def create_embeddings(
    chunks: list[Document],
    batch_size: int = EMBEDDING_BATCH_SIZE,
) -> np.ndarray:
    if not chunks:
        raise ValueError("The chunk list is empty.")

    texts = [build_passage_text(chunk) for chunk in chunks]
    embeddings = load_embedding_model().encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    embeddings = np.asarray(embeddings, dtype=np.float32)

    if embeddings.ndim != 2:
        raise ValueError("Embeddings must be a two-dimensional matrix.")
    if embeddings.shape[0] != len(chunks):
        raise ValueError("The embeddings count does not match the chunks count.")
    if not np.isfinite(embeddings).all():
        raise ValueError("Embeddings contain invalid values.")

    norms = np.linalg.norm(embeddings, axis=1)
    if not np.allclose(norms, 1.0, atol=1e-3):
        raise ValueError("Embedding normalization check failed.")

    LOGGER.info("Created embeddings with shape %s.", embeddings.shape)
    return embeddings


def create_query_embedding(query: str) -> np.ndarray:
    embedding = load_embedding_model().encode(
        [build_query_text(query)],
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    vector = np.asarray(embedding[0], dtype=np.float32)
    if vector.ndim != 1 or not np.isfinite(vector).all():
        raise ValueError("Invalid query embedding.")
    return vector


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    raw_documents = load_documents()
    cleaned_documents = preprocess_documents(raw_documents)
    chunks = create_chunks(cleaned_documents)
    embeddings = create_embeddings(chunks)

    print("\nEMBEDDINGS REPORT")
    print("=" * 72)
    print(f"Model: {EMBEDDING_MODEL_NAME}")
    print(f"Chunks: {len(chunks)}")
    print(f"Shape: {embeddings.shape}")
    print(f"First vector norm: {np.linalg.norm(embeddings[0]):.6f}")


if __name__ == "__main__":
    main()