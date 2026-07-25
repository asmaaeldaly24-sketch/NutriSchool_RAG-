from __future__ import annotations

import json
import logging
from pathlib import Path

from vector_store import (
    CHROMA_DIRECTORY,
    COLLECTION_NAME,
    INDEX_MANIFEST_PATH,
    build_vector_store,
)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    print("=" * 72)
    print("REBUILDING NUTRISCHOOL SCHOOL-AGE INDEX")
    print("=" * 72)
    print(f"Chroma directory: {CHROMA_DIRECTORY}")
    print(f"Collection: {COLLECTION_NAME}")

    collection = build_vector_store(reset=True)
    manifest = json.loads(Path(INDEX_MANIFEST_PATH).read_text(encoding="utf-8"))

    print("\nINDEX BUILD COMPLETED")
    print("=" * 72)
    print(f"Stored records: {collection.count()}")
    print(f"Embedding model: {manifest['embedding_model']}")
    print(f"Embedding dimension: {manifest['embedding_dimension']}")
    print(f"Unique hierarchy paths: {manifest['unique_hierarchy_paths']}")
    print(f"Books: {manifest['books']}")
    print(f"Suspicious chunks: {manifest['audit']['suspicious_chunk_count']}")
    print(f"Manifest: {INDEX_MANIFEST_PATH}")


if __name__ == "__main__":
    main()