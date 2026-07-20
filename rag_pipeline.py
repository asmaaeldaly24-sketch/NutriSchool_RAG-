from __future__ import annotations

import json
import pickle
import time
from functools import lru_cache
from pathlib import Path

import faiss
import numpy as np
from langchain_groq import ChatGroq
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer


INDEX_DIR = Path("rag_index")
LOG_FILE = Path("logs/retrieval.jsonl")

EMBED_MODEL = (
    "sentence-transformers/"
    "paraphrase-multilingual-mpnet-base-v2"
)

RERANK_MODEL = (
    "cross-encoder/"
    "mmarco-mMiniLMv2-L12-H384-v1"
)


def tokenize(text: str) -> list[str]:
    return text.lower().split()


@lru_cache(maxsize=1)
def load_pipeline() -> dict:
    with open(INDEX_DIR / "chunks.pkl", "rb") as file:
        chunks = pickle.load(file)

    index = faiss.read_index(
        str(INDEX_DIR / "hnsw.faiss")
    )

    embedder = SentenceTransformer(
        EMBED_MODEL
    )

    bm25 = BM25Okapi(
        [
            tokenize(doc.page_content)
            for doc in chunks
        ]
    )

    reranker = CrossEncoder(
        RERANK_MODEL
    )

    llm = ChatGroq(
        model="openai/gpt-oss-120b",
        temperature=0.1,
        max_tokens=900,
    )

    return {
        "chunks": chunks,
        "index": index,
        "embedder": embedder,
        "bm25": bm25,
        "reranker": reranker,
        "llm": llm,
    }


def allowed(
    doc,
    filters: dict | None,
) -> bool:
    if not filters:
        return True

    return all(
        value in (None, "", "All")
        or doc.metadata.get(key) == value
        for key, value in filters.items()
    )


def rrf(
    *rankings: list[int],
    k: int = 60,
) -> list[int]:
    scores: dict[int, float] = {}

    for ranking in rankings:
        for rank, doc_id in enumerate(
            ranking,
            start=1,
        ):
            scores[doc_id] = (
                scores.get(doc_id, 0.0)
                + 1 / (k + rank)
            )

    return sorted(
        scores,
        key=scores.get,
        reverse=True,
    )


def retrieve(
    question: str,
    filters: dict | None = None,
    top_k: int = 3,
    profile: str = "",
) -> tuple[list, list[float], str]:
    started = time.perf_counter()
    pipeline = load_pipeline()

    query = question.strip()

    if profile.strip():
        query = f"{question}\nChild profile: {profile}"

    vector = pipeline["embedder"].encode(
        [query],
        normalize_embeddings=True,
    ).astype("float32")

    _, dense_ids = pipeline["index"].search(
        vector,
        15,
    )

    bm25_scores = pipeline["bm25"].get_scores(
        tokenize(query)
    )

    bm25_ids = np.argsort(
        bm25_scores
    )[::-1][:15].tolist()

    fused_ids = rrf(
        dense_ids[0].tolist(),
        bm25_ids,
    )

    candidate_ids = []

    for doc_id in fused_ids:
        if doc_id < 0:
            continue

        doc = pipeline["chunks"][doc_id]

        if allowed(doc, filters):
            candidate_ids.append(doc_id)

        if len(candidate_ids) >= 8:
            break

    if not candidate_ids:
        return [], [], query

    pairs = [
        (
            question,
            pipeline["chunks"][doc_id].page_content,
        )
        for doc_id in candidate_ids
    ]

    rerank_scores = pipeline[
        "reranker"
    ].predict(pairs)

    ranked = sorted(
        zip(
            candidate_ids,
            rerank_scores,
        ),
        key=lambda item: float(item[1]),
        reverse=True,
    )[:top_k]

    docs = [
        pipeline["chunks"][doc_id]
        for doc_id, _ in ranked
    ]

    scores = [
        float(score)
        for _, score in ranked
    ]

    LOG_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        LOG_FILE,
        "a",
        encoding="utf-8",
    ) as file:
        file.write(
            json.dumps(
                {
                    "question": question,
                    "query": query,
                    "filters": filters,
                    "chunk_ids": [
                        doc.metadata.get(
                            "chunk_id",
                            "unknown",
                        )
                        for doc in docs
                    ],
                    "rerank_scores": scores,
                    "latency_ms": round(
                        (
                            time.perf_counter()
                            - started
                        )
                        * 1000,
                        2,
                    ),
                },
                ensure_ascii=False,
            )
            + "\n"
        )

    return docs, scores, query


def answer(
    question: str,
    profile: str = "",
    filters: dict | None = None,
) -> str:
    docs, _, _ = retrieve(
        question=question,
        filters=filters,
        profile=profile,
    )

    if not docs:
        is_arabic = any(
            "\u0600" <= char <= "\u06FF"
            for char in question
        )

        if is_arabic:
            return (
                "عذرًا، لم أجد معلومات كافية "
                "في المراجع المتاحة للإجابة عن هذا السؤال."
            )

        return (
            "Sorry, I could not find enough information "
            "in the available references."
        )

    context = "\n\n".join(
        doc.page_content
        for doc in docs
    )

    prompt = f"""
You are an evidence-based pediatric nutrition assistant.

Rules:
- Answer the user's question using only the retrieved evidence below.
- Personalize the answer using the child's profile when relevant.
- Focus on practical pediatric nutrition guidance.
- Prefer affordable and commonly available Egyptian foods when suitable.
- Respect allergies, intolerances, health conditions, and disliked foods.
- If the evidence is insufficient, state that clearly and do not guess.
- Answer in the same language as the user's question.
- Keep the answer clear, practical, and concise.
- Preserve numbers and measurement units exactly.
- Use metric units such as grams, kilograms, milliliters, cups,
  tablespoons, eggs, slices, and baladi bread.
- Do not use imperial units.
- Do not show source names, page numbers, file names,
  citation numbers, or a Sources section.
- Do not diagnose diseases.
- Do not replace medical advice from a pediatrician
  or registered dietitian.

Child profile:
{profile}

Question:
{question}

Retrieved evidence:
{context}
"""

    result = load_pipeline()["llm"].invoke(
        prompt
    )

    return result.content.strip()