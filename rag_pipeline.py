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
EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
RERANK_MODEL = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
MIN_RERANK_SCORE = -1.5


def tokenize(text: str) -> list[str]:
    return text.lower().split()


@lru_cache(maxsize=1)
def load_pipeline() -> dict:
    with open(INDEX_DIR / "chunks.pkl", "rb") as file:
        chunks = pickle.load(file)

    return {
        "chunks": chunks,
        "index": faiss.read_index(str(INDEX_DIR / "hnsw.faiss")),
        "embedder": SentenceTransformer(EMBED_MODEL),
        "bm25": BM25Okapi([tokenize(doc.page_content) for doc in chunks]),
        "reranker": CrossEncoder(RERANK_MODEL),
        "llm": ChatGroq(
            model="openai/gpt-oss-120b",
            temperature=0.1,
            max_tokens=1200,
        ),
    }


def allowed(doc, filters: dict | None) -> bool:
    if not filters:
        return True

    return all(
        value in (None, "", "All")
        or doc.metadata.get(key) == value
        for key, value in filters.items()
    )


def rrf(*rankings: list[int], k: int = 60) -> list[int]:
    scores: dict[int, float] = {}

    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1 / (k + rank)

    return sorted(scores, key=scores.get, reverse=True)


def rewrite_query(question: str, profile: str = "") -> str:
    prompt = f"""Rewrite the question as one precise English retrieval query.
Keep pediatric nutrition, age, condition, allergy, food and nutrient terms.
Do not answer the question.

Question: {question}
Child profile: {profile}
"""
    result = load_pipeline()["llm"].invoke(prompt)
    return result.content.strip()


def retrieve(
    question: str,
    filters: dict | None = None,
    top_k: int = 5,
    profile: str = "",
) -> tuple[list, list[float], str]:
    started = time.perf_counter()
    pipeline = load_pipeline()
    query = rewrite_query(question, profile)

    vector = pipeline["embedder"].encode(
        [query],
        normalize_embeddings=True,
    ).astype("float32")

    _, dense_ids = pipeline["index"].search(vector, 30)

    bm25_scores = pipeline["bm25"].get_scores(tokenize(query))
    bm25_ids = np.argsort(bm25_scores)[::-1][:30].tolist()

    candidate_ids = [
        doc_id
        for doc_id in rrf(dense_ids[0].tolist(), bm25_ids)
        if allowed(pipeline["chunks"][doc_id], filters)
    ][:20]

    if not candidate_ids:
        return [], [], query

    pairs = [
        (query, pipeline["chunks"][doc_id].page_content)
        for doc_id in candidate_ids
    ]
    rerank_scores = pipeline["reranker"].predict(pairs)

    ranked = sorted(
        zip(candidate_ids, rerank_scores),
        key=lambda item: float(item[1]),
        reverse=True,
    )[:top_k]

    docs = [pipeline["chunks"][doc_id] for doc_id, _ in ranked]
    scores = [float(score) for _, score in ranked]

    LOG_FILE.parent.mkdir(exist_ok=True)
    with open(LOG_FILE, "a", encoding="utf-8") as file:
        file.write(
            json.dumps(
                {
                    "question": question,
                    "query": query,
                    "filters": filters,
                    "chunk_ids": [doc.metadata["chunk_id"] for doc in docs],
                    "rerank_scores": scores,
                    "latency_ms": round(
                        (time.perf_counter() - started) * 1000,
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
    docs, scores, _ = retrieve(
        question=question,
        filters=filters,
        profile=profile,
    )

    if not docs or not scores or scores[0] < MIN_RERANK_SCORE:
        return (
            "عذرًا، هذا السؤال خارج نطاق تغذية الأطفال والمراجع المتاحة."
            if any("\u0600" <= char <= "\u06FF" for char in question)
            else "Sorry, this question is outside the pediatric nutrition scope of the available references."
        )

    context = "\n\n".join(
        f"[{index}] {doc.metadata['source']}, page {doc.metadata['page']}\n"
        f"{doc.page_content}"
        for index, doc in enumerate(docs, start=1)
    )

    prompt = f"""You are a pediatric nutrition assistant.

Rules:
- Answer only questions about pediatric nutrition, child feeding, growth, food allergies, nutrients, school meals, or diet-related child health.
- Use only the evidence below.
- If the question is outside this scope, refuse briefly.
- If the evidence is insufficient, say that clearly and do not guess.
- Answer in the same language as the question.
- Preserve numbers and units exactly.
- Cite every factual paragraph using [1], [2], and so on.
- End with a short Sources section.
- Do not diagnose and do not replace a doctor.

Child profile:
{profile}

Question:
{question}

Evidence:
{context}
"""

    result = load_pipeline()["llm"].invoke(prompt)
    return result.content.strip()