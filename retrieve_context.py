from __future__ import annotations

import json
import logging
import math
import os
import re
import unicodedata
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Mapping

import chromadb
import numpy as np
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

from book_catalog import TOPICS, preferred_title_score
from topic_router import (
    build_topic_anchor_ranking,
    enhanced_detect_topic_names as detect_topic_names,
)
from embeddings import create_query_embedding


LOGGER = logging.getLogger("nutrischool.retrieve_context")
PROJECT_ROOT = Path(__file__).resolve().parent
load_dotenv(PROJECT_ROOT / ".env")

CHROMA_DIRECTORY = Path(
    os.getenv("RAG_CHROMA_DIRECTORY", str(PROJECT_ROOT / "chroma_db"))
).expanduser().resolve()
COLLECTION_NAME = os.getenv(
    "RAG_COLLECTION_NAME",
    "nutrischool_pediatric_nutrition",
)
RERANKER_MODEL_NAME = os.getenv(
    "RAG_RERANKER_MODEL",
    "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1",
)
QUERY_REWRITER_MODEL = os.getenv(
    "RAG_QUERY_REWRITER_MODEL",
    "llama-3.3-70b-versatile",
)
WEIGHTS_PATH = PROJECT_ROOT / "evaluation" / "retrieval_weights.json"

DENSE_RESULTS_PER_QUERY = max(20, int(os.getenv("RAG_DENSE_RESULTS_PER_QUERY", "60")))
BM25_RESULTS_PER_QUERY = max(20, int(os.getenv("RAG_BM25_RESULTS_PER_QUERY", "50")))
RERANK_CANDIDATE_LIMIT = max(30, int(os.getenv("RAG_RERANK_CANDIDATE_LIMIT", "90")))
RRF_CONSTANT = max(1, int(os.getenv("RAG_RRF_CONSTANT", "60")))
DEFAULT_FINAL_K = max(1, int(os.getenv("RAG_FINAL_K", "5")))
ENABLE_LLM_REWRITE = os.getenv("RAG_ENABLE_LLM_REWRITE", "true").strip().lower() not in {
    "0",
    "false",
    "no",
    "off",
}

DEFAULT_WEIGHTS: dict[str, float] = {
    "rerank": 0.62,
    "semantic": 0.25,
    "rrf": 0.08,
    "metadata": 0.05,
    "bm25": 0.00,
}

QUERY_REWRITER_SYSTEM_PROMPT = """
You are a multilingual medical search-query rewriter.

The input question may be written in Arabic or English.
If the input is Arabic, silently translate it into English.

Return exactly one concise English retrieval query for pediatric-nutrition
textbooks covering school-age children and adolescents.

Rules:
- Never ask the user to translate or provide the question in English.
- Preserve the exact condition, nutrition topic, age group, and user intent.
- Use specific clinical textbook terminology.
- Do not answer the question.
- Do not explain, apologize, add labels, quotes, or bullet points.
- Return English text only.
- Use approximately 5 to 20 words.

Example Arabic input:
??? ?????? ?? ???????????? ?? ????? ????? ?????? ?????? ????????

Valid output:
pediatric diabetes carbohydrate counting meal planning school-age student
""".strip()

TOKEN_RE = re.compile(r"[A-Za-z0-9]+|[\u0600-\u06FF]+", re.UNICODE)
ARABIC_DIACRITICS_RE = re.compile(r"[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06ED]")
REFERENCE_TITLE_RE = re.compile(r"^\d{1,3}\s+[A-Z][A-Za-z'’\-]+(?:,|\s+[A-Z]{1,4},)")

STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "child",
    "children",
    "during",
    "for",
    "from",
    "how",
    "in",
    "is",
    "of",
    "on",
    "or",
    "pediatric",
    "school",
    "the",
    "to",
    "what",
    "which",
    "with",
}


@dataclass(frozen=True)
class RetrievedChunk:
    chunk_id: str
    text: str
    book_title: str
    source: str
    page_number: int
    chapter_title: str
    section_title: str
    part_title: str
    hierarchy_path: str
    semantic_similarity: float = 0.0
    bm25_score: float = 0.0
    rrf_score: float = 0.0
    rerank_score: float = 0.0
    metadata_score: float = 0.0
    final_score: float = 0.0

    def citation(self) -> str:
        hierarchy = self.chapter_title or self.section_title or self.hierarchy_path
        return f"{self.book_title}, {hierarchy}, PDF page {self.page_number}"


@dataclass(frozen=True)
class Corpus:
    ids: tuple[str, ...]
    documents: tuple[str, ...]
    metadatas: tuple[dict[str, Any], ...]
    chunks: tuple[RetrievedChunk, ...]
    id_to_index: Mapping[str, int]
    bm25: BM25Okapi


def normalize_text(value: object) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).lower()
    text = ARABIC_DIACRITICS_RE.sub("", text)
    text = (
        text.replace("\u0623", "\u0627")
        .replace("\u0625", "\u0627")
        .replace("\u0622", "\u0627")
        .replace("\u0649", "\u064a")
        .replace("\u0624", "\u0648")
        .replace("\u0626", "\u064a")
        .replace("\u0640", "")
    )
    return " ".join(TOKEN_RE.findall(text))


def tokenize(value: object) -> list[str]:
    tokens = [token for token in normalize_text(value).split() if token not in STOP_WORDS]
    return tokens or ["__empty__"]


def normalize_query(query: str) -> str:
    clean_query = " ".join(str(query or "").split()).strip()
    if not clean_query:
        raise ValueError("The query cannot be empty.")
    return clean_query


def unique_strings(values: Iterable[str]) -> list[str]:
    output: list[str] = []
    seen: set[str] = set()
    for value in values:
        clean = normalize_query(value)
        key = normalize_text(clean)
        if key and key not in seen:
            seen.add(key)
            output.append(clean)
    return output


def detect_age_scope(value: object) -> str:
    normalized = normalize_text(value)
    age_match = re.search(
        r"\bage\s*(\d{1,2})\b|\b(\d{1,2})\s*(?:years?|yrs?)\b|"
        r"\b(\d{1,2})\s*(?:\u0633\u0646\u0647|\u0633\u0646\u0648\u0627\u062a|\u0639\u0627\u0645)\b",
        normalized,
    )
    if age_match:
        age = int(next(group for group in age_match.groups() if group))
        return "school_age" if age <= 12 else "adolescent"
    if any(marker in normalized for marker in ("adolescent", "teen", "puberty", "\u0645\u0631\u0627\u0647\u0642", "\u0645\u0631\u0627\u0647\u0642\u0647", "\u0628\u0644\u0648\u063a")):
        return "adolescent"
    return "school_age_adolescent"


def deterministic_expansion(question: str) -> str:
    topics = detect_topic_names(question)
    phrases = [TOPICS[topic].retrieval_phrase for topic in topics]
    age_phrase = {
        "school_age": "school-age children",
        "adolescent": "adolescents",
        "school_age_adolescent": "school-age children and adolescents",
    }[detect_age_scope(question)]
    return " ".join([*phrases, age_phrase, "pediatric nutrition"]).strip()


@lru_cache(maxsize=1)
def load_query_rewriter() -> ChatGroq | None:
    if not ENABLE_LLM_REWRITE:
        return None
    api_key = os.getenv("GROQ_API_KEY", "").strip()
    if not api_key:
        return None
    return ChatGroq(
        api_key=api_key,
        model=QUERY_REWRITER_MODEL,
        temperature=0.0,
        max_retries=1,
        timeout=45,
    )


def rewrite_query(question: str) -> str:
    rewriter = load_query_rewriter()
    if rewriter is None:
        return ""

    clean_question = normalize_query(question)

    try:
        response = rewriter.invoke(
            [
                SystemMessage(content=QUERY_REWRITER_SYSTEM_PROMPT),
                HumanMessage(
                    content=(
                        "Rewrite this Arabic or English question as one "
                        "concise English textbook retrieval query only:\n\n"
                        f"{clean_question}"
                    )
                ),
            ]
        )

        rewritten = " ".join(
            str(response.content or "").split()
        ).strip()

        lowered = rewritten.lower()

        refusal_markers = (
            "provide the question",
            "input is not in english",
            "could you please",
            "please provide",
            "it seems like",
            "i cannot",
            "i can't",
            "as an ai",
        )

        invalid = (
            not rewritten
            or bool(re.search(r"[\u0600-\u06FF]", rewritten))
            or not bool(re.search(r"[A-Za-z]", rewritten))
            or any(marker in lowered for marker in refusal_markers)
            or len(rewritten.split()) < 3
            or len(rewritten.split()) > 30
        )

        if invalid:
            LOGGER.warning(
                "Invalid query-rewriter output ignored: %s",
                rewritten,
            )
            return ""

        return rewritten

    except Exception as error:
        LOGGER.warning(
            "Query rewriting failed; deterministic expansion will be used. %s",
            error,
        )
        return ""


@lru_cache(maxsize=1)
def get_collection():
    if not CHROMA_DIRECTORY.exists():
        raise FileNotFoundError(
            f"Chroma directory not found: {CHROMA_DIRECTORY}. Run build_index.py first."
        )
    client = chromadb.PersistentClient(path=str(CHROMA_DIRECTORY))
    collection = client.get_collection(name=COLLECTION_NAME)
    if collection.count() == 0:
        raise RuntimeError("The Chroma collection is empty.")
    return collection


def chunk_from_record(
    chunk_id: str,
    document: str,
    metadata: Mapping[str, Any],
    similarity: float = 0.0,
) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=str(chunk_id),
        text=str(document or "").strip(),
        book_title=str(metadata.get("book_title", "Unknown Book")),
        source=str(metadata.get("source", "")),
        page_number=int(metadata.get("page_number", 0) or 0),
        chapter_title=str(metadata.get("chapter_title", "")),
        section_title=str(metadata.get("section_title", "")),
        part_title=str(metadata.get("part_title", "")),
        hierarchy_path=str(metadata.get("hierarchy_path", "")),
        semantic_similarity=float(similarity),
    )


@lru_cache(maxsize=1)
def load_corpus() -> Corpus:
    result = get_collection().get(include=["documents", "metadatas"])
    ids = tuple(str(value) for value in result.get("ids", []))
    documents = tuple(str(value or "") for value in result.get("documents", []))
    metadatas = tuple(dict(value or {}) for value in result.get("metadatas", []))

    if not ids or len(ids) != len(documents) or len(ids) != len(metadatas):
        raise RuntimeError("Chroma corpus records are incomplete or inconsistent.")

    chunks = tuple(
        chunk_from_record(chunk_id, document, metadata)
        for chunk_id, document, metadata in zip(ids, documents, metadatas)
    )
    tokenized = [
        tokenize(
            f"{chunk.book_title} {chunk.hierarchy_path} {chunk.chapter_title} "
            f"{chunk.section_title} {chunk.text}"
        )
        for chunk in chunks
    ]
    return Corpus(
        ids=ids,
        documents=documents,
        metadatas=metadatas,
        chunks=chunks,
        id_to_index={chunk_id: index for index, chunk_id in enumerate(ids)},
        bm25=BM25Okapi(tokenized),
    )


@lru_cache(maxsize=1)
def load_reranker() -> CrossEncoder:
    LOGGER.info("Loading reranker: %s", RERANKER_MODEL_NAME)
    return CrossEncoder(RERANKER_MODEL_NAME, max_length=512)


def is_valid_candidate(chunk: RetrievedChunk) -> bool:
    hierarchy = chunk.hierarchy_path or chunk.chapter_title
    if not hierarchy or hierarchy.lower() == "front matter":
        return False
    if REFERENCE_TITLE_RE.match(chunk.chapter_title) or " et al" in chunk.chapter_title.lower():
        return False
    return bool(chunk.text.strip())


def dense_search(query: str, top_k: int) -> list[RetrievedChunk]:
    collection = get_collection()
    result = collection.query(
        query_embeddings=[create_query_embedding(query).tolist()],
        n_results=min(max(1, int(top_k)), collection.count()),
        include=["documents", "metadatas", "distances"],
    )
    output: list[RetrievedChunk] = []
    for chunk_id, document, metadata, distance in zip(
        result.get("ids", [[]])[0],
        result.get("documents", [[]])[0],
        result.get("metadatas", [[]])[0],
        result.get("distances", [[]])[0],
    ):
        chunk = chunk_from_record(
            str(chunk_id),
            str(document or ""),
            dict(metadata or {}),
            similarity=max(-1.0, min(1.0, 1.0 - float(distance))),
        )
        if is_valid_candidate(chunk):
            output.append(chunk)
    return output


def bm25_search(query: str, top_k: int) -> list[RetrievedChunk]:
    corpus = load_corpus()
    scores = np.asarray(corpus.bm25.get_scores(tokenize(query)), dtype=np.float32)
    if scores.size == 0:
        return []
    count = min(max(1, int(top_k)), scores.size)
    indexes = np.argpartition(scores, -count)[-count:]
    indexes = indexes[np.argsort(scores[indexes])[::-1]]
    output: list[RetrievedChunk] = []
    for index in indexes.tolist():
        base = corpus.chunks[index]
        if scores[index] <= 0 or not is_valid_candidate(base):
            continue
        output.append(replace(base, bm25_score=float(scores[index])))
    return output


def min_max(values: list[float]) -> list[float]:
    if not values:
        return []
    low = min(values)
    high = max(values)
    if math.isclose(low, high):
        return [1.0 if high > 0 else 0.0 for _ in values]
    return [(value - low) / (high - low) for value in values]


def lexical_metadata_score(query: str, chunk: RetrievedChunk) -> float:
    query_tokens = set(tokenize(query))
    hierarchy_tokens = set(tokenize(f"{chunk.chapter_title} {chunk.section_title} {chunk.hierarchy_path}"))
    overlap = len(query_tokens & hierarchy_tokens) / max(1, len(query_tokens))
    preferred = preferred_title_score(query, chunk.hierarchy_path)
    return min(1.0, 0.45 * overlap + 0.55 * preferred)


def load_retrieval_weights() -> dict[str, float]:
    if WEIGHTS_PATH.is_file():
        try:
            payload = json.loads(WEIGHTS_PATH.read_text(encoding="utf-8"))
            candidate = payload.get("weights", payload)
            weights = {key: float(candidate[key]) for key in DEFAULT_WEIGHTS}
            if math.isclose(sum(weights.values()), 1.0, abs_tol=0.02):
                return weights
        except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
            LOGGER.warning("Invalid retrieval_weights.json; using defaults.")
    return dict(DEFAULT_WEIGHTS)


def build_candidates(
    queries: list[str],
) -> tuple[list[RetrievedChunk], list[RetrievedChunk], dict[str, RetrievedChunk]]:
    candidates: dict[str, RetrievedChunk] = {}
    rrf_scores: dict[str, float] = {}
    dense_best: dict[str, RetrievedChunk] = {}
    bm25_best: dict[str, RetrievedChunk] = {}

    query_weights = [1.0, 0.90, 0.78]

    for query_index, query in enumerate(queries):
        query_weight = query_weights[min(query_index, len(query_weights) - 1)]
        dense_results = dense_search(query, DENSE_RESULTS_PER_QUERY)
        for rank, chunk in enumerate(dense_results, start=1):
            previous = dense_best.get(chunk.chunk_id)
            if previous is None or chunk.semantic_similarity > previous.semantic_similarity:
                dense_best[chunk.chunk_id] = chunk
            candidates.setdefault(chunk.chunk_id, chunk)
            rrf_scores[chunk.chunk_id] = rrf_scores.get(chunk.chunk_id, 0.0) + (
                query_weight * 1.0 / (RRF_CONSTANT + rank)
            )

        bm25_results = bm25_search(query, BM25_RESULTS_PER_QUERY)
        for rank, chunk in enumerate(bm25_results, start=1):
            previous = bm25_best.get(chunk.chunk_id)
            if previous is None or chunk.bm25_score > previous.bm25_score:
                bm25_best[chunk.chunk_id] = chunk
            candidates.setdefault(chunk.chunk_id, chunk)
            rrf_scores[chunk.chunk_id] = rrf_scores.get(chunk.chunk_id, 0.0) + (
                query_weight * 0.22 / (RRF_CONSTANT + rank)
            )

    merged: dict[str, RetrievedChunk] = {}
    for chunk_id, base in candidates.items():
        dense = dense_best.get(chunk_id)
        lexical = bm25_best.get(chunk_id)
        merged[chunk_id] = replace(
            base,
            semantic_similarity=(dense.semantic_similarity if dense else 0.0),
            bm25_score=(lexical.bm25_score if lexical else 0.0),
            rrf_score=rrf_scores.get(chunk_id, 0.0),
        )

    dense_stage = sorted(
        dense_best.values(),
        key=lambda item: item.semantic_similarity,
        reverse=True,
    )
    hybrid_stage = sorted(
        merged.values(),
        key=lambda item: (
            item.semantic_similarity + 8.0 * item.rrf_score + 0.01 * item.bm25_score
        ),
        reverse=True,
    )
    return dense_stage, hybrid_stage, merged


def rerank_candidates(query: str, candidates: list[RetrievedChunk]) -> list[RetrievedChunk]:
    limited = candidates[:RERANK_CANDIDATE_LIMIT]
    if not limited:
        return []
    pairs = [
        [
            query,
            (
                f"Book: {chunk.book_title}\n"
                f"Hierarchy: {chunk.hierarchy_path}\n"
                f"Text: {chunk.text[:2400]}"
            ),
        ]
        for chunk in limited
    ]
    try:
        raw_scores = np.asarray(
            load_reranker().predict(
                pairs,
                batch_size=8,
                show_progress_bar=False,
                convert_to_numpy=True,
            ),
            dtype=np.float32,
        ).reshape(-1)
    except Exception as error:
        LOGGER.warning("Reranking failed; fusion ordering will be retained. %s", error)
        raw_scores = np.zeros(len(limited), dtype=np.float32)

    output = [
        replace(chunk, rerank_score=float(score))
        for chunk, score in zip(limited, raw_scores.tolist())
    ]
    output.sort(key=lambda item: item.rerank_score, reverse=True)
    return output


def score_candidates(
    query: str,
    candidates: list[RetrievedChunk],
    weights: Mapping[str, float] | None = None,
) -> list[RetrievedChunk]:
    if not candidates:
        return []
    active = dict(weights or load_retrieval_weights())
    rerank_values = min_max([chunk.rerank_score for chunk in candidates])
    semantic_values = min_max([chunk.semantic_similarity for chunk in candidates])
    bm25_values = min_max([chunk.bm25_score for chunk in candidates])
    rrf_values = min_max([chunk.rrf_score for chunk in candidates])
    metadata_values = [lexical_metadata_score(query, chunk) for chunk in candidates]

    output: list[RetrievedChunk] = []
    for chunk, rerank, semantic, bm25, rrf, metadata in zip(
        candidates,
        rerank_values,
        semantic_values,
        bm25_values,
        rrf_values,
        metadata_values,
    ):
        final_score = (
            active["rerank"] * rerank
            + active["semantic"] * semantic
            + active["rrf"] * rrf
            + active["metadata"] * metadata
            + active["bm25"] * bm25
        )
        output.append(
            replace(
                chunk,
                metadata_score=float(metadata),
                final_score=float(final_score),
            )
        )
    output.sort(key=lambda item: item.final_score, reverse=True)
    return output


def deduplicate_and_diversify(
    chunks: list[RetrievedChunk],
    maximum_per_hierarchy: int = 1,
) -> list[RetrievedChunk]:
    output: list[RetrievedChunk] = []
    seen_ids: set[str] = set()
    seen_text: set[str] = set()
    hierarchy_counts: dict[tuple[str, str], int] = {}

    for chunk in chunks:
        text_key = normalize_text(chunk.text[:1000])
        hierarchy_key = (
            normalize_text(chunk.book_title),
            normalize_text(chunk.hierarchy_path or chunk.chapter_title),
        )
        if chunk.chunk_id in seen_ids or text_key in seen_text:
            continue
        if hierarchy_counts.get(hierarchy_key, 0) >= maximum_per_hierarchy:
            continue
        seen_ids.add(chunk.chunk_id)
        seen_text.add(text_key)
        hierarchy_counts[hierarchy_key] = hierarchy_counts.get(hierarchy_key, 0) + 1
        output.append(chunk)

    return output


def stage_top_chunks(chunks: list[RetrievedChunk], count: int) -> list[RetrievedChunk]:
    return deduplicate_and_diversify(chunks, maximum_per_hierarchy=1)[:count]


def build_context(chunks: list[RetrievedChunk], maximum_characters: int = 12000) -> str:
    blocks: list[str] = []
    used = 0
    for index, chunk in enumerate(chunks, start=1):
        block = (
            f"[SOURCE {index}]\n"
            f"Book: {chunk.book_title}\n"
            f"Hierarchy: {chunk.hierarchy_path}\n"
            f"PDF page: {chunk.page_number}\n"
            f"Text:\n{chunk.text}\n"
        )
        if blocks and used + len(block) > maximum_characters:
            break
        blocks.append(block)
        used += len(block)
    return "\n".join(blocks)


def retrieve_context(
    question: str,
    top_k: int = 50,
    final_k: int = DEFAULT_FINAL_K,
    weights: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    original_question = normalize_query(question)
    rewritten_query = rewrite_query(original_question)
    expansion = deterministic_expansion(original_question)
    queries = unique_strings(
        value
        for value in (original_question, expansion, rewritten_query)
        if value.strip()
    )

    dense_stage, hybrid_stage, merged = build_candidates(queries)
    preliminary = hybrid_stage[: max(top_k, RERANK_CANDIDATE_LIMIT)]
    rerank_query = " | ".join(unique_strings([expansion, rewritten_query or original_question, original_question]))
    reranked = rerank_candidates(rerank_query, preliminary)
    scored = score_candidates(rerank_query, reranked, weights=weights)
    final_chunks = build_topic_anchor_ranking(
        original_question,
        dense_stage=dense_stage,
        hybrid_stage=hybrid_stage,
        reranked_stage=reranked,
        final_k=max(1, int(final_k)),
    )

    return {
        "question": original_question,
        "rewritten_query": rewritten_query,
        "expanded_query": expansion,
        "queries": queries,
        "weights": dict(weights or load_retrieval_weights()),
        "dense_chunks": stage_top_chunks(dense_stage, max(5, final_k)),
        "hybrid_chunks": stage_top_chunks(hybrid_stage, max(5, final_k)),
        "reranked_chunks": stage_top_chunks(reranked, max(5, final_k)),
        "candidate_chunks": scored,
        "chunks": final_chunks,
        "context": build_context(final_chunks),
        "detected_topics": list(detect_topic_names(original_question)),
        "age_scope": detect_age_scope(original_question),
    }


def clear_runtime_caches() -> None:
    get_collection.cache_clear()
    load_corpus.cache_clear()
    load_reranker.cache_clear()
    load_query_rewriter.cache_clear()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    question = "How should a school manage meals for a student with food allergy?"
    result = retrieve_context(question=question, final_k=5)
    print("\nCONTEXT RETRIEVAL REPORT")
    print("=" * 72)
    print(f"Question: {result['question']}")
    print(f"Expanded: {result['expanded_query']}")
    print(f"Rewritten: {result['rewritten_query']}")
    print(f"Weights: {result['weights']}")
    for index, chunk in enumerate(result["chunks"], start=1):
        print(f"\nSource {index}")
        print(f"Score: {chunk.final_score:.4f}")
        print(f"Citation: {chunk.citation()}")
        print(f"Preview: {chunk.text[:260].replace(chr(10), ' ')}")


if __name__ == "__main__":
    main()