from __future__ import annotations

import hashlib
import html
import json
import logging
import os
import pickle
import re
import threading
import time
import unicodedata
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import faiss
import numpy as np
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer

load_dotenv()

PROJECT_DIR = Path(__file__).resolve().parent


def _project_path(value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else PROJECT_DIR / path


INDEX_DIR = _project_path(os.getenv("RAG_INDEX_DIR", "rag_index"))
CHUNKS_FILE = INDEX_DIR / "chunks.pkl"
FAISS_FILE = INDEX_DIR / "hnsw.faiss"
MANIFEST_FILE = INDEX_DIR / "manifest.json"
LOG_FILE = _project_path(os.getenv("RAG_LOG_FILE", "logs/retrieval.jsonl"))

_DEFAULT_EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
_configured_embed_model = os.getenv("RAG_EMBED_MODEL", "").strip()
if _configured_embed_model:
    EMBED_MODEL = _configured_embed_model
elif MANIFEST_FILE.is_file():
    try:
        _manifest_payload = json.loads(MANIFEST_FILE.read_text(encoding="utf-8"))
        EMBED_MODEL = str(
            _manifest_payload.get("embedding_model") or _DEFAULT_EMBED_MODEL
        ).strip()
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        EMBED_MODEL = _DEFAULT_EMBED_MODEL
else:
    EMBED_MODEL = _DEFAULT_EMBED_MODEL

RERANK_MODEL = os.getenv(
    "RAG_RERANK_MODEL",
    "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1",
)
LLM_MODEL = os.getenv("RAG_LLM_MODEL", "openai/gpt-oss-120b")

DENSE_FETCH_K = max(10, int(os.getenv("RAG_DENSE_FETCH_K", "50")))
BM25_FETCH_K = max(10, int(os.getenv("RAG_BM25_FETCH_K", "50")))
RERANK_CANDIDATE_K = max(10, int(os.getenv("RAG_RERANK_CANDIDATE_K", "35")))
DEFAULT_TOP_K = max(3, int(os.getenv("RAG_TOP_K", "7")))
RRF_K = max(1, int(os.getenv("RAG_RRF_K", "60")))
MAX_EVIDENCE_CHARS = max(700, int(os.getenv("RAG_MAX_EVIDENCE_CHARS", "3500")))
MAX_PROFILE_CHARS = max(300, int(os.getenv("RAG_MAX_PROFILE_CHARS", "1800")))
ENABLE_GROUNDING_VALIDATION = os.getenv(
    "RAG_VALIDATE_GROUNDING",
    "true",
).strip().lower() not in {"0", "false", "no", "off"}
LOG_RAW_QUERIES = os.getenv(
    "RAG_LOG_RAW_QUERIES",
    "false",
).strip().lower() in {"1", "true", "yes", "on"}

HNSW_M = max(8, int(os.getenv("RAG_HNSW_M", "32")))
HNSW_EF_CONSTRUCTION = max(40, int(os.getenv("RAG_HNSW_EF_CONSTRUCTION", "200")))
HNSW_EF_SEARCH = max(20, int(os.getenv("RAG_HNSW_EF_SEARCH", "96")))
DEFAULT_CHUNK_SIZE = max(300, int(os.getenv("RAG_CHUNK_SIZE", "900")))
DEFAULT_CHUNK_OVERLAP = max(30, int(os.getenv("RAG_CHUNK_OVERLAP", "180")))

LOGGER = logging.getLogger("elite_school_rag")
_RESOURCE_LOCK = threading.RLock()
_LOG_LOCK = threading.Lock()

_ARABIC_DIACRITICS_RE = re.compile(r"[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06ED]")
_TOKEN_RE = re.compile(r"[A-Za-z0-9]+|[\u0600-\u06FF]+", re.UNICODE)
_WHITESPACE_RE = re.compile(r"\s+")
_MARKDOWN_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)
_CITATION_IN_TEXT_RE = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")
_SOURCE_HEADING_RE = re.compile(
    r"(?:^|\n)\s*(?:sources?|references?|المصادر|المراجع)\s*:?\s*.*$",
    re.IGNORECASE | re.DOTALL,
)


@dataclass(frozen=True)
class QueryPlan:
    language: str
    queries: tuple[str, ...]
    in_scope_hint: bool = True


@dataclass(frozen=True)
class CitationReference:
    number: int
    title: str
    page: str = ""
    section: str = ""


@dataclass(frozen=True)
class CitedText:
    text: str
    citations: tuple[int, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class GuidanceRow:
    recommendation: str
    suitable_foods: str
    quantity_frequency: str
    practical_notes: str
    warnings: str
    citations: tuple[int, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class NutritionAnswer:
    status: str
    language: str
    title: str
    summary: tuple[CitedText, ...] = field(default_factory=tuple)
    rows: tuple[GuidanceRow, ...] = field(default_factory=tuple)
    closing_note: CitedText | None = None
    medical_notice: str = ""
    references: tuple[CitationReference, ...] = field(default_factory=tuple)
    retrieval_query: str = ""

    @property
    def is_success(self) -> bool:
        return self.status == "ok" and bool(self.summary or self.rows)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PipelineResources:
    chunks: list[Any]
    index: Any
    embedder: SentenceTransformer
    bm25: BM25Okapi
    reranker: CrossEncoder
    llm: ChatGroq
    chunk_tokens: list[list[str]]


_LEGACY_DOCUMENT_MODULES = {
    "langchain.schema",
    "langchain.schema.document",
    "langchain.docstore.document",
    "langchain_core.documents",
    "langchain_core.documents.base",
}


class _CompatibleUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str) -> Any:
        if name == "Document" and module in _LEGACY_DOCUMENT_MODULES:
            return Document
        return super().find_class(module, name)


def detect_language(text: str) -> str:
    arabic_count = len(re.findall(r"[\u0600-\u06FF]", text or ""))
    latin_count = len(re.findall(r"[A-Za-z]", text or ""))
    return "ar" if arabic_count >= latin_count else "en"


def _normalise_arabic(text: str) -> str:
    text = _ARABIC_DIACRITICS_RE.sub("", text)
    return (
        text.replace("أ", "ا")
        .replace("إ", "ا")
        .replace("آ", "ا")
        .replace("ى", "ي")
        .replace("ؤ", "و")
        .replace("ئ", "ي")
        .replace("ـ", "")
    )


def normalise_text(text: Any) -> str:
    value = unicodedata.normalize("NFKC", str(text or ""))
    value = html.unescape(value)
    value = _normalise_arabic(value.lower())
    return _WHITESPACE_RE.sub(" ", value).strip()


def tokenize(text: str) -> list[str]:
    tokens = _TOKEN_RE.findall(normalise_text(text))
    return tokens or ["__empty__"]


def _doc_text(doc: Any) -> str:
    if isinstance(doc, str):
        return doc
    if isinstance(doc, Document):
        return str(doc.page_content or "")
    if isinstance(doc, Mapping):
        return str(
            doc.get("page_content")
            or doc.get("content")
            or doc.get("text")
            or ""
        )
    return str(getattr(doc, "page_content", "") or "")


def _doc_metadata(doc: Any) -> dict[str, Any]:
    if isinstance(doc, Document):
        return dict(doc.metadata or {})
    if isinstance(doc, Mapping):
        metadata = doc.get("metadata", {})
        return dict(metadata) if isinstance(metadata, Mapping) else {}
    metadata = getattr(doc, "metadata", {})
    return dict(metadata) if isinstance(metadata, Mapping) else {}


def _safe_metadata_value(metadata: Mapping[str, Any], *keys: str) -> str:
    for key in keys:
        value = metadata.get(key)
        if value not in (None, "", [], {}):
            if isinstance(value, (list, tuple, set)):
                return ", ".join(str(item) for item in value if item not in (None, ""))
            return str(value)
    return ""


def _reference_for_doc(doc: Any, number: int) -> CitationReference:
    metadata = _doc_metadata(doc)
    title = _safe_metadata_value(
        metadata,
        "document_title",
        "reference_title",
        "title",
        "chapter_title",
        "book_title",
        "publication_title",
    )
    section = _safe_metadata_value(
        metadata,
        "section_title",
        "section",
        "heading",
        "chapter",
    )
    page = _safe_metadata_value(
        metadata,
        "page_label",
        "page_number",
        "page",
    )

    # Do not expose a path or filename when the index only contains `source`.
    if not title:
        title = f"Reference {number}"

    title = _clean_plain_text(title, max_length=160) or f"Reference {number}"
    lower_title = title.lower()
    filename_extensions = (
        ".pdf",
        ".doc",
        ".docx",
        ".txt",
        ".csv",
        ".xlsx",
        ".xls",
        ".ppt",
        ".pptx",
        ".json",
        ".html",
        ".htm",
    )
    if (
        "/" in title
        or "\\" in title
        or lower_title.endswith(filename_extensions)
    ):
        title = f"Reference {number}"
    section = _clean_plain_text(section, max_length=140)
    page = _clean_plain_text(page, max_length=30)

    return CitationReference(
        number=number,
        title=title,
        page=page,
        section=section,
    )


def _validate_required_files() -> None:
    missing = [path for path in (CHUNKS_FILE, FAISS_FILE) if not path.is_file()]
    if missing:
        missing_names = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(
            "The RAG index is incomplete. Missing: "
            f"{missing_names}. Keep the existing rag_index folder beside rag_pipeline.py."
        )


def _load_chunks() -> list[Any]:
    with CHUNKS_FILE.open("rb") as file:
        loaded = _CompatibleUnpickler(file).load()

    if not isinstance(loaded, (list, tuple)) or not loaded:
        raise ValueError("chunks.pkl must contain a non-empty list of indexed chunks.")

    chunks = list(loaded)
    empty_count = sum(not _doc_text(doc).strip() for doc in chunks)
    if empty_count == len(chunks):
        raise ValueError("All indexed chunks are empty.")

    return chunks


def _new_llm(temperature: float = 0.05, max_tokens: int = 1800) -> ChatGroq:
    if not os.getenv("GROQ_API_KEY"):
        raise RuntimeError("GROQ_API_KEY is missing from the environment or .env file.")

    return ChatGroq(
        model=LLM_MODEL,
        temperature=temperature,
        max_tokens=max_tokens,
        max_retries=2,
        timeout=90,
    )


@lru_cache(maxsize=1)
def load_pipeline() -> PipelineResources:
    with _RESOURCE_LOCK:
        _validate_required_files()
        chunks = _load_chunks()
        index = faiss.read_index(str(FAISS_FILE))

        if int(index.ntotal) != len(chunks):
            raise ValueError(
                "The FAISS index and chunks.pkl are inconsistent: "
                f"index.ntotal={index.ntotal}, chunks={len(chunks)}."
            )

        if hasattr(index, "hnsw"):
            index.hnsw.efSearch = HNSW_EF_SEARCH

        embedder = SentenceTransformer(EMBED_MODEL)
        embedding_dimension = int(embedder.get_sentence_embedding_dimension())
        if int(index.d) != embedding_dimension:
            raise ValueError(
                "Embedding dimension mismatch. The existing index has dimension "
                f"{index.d}, but {EMBED_MODEL!r} produces {embedding_dimension}. "
                "Use the exact embedding model that created the index."
            )

        chunk_tokens = [tokenize(_doc_text(doc)) for doc in chunks]
        bm25 = BM25Okapi(chunk_tokens)

        return PipelineResources(
            chunks=chunks,
            index=index,
            embedder=embedder,
            bm25=bm25,
            reranker=CrossEncoder(RERANK_MODEL),
            llm=_new_llm(),
            chunk_tokens=chunk_tokens,
        )


def clear_pipeline_cache() -> None:
    load_pipeline.cache_clear()
    _rewrite_query_cached.cache_clear()


def _normalise_filter_value(value: Any) -> set[str]:
    if value in (None, "", "All", "الكل"):
        return set()
    values: Iterable[Any]
    if isinstance(value, (list, tuple, set, frozenset)):
        values = value
    else:
        values = (value,)
    return {normalise_text(item) for item in values if str(item).strip()}


def allowed(doc: Any, filters: Mapping[str, Any] | None) -> bool:
    if not filters:
        return True

    metadata = _doc_metadata(doc)
    for key, expected in filters.items():
        expected_values = _normalise_filter_value(expected)
        if not expected_values:
            continue

        actual = metadata.get(key)
        if isinstance(actual, (list, tuple, set, frozenset)):
            actual_values = {normalise_text(item) for item in actual}
        else:
            actual_values = {normalise_text(actual)}

        if not expected_values.intersection(actual_values):
            return False

    return True


def rrf(*rankings: Sequence[int], k: int = RRF_K) -> list[int]:
    scores: dict[int, float] = {}
    for ranking in rankings:
        seen_in_ranking: set[int] = set()
        for rank, raw_doc_id in enumerate(ranking, start=1):
            doc_id = int(raw_doc_id)
            if doc_id < 0 or doc_id in seen_in_ranking:
                continue
            seen_in_ranking.add(doc_id)
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores, key=lambda doc_id: (-scores[doc_id], doc_id))


def _extract_json_object(raw_text: str) -> dict[str, Any]:
    cleaned = _MARKDOWN_FENCE_RE.sub("", str(raw_text or "").strip())
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start < 0 or end < start:
        raise ValueError("The model response did not contain a JSON object.")
    return json.loads(cleaned[start : end + 1])


def _invoke_text(llm: ChatGroq, prompt: str) -> str:
    response = llm.invoke(prompt)
    content = getattr(response, "content", response)
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, Mapping):
                parts.append(str(item.get("text", "")))
            else:
                parts.append(str(item))
        return "\n".join(part for part in parts if part).strip()
    return str(content or "").strip()


@lru_cache(maxsize=256)
def _rewrite_query_cached(question: str, profile: str) -> QueryPlan:
    language = detect_language(question)
    resources = load_pipeline()
    prompt = f"""
You rewrite search queries for a pediatric nutrition RAG system.
Return one valid JSON object only, with no Markdown:
{{
  "in_scope": true,
  "arabic_query": "...",
  "english_query": "...",
  "broad_query": "..."
}}

Requirements:
- Do not answer the question.
- Preserve age, food, nutrient, allergy, condition, symptom, school-meal and growth terms.
- Translate meaning rather than words.
- Expand common parent phrasing into clinical nutrition terminology without adding a diagnosis.
- The broad query must improve recall and remain within pediatric nutrition.
- Set in_scope=false only when the request is clearly unrelated to child nutrition, feeding,
  growth, food allergies, nutrients, school meals or diet-related child health.

Question:
{question}

Child profile, when relevant:
{profile[:MAX_PROFILE_CHARS]}
""".strip()

    try:
        payload = _extract_json_object(_invoke_text(resources.llm, prompt))
        candidates = [
            question,
            str(payload.get("arabic_query", "")),
            str(payload.get("english_query", "")),
            str(payload.get("broad_query", "")),
        ]
        in_scope_hint = bool(payload.get("in_scope", True))
    except Exception:
        LOGGER.exception("Query rewriting failed; using multilingual fallback queries.")
        candidates = [question, normalise_text(question)]
        in_scope_hint = True

    deduplicated: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        candidate = _WHITESPACE_RE.sub(" ", str(candidate or "")).strip()
        fingerprint = normalise_text(candidate)
        if candidate and fingerprint not in seen:
            deduplicated.append(candidate)
            seen.add(fingerprint)

    return QueryPlan(
        language=language,
        queries=tuple(deduplicated[:4] or [question]),
        in_scope_hint=in_scope_hint,
    )


def rewrite_query(question: str, profile: str = "") -> QueryPlan:
    clean_question = _WHITESPACE_RE.sub(" ", str(question or "")).strip()
    if not clean_question:
        raise ValueError("Question cannot be empty.")
    clean_profile = _WHITESPACE_RE.sub(" ", str(profile or "")).strip()[:MAX_PROFILE_CHARS]
    return _rewrite_query_cached(clean_question, clean_profile)


def _dense_ranking(resources: PipelineResources, query: str) -> list[int]:
    vector = resources.embedder.encode(
        [query],
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    vector = np.asarray(vector, dtype="float32")

    fetch_k = min(DENSE_FETCH_K, len(resources.chunks))
    _, dense_ids = resources.index.search(vector, fetch_k)
    return [
        int(doc_id)
        for doc_id in dense_ids[0].tolist()
        if 0 <= int(doc_id) < len(resources.chunks)
    ]


def _bm25_ranking(resources: PipelineResources, query: str) -> list[int]:
    scores = np.asarray(resources.bm25.get_scores(tokenize(query)), dtype=float)
    fetch_k = min(BM25_FETCH_K, len(resources.chunks))
    if fetch_k <= 0:
        return []
    if fetch_k == len(scores):
        ordered = np.argsort(scores)[::-1]
    else:
        selected = np.argpartition(scores, -fetch_k)[-fetch_k:]
        ordered = selected[np.argsort(scores[selected])[::-1]]
    return [int(doc_id) for doc_id in ordered.tolist()]


def _doc_fingerprint(doc: Any) -> str:
    text = normalise_text(_doc_text(doc))
    metadata = _doc_metadata(doc)
    source_hint = _safe_metadata_value(
        metadata,
        "document_id",
        "source_id",
        "source",
        "title",
    )
    page_hint = _safe_metadata_value(metadata, "page", "page_number", "page_label")
    payload = f"{source_hint}|{page_hint}|{text[:1800]}".encode("utf-8", errors="ignore")
    return hashlib.sha256(payload).hexdigest()


def _deduplicate_ranked(
    ranked_items: Sequence[tuple[int, float]],
    chunks: Sequence[Any],
    top_k: int,
) -> list[tuple[int, float]]:
    output: list[tuple[int, float]] = []
    seen: set[str] = set()
    for doc_id, score in ranked_items:
        fingerprint = _doc_fingerprint(chunks[doc_id])
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        output.append((doc_id, float(score)))
        if len(output) >= top_k:
            break
    return output


def _safe_log(payload: Mapping[str, Any]) -> None:
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    serialisable = dict(payload)
    with _LOG_LOCK:
        with LOG_FILE.open("a", encoding="utf-8") as file:
            file.write(json.dumps(serialisable, ensure_ascii=False, default=str) + "\n")


def _chunk_identifier(doc: Any, fallback_id: int) -> str:
    metadata = _doc_metadata(doc)
    return _safe_metadata_value(
        metadata,
        "chunk_id",
        "id",
        "document_id",
    ) or str(fallback_id)


def retrieve(
    question: str,
    filters: Mapping[str, Any] | None = None,
    top_k: int = DEFAULT_TOP_K,
    profile: str = "",
) -> tuple[list[Any], list[float], QueryPlan]:
    started = time.perf_counter()
    resources = load_pipeline()
    plan = rewrite_query(question, profile)

    rankings: list[list[int]] = []
    for query_variant in plan.queries:
        rankings.append(_dense_ranking(resources, query_variant))
        rankings.append(_bm25_ranking(resources, query_variant))

    fused_ids = [
        doc_id
        for doc_id in rrf(*rankings)
        if 0 <= doc_id < len(resources.chunks)
    ]

    filters_relaxed = False
    filtered_ids = [
        doc_id
        for doc_id in fused_ids
        if allowed(resources.chunks[doc_id], filters)
    ]

    # Metadata remains the first-choice constraint. It is relaxed only when it
    # eliminates every candidate, so wording differences do not cause a false
    # "unavailable" response.
    if filters and not filtered_ids:
        filters_relaxed = True
        filtered_ids = fused_ids

    candidate_ids = filtered_ids[:RERANK_CANDIDATE_K]
    if not candidate_ids:
        candidate_ids = list(range(min(RERANK_CANDIDATE_K, len(resources.chunks))))
        filters_relaxed = bool(filters)

    rerank_query = "\n".join(
        [
            str(question).strip(),
            *plan.queries,
            str(profile or "").strip()[:MAX_PROFILE_CHARS],
        ]
    ).strip()

    pairs = [
        (rerank_query, _doc_text(resources.chunks[doc_id])[:MAX_EVIDENCE_CHARS])
        for doc_id in candidate_ids
    ]
    raw_scores = resources.reranker.predict(
        pairs,
        batch_size=min(32, max(1, len(pairs))),
        show_progress_bar=False,
    )
    scores_array = np.asarray(raw_scores, dtype=float).reshape(-1)

    ranked_items = sorted(
        zip(candidate_ids, scores_array.tolist()),
        key=lambda item: (-float(item[1]), int(item[0])),
    )
    ranked_items = _deduplicate_ranked(
        ranked_items,
        resources.chunks,
        max(1, int(top_k)),
    )

    docs = [resources.chunks[doc_id] for doc_id, _ in ranked_items]
    scores = [float(score) for _, score in ranked_items]

    question_hash = hashlib.sha256(
        str(question).encode("utf-8", errors="ignore")
    ).hexdigest()[:16]
    log_payload: dict[str, Any] = {
        "event": "retrieval",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "question_hash": question_hash,
        "language": plan.language,
        "query_variants": list(plan.queries),
        "filters": dict(filters or {}),
        "filters_relaxed": filters_relaxed,
        "candidate_count": len(candidate_ids),
        "returned_count": len(docs),
        "chunk_ids": [
            _chunk_identifier(doc, fallback_id=index)
            for index, doc in enumerate(docs)
        ],
        "rerank_scores": [round(score, 6) for score in scores],
        "latency_ms": round((time.perf_counter() - started) * 1000, 2),
    }
    if LOG_RAW_QUERIES:
        log_payload["question"] = str(question)
    _safe_log(log_payload)

    return docs, scores, plan


def _flatten_text_value(value: Any, depth: int = 0) -> str:
    if value is None or depth > 3:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, Mapping):
        for preferred_key in ("text", "value", "content", "answer"):
            if preferred_key in value:
                return _flatten_text_value(value.get(preferred_key), depth + 1)
        parts = [
            _flatten_text_value(item, depth + 1)
            for item in value.values()
        ]
        return "؛ ".join(part for part in parts if part)
    if isinstance(value, (list, tuple, set, frozenset)):
        parts = [_flatten_text_value(item, depth + 1) for item in value]
        return "، ".join(part for part in parts if part)
    return str(value)


def _clean_plain_text(value: Any, max_length: int = 1000) -> str:
    text = html.unescape(_flatten_text_value(value))
    text = text.replace("\\n", " ").replace("\\t", " ")
    text = re.sub(r"<[^>]+>", " ", text)
    text = _MARKDOWN_FENCE_RE.sub("", text)
    text = _SOURCE_HEADING_RE.sub("", text)
    text = re.sub(r"^\s*[#>*\-]+\s*", "", text, flags=re.MULTILINE)
    text = _CITATION_IN_TEXT_RE.sub("", text)
    text = _WHITESPACE_RE.sub(" ", text).strip(" \n\t,;|")
    return text[:max_length].strip()


def _valid_citations(value: Any, maximum: int) -> tuple[int, ...]:
    if isinstance(value, str):
        candidates = re.findall(r"\d+", value)
    elif isinstance(value, (list, tuple, set)):
        candidates = value
    elif value is None:
        candidates = ()
    else:
        candidates = (value,)

    output: list[int] = []
    for candidate in candidates:
        try:
            number = int(candidate)
        except (TypeError, ValueError):
            continue
        if 1 <= number <= maximum and number not in output:
            output.append(number)
    return tuple(output)


def _evidence_context(docs: Sequence[Any]) -> str:
    blocks: list[str] = []
    for index, doc in enumerate(docs, start=1):
        reference = _reference_for_doc(doc, index)
        heading_parts = [f"[{index}]", reference.title]
        if reference.section:
            heading_parts.append(reference.section)
        if reference.page:
            heading_parts.append(f"page {reference.page}")
        text = _doc_text(doc).strip()[:MAX_EVIDENCE_CHARS]
        blocks.append(" | ".join(heading_parts) + "\n" + text)
    return "\n\n".join(blocks)


def _generation_prompt(
    question: str,
    profile: str,
    evidence: str,
    language: str,
    retry_note: str = "",
) -> str:
    language_rule = (
        "Write all user-facing text in clear Arabic suitable for mothers."
        if language == "ar"
        else "Write all user-facing text in clear English suitable for parents."
    )
    return f"""
You are the grounded pediatric nutrition assistant for Elite School.
Use only the numbered evidence supplied below. Do not use outside medical knowledge.

Return exactly one valid JSON object and no Markdown, code fences, tables, source list,
metadata, filenames, retrieval scores or commentary. Use this schema:
{{
  "status": "ok",
  "language": "{language}",
  "title": "short title",
  "summary": [
    {{"text": "short paragraph", "citations": [1]}}
  ],
  "table": [
    {{
      "recommendation": "...",
      "suitable_foods": "...",
      "quantity_frequency": "...",
      "practical_notes": "...",
      "warnings": "...",
      "citations": [1, 2]
    }}
  ],
  "closing_note": {{"text": "...", "citations": [1]}},
  "medical_notice": "..."
}}

Rules:
- {language_rule}
- Answer the user's actual intent even when the wording differs from the references.
- Give a useful partial answer whenever the evidence supports one; do not claim that
  information is unavailable merely because terminology differs.
- Every factual paragraph and every table row must cite one or more evidence numbers.
- A citation may be used only when its evidence directly supports that claim.
- Preserve numbers, ages and units exactly as written in the evidence.
- Never invent a quantity, frequency, diagnosis, treatment, allergy rule or medical fact.
- If the evidence has no numeric quantity, say that the reference does not specify a
  numeric amount and provide only the qualitative guidance it supports.
- Keep paragraphs short and create 3 to 6 practical table rows when evidence permits.
- In suitable_foods, list only foods explicitly supported by the evidence.
- In warnings, state only evidence-supported cautions; otherwise use a neutral dash.
- Do not expose raw passages, JSON, Python dictionaries, metadata or filenames.
- Do not place [1] citation markers inside text fields; use the citations arrays.
- The profile may personalize wording but must not be treated as medical evidence.
- Do not diagnose. The medical notice must state that this is general guidance and does
  not replace a pediatrician or registered dietitian.
- For a clearly unrelated question, return status="out_of_scope", an empty table and a
  brief scope message in summary.
- Only when the evidence is genuinely insufficient after broad retrieval, return
  status="insufficient_evidence", explain the limitation briefly, and do not guess.

{retry_note}

Child profile:
{profile[:MAX_PROFILE_CHARS] or "Not provided"}

Question:
{question}

Numbered evidence:
{evidence}
""".strip()


def _coerce_cited_text(item: Any, maximum: int) -> CitedText | None:
    if isinstance(item, Mapping):
        text = _clean_plain_text(item.get("text", ""), max_length=1200)
        citations = _valid_citations(item.get("citations"), maximum)
    else:
        text = _clean_plain_text(item, max_length=1200)
        citations = ()
    if not text:
        return None
    return CitedText(text=text, citations=citations)


def _parse_nutrition_answer(
    payload: Mapping[str, Any],
    docs: Sequence[Any],
    plan: QueryPlan,
) -> NutritionAnswer:
    maximum = len(docs)
    raw_status = str(payload.get("status", "ok")).strip().lower()
    status = raw_status if raw_status in {
        "ok",
        "out_of_scope",
        "insufficient_evidence",
    } else "ok"
    language = str(payload.get("language", plan.language)).strip().lower()
    language = "ar" if language.startswith("ar") else "en"
    default_title = "إرشادات غذائية شخصية" if language == "ar" else "Personalized Nutrition Guidance"
    title = _clean_plain_text(payload.get("title"), max_length=180) or default_title

    raw_summary = payload.get("summary", [])
    if isinstance(raw_summary, (str, Mapping)):
        raw_summary = [raw_summary]
    summary_items: list[CitedText] = []
    for item in raw_summary if isinstance(raw_summary, Sequence) else []:
        cited_text = _coerce_cited_text(item, maximum)
        if cited_text:
            summary_items.append(cited_text)

    raw_rows = payload.get("table", payload.get("rows", []))
    if isinstance(raw_rows, Mapping):
        raw_rows = [raw_rows]
    rows: list[GuidanceRow] = []
    for item in raw_rows if isinstance(raw_rows, Sequence) else []:
        if not isinstance(item, Mapping):
            continue
        citations = _valid_citations(item.get("citations"), maximum)
        recommendation = _clean_plain_text(item.get("recommendation"), 700)
        if not recommendation:
            continue
        rows.append(
            GuidanceRow(
                recommendation=recommendation,
                suitable_foods=_clean_plain_text(item.get("suitable_foods"), 700) or "—",
                quantity_frequency=_clean_plain_text(
                    item.get("quantity_frequency"),
                    700,
                ) or "—",
                practical_notes=_clean_plain_text(item.get("practical_notes"), 800) or "—",
                warnings=_clean_plain_text(item.get("warnings"), 700) or "—",
                citations=citations,
            )
        )

    closing_note = _coerce_cited_text(payload.get("closing_note"), maximum)
    medical_notice = _clean_plain_text(payload.get("medical_notice"), 500)
    if not medical_notice:
        medical_notice = (
            "هذه إرشادات عامة ولا تغني عن تقييم طبيب الأطفال أو اختصاصي تغذية مسجل."
            if language == "ar"
            else "This is general guidance and does not replace a pediatrician or registered dietitian."
        )

    # Successful factual content must have at least one valid citation. Status messages
    # may be uncited because they contain no medical claim.
    if status == "ok":
        summary_items = [item for item in summary_items if item.citations]
        rows = [row for row in rows if row.citations]
        if closing_note and not closing_note.citations:
            closing_note = None

    references = tuple(_reference_for_doc(doc, index) for index, doc in enumerate(docs, 1))
    return NutritionAnswer(
        status=status,
        language=language,
        title=title,
        summary=tuple(summary_items),
        rows=tuple(rows),
        closing_note=closing_note,
        medical_notice=medical_notice,
        references=references,
        retrieval_query=" | ".join(plan.queries),
    )


def _answer_claims(answer_result: NutritionAnswer) -> list[dict[str, Any]]:
    claims: list[dict[str, Any]] = []
    for index, item in enumerate(answer_result.summary):
        claims.append(
            {
                "id": f"summary_{index}",
                "text": item.text,
                "citations": list(item.citations),
            }
        )
    for index, row in enumerate(answer_result.rows):
        claims.append(
            {
                "id": f"row_{index}",
                "text": " | ".join(
                    [
                        row.recommendation,
                        row.suitable_foods,
                        row.quantity_frequency,
                        row.practical_notes,
                        row.warnings,
                    ]
                ),
                "citations": list(row.citations),
            }
        )
    if answer_result.closing_note:
        claims.append(
            {
                "id": "closing",
                "text": answer_result.closing_note.text,
                "citations": list(answer_result.closing_note.citations),
            }
        )
    return claims


def _validate_grounding(
    draft: NutritionAnswer,
    docs: Sequence[Any],
    llm: ChatGroq,
) -> NutritionAnswer:
    if draft.status != "ok" or not ENABLE_GROUNDING_VALIDATION:
        return draft

    claims = _answer_claims(draft)
    if not claims:
        return draft

    evidence_by_number = {
        index: _doc_text(doc)[:MAX_EVIDENCE_CHARS]
        for index, doc in enumerate(docs, start=1)
    }
    validation_payload = []
    for claim in claims:
        cited_evidence = {
            str(number): evidence_by_number[number]
            for number in claim["citations"]
            if number in evidence_by_number
        }
        validation_payload.append(
            {
                **claim,
                "cited_evidence": cited_evidence,
            }
        )

    prompt = f"""
Act as a strict citation validator.
For each claim, decide whether every meaningful factual statement is directly supported
by at least one of its cited evidence passages. Exact numeric claims must appear in the
evidence. Do not use outside knowledge.

Return one JSON object only:
{{"supported_ids": ["summary_0", "row_0"], "unsupported_ids": ["row_1"]}}

Claims:
{json.dumps(validation_payload, ensure_ascii=False)}
""".strip()

    try:
        payload = _extract_json_object(_invoke_text(llm, prompt))
        supported = {
            str(item)
            for item in payload.get("supported_ids", [])
            if isinstance(item, (str, int))
        }
    except Exception:
        LOGGER.exception("Grounding validation failed; retaining citation-range validation.")
        return draft

    summary = tuple(
        item
        for index, item in enumerate(draft.summary)
        if f"summary_{index}" in supported
    )
    rows = tuple(
        row
        for index, row in enumerate(draft.rows)
        if f"row_{index}" in supported
    )
    closing_note = (
        draft.closing_note
        if draft.closing_note and "closing" in supported
        else None
    )

    return NutritionAnswer(
        status=draft.status,
        language=draft.language,
        title=draft.title,
        summary=summary,
        rows=rows,
        closing_note=closing_note,
        medical_notice=draft.medical_notice,
        references=draft.references,
        retrieval_query=draft.retrieval_query,
    )


def _status_answer(
    status: str,
    language: str,
    docs: Sequence[Any] = (),
    plan: QueryPlan | None = None,
) -> NutritionAnswer:
    if language == "ar":
        if status == "out_of_scope":
            message = "يمكنني المساعدة في أسئلة تغذية الأطفال، النمو، الحساسية الغذائية والوجبات المدرسية."
        else:
            message = (
                "لم أجد بعد التوسيع وإعادة الصياغة دليلًا كافيًا في المراجع المفهرسة "
                "لإعطاء توصية آمنة، لذلك لن أخمّن."
            )
        notice = "هذه إرشادات عامة ولا تغني عن تقييم طبيب الأطفال أو اختصاصي تغذية مسجل."
        title = "إرشادات غذائية شخصية"
    else:
        if status == "out_of_scope":
            message = (
                "I can help with pediatric nutrition, growth, food allergies "
                "and school-meal questions."
            )
        else:
            message = (
                "After query expansion and broader retrieval, the indexed references "
                "still did not provide enough evidence for a safe recommendation, "
                "so I will not guess."
            )
        notice = (
            "This is general guidance and does not replace a pediatrician "
            "or registered dietitian."
        )
        title = "Personalized Nutrition Guidance"

    references = tuple(_reference_for_doc(doc, index) for index, doc in enumerate(docs, 1))
    return NutritionAnswer(
        status=status,
        language=language,
        title=title,
        summary=(CitedText(message, ()),),
        medical_notice=notice,
        references=references,
        retrieval_query=" | ".join(plan.queries) if plan else "",
    )


def answer(
    question: str,
    profile: str = "",
    filters: Mapping[str, Any] | None = None,
) -> NutritionAnswer:
    clean_question = _WHITESPACE_RE.sub(" ", str(question or "")).strip()
    if not clean_question:
        raise ValueError("Question cannot be empty.")

    started = time.perf_counter()
    docs, scores, plan = retrieve(
        question=clean_question,
        filters=filters,
        profile=profile,
    )

    if not docs:
        result = _status_answer("insufficient_evidence", plan.language, plan=plan)
        _safe_log(
            {
                "event": "generation",
                "status": result.status,
                "latency_ms": round((time.perf_counter() - started) * 1000, 2),
                "citation_count": 0,
            }
        )
        return result

    evidence = _evidence_context(docs)
    resources = load_pipeline()
    last_error: Exception | None = None
    draft: NutritionAnswer | None = None

    for attempt in range(2):
        retry_note = ""
        if attempt:
            retry_note = (
                "Previous output was invalid or had no citation-supported content. "
                "Return valid JSON and attach valid evidence citations to every factual item."
            )
        try:
            raw = _invoke_text(
                resources.llm,
                _generation_prompt(
                    question=clean_question,
                    profile=str(profile or ""),
                    evidence=evidence,
                    language=plan.language,
                    retry_note=retry_note,
                ),
            )
            payload = _extract_json_object(raw)
            draft = _parse_nutrition_answer(payload, docs, plan)
            if draft.status != "ok" or draft.summary or draft.rows:
                break
            raise ValueError("The generated answer contained no citation-supported content.")
        except Exception as error:
            last_error = error
            LOGGER.exception("Answer generation attempt %s failed.", attempt + 1)

    if draft is None:
        if last_error:
            LOGGER.error("Answer generation failed after retries: %s", last_error)
        result = _status_answer(
            "insufficient_evidence",
            plan.language,
            docs=docs,
            plan=plan,
        )
    elif draft.status == "out_of_scope":
        result = _status_answer(
            "out_of_scope",
            draft.language,
            docs=(),
            plan=plan,
        )
    elif draft.status == "insufficient_evidence":
        result = _status_answer(
            "insufficient_evidence",
            draft.language,
            docs=docs,
            plan=plan,
        )
    else:
        result = _validate_grounding(draft, docs, resources.llm)
        if not result.summary and not result.rows:
            # A strict validator removed all claims. Regenerate once with the same evidence
            # rather than exposing unsupported text.
            try:
                raw = _invoke_text(
                    resources.llm,
                    _generation_prompt(
                        question=clean_question,
                        profile=str(profile or ""),
                        evidence=evidence,
                        language=plan.language,
                        retry_note=(
                            "A citation validator rejected the previous claims. "
                            "Use fewer, directly supported claims and exact citations."
                        ),
                    ),
                )
                regenerated = _parse_nutrition_answer(
                    _extract_json_object(raw),
                    docs,
                    plan,
                )
                result = _validate_grounding(regenerated, docs, resources.llm)
            except Exception:
                LOGGER.exception("Grounded regeneration failed.")

        if result.status == "ok" and not result.summary and not result.rows:
            result = _status_answer(
                "insufficient_evidence",
                plan.language,
                docs=docs,
                plan=plan,
            )

    _safe_log(
        {
            "event": "generation",
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "status": result.status,
            "language": result.language,
            "retrieved_count": len(docs),
            "top_rerank_score": round(scores[0], 6) if scores else None,
            "answer_row_count": len(result.rows),
            "summary_count": len(result.summary),
            "citation_count": len(
                {
                    citation
                    for item in result.summary
                    for citation in item.citations
                }
                | {
                    citation
                    for row in result.rows
                    for citation in row.citations
                }
            ),
            "latency_ms": round((time.perf_counter() - started) * 1000, 2),
        }
    )
    return result


def _coerce_source_document(item: Any, index: int) -> Document:
    if isinstance(item, Document):
        return Document(
            page_content=str(item.page_content or ""),
            metadata=dict(item.metadata or {}),
        )
    if isinstance(item, Mapping):
        text = str(
            item.get("page_content")
            or item.get("content")
            or item.get("text")
            or ""
        )
        metadata = item.get("metadata", {})
        metadata_dict = dict(metadata) if isinstance(metadata, Mapping) else {}
        for key, value in item.items():
            if key not in {"page_content", "content", "text", "metadata"}:
                metadata_dict.setdefault(str(key), value)
        return Document(page_content=text, metadata=metadata_dict)
    return Document(
        page_content=str(item or ""),
        metadata={"source_document_index": index},
    )


def build_hnsw_index(
    documents: Sequence[Document | Mapping[str, Any] | str],
    *,
    index_dir: str | Path | None = None,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    force: bool = False,
) -> dict[str, Any]:
    """
    Build the doctor's required indexing path:
    RecursiveCharacterTextSplitter with overlap -> the same multilingual embedding model
    used at query time -> FAISS HNSW.

    The function is optional at runtime because the application loads the current
    rag_index/chunks.pkl and rag_index/hnsw.faiss without rebuilding them.
    """
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be >= 0 and smaller than chunk_size.")
    if not documents:
        raise ValueError("At least one source document is required.")

    target_dir = (
        Path(index_dir).expanduser()
        if index_dir is not None
        else INDEX_DIR
    )
    if not target_dir.is_absolute():
        target_dir = PROJECT_DIR / target_dir

    target_chunks = target_dir / "chunks.pkl"
    target_faiss = target_dir / "hnsw.faiss"
    target_manifest = target_dir / "manifest.json"

    if not force and (target_chunks.exists() or target_faiss.exists()):
        raise FileExistsError(
            f"{target_dir} already contains an index. Pass force=True to rebuild it."
        )

    source_documents = [
        _coerce_source_document(item, index)
        for index, item in enumerate(documents)
        if _doc_text(item).strip()
    ]
    if not source_documents:
        raise ValueError("All supplied source documents are empty.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=int(chunk_size),
        chunk_overlap=int(chunk_overlap),
        length_function=len,
        is_separator_regex=False,
        separators=["\n\n", "\n", "؟ ", ". ", "! ", "; ", "؛ ", " ", ""],
    )
    chunks = splitter.split_documents(source_documents)
    if not chunks:
        raise ValueError("The text splitter produced no chunks.")

    prepared_chunks: list[Document] = []
    for index, chunk in enumerate(chunks):
        text = _WHITESPACE_RE.sub(" ", str(chunk.page_content or "")).strip()
        if not text:
            continue
        metadata = dict(chunk.metadata or {})
        metadata["chunk_id"] = str(metadata.get("chunk_id") or f"chunk_{index:07d}")
        metadata["embedding_model"] = EMBED_MODEL
        prepared_chunks.append(Document(page_content=text, metadata=metadata))

    if not prepared_chunks:
        raise ValueError("All produced chunks were empty after cleaning.")

    embedder = SentenceTransformer(EMBED_MODEL)
    embeddings = embedder.encode(
        [chunk.page_content for chunk in prepared_chunks],
        batch_size=32,
        normalize_embeddings=True,
        show_progress_bar=True,
        convert_to_numpy=True,
    )
    embeddings = np.asarray(embeddings, dtype="float32")
    dimension = int(embeddings.shape[1])

    try:
        index = faiss.IndexHNSWFlat(dimension, HNSW_M, faiss.METRIC_INNER_PRODUCT)
        metric = "inner_product"
    except TypeError:
        index = faiss.IndexHNSWFlat(dimension, HNSW_M)
        metric = "l2_on_normalized_vectors"

    index.hnsw.efConstruction = HNSW_EF_CONSTRUCTION
    index.hnsw.efSearch = HNSW_EF_SEARCH
    index.add(embeddings)

    target_dir.mkdir(parents=True, exist_ok=True)
    temp_chunks = target_dir / "chunks.pkl.tmp"
    temp_faiss = target_dir / "hnsw.faiss.tmp"
    temp_manifest = target_dir / "manifest.json.tmp"

    with temp_chunks.open("wb") as file:
        pickle.dump(prepared_chunks, file, protocol=pickle.HIGHEST_PROTOCOL)
    faiss.write_index(index, str(temp_faiss))

    manifest = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "embedding_model": EMBED_MODEL,
        "embedding_dimension": dimension,
        "index_type": "HNSW",
        "metric": metric,
        "hnsw_m": HNSW_M,
        "hnsw_ef_construction": HNSW_EF_CONSTRUCTION,
        "hnsw_ef_search": HNSW_EF_SEARCH,
        "chunk_size": int(chunk_size),
        "chunk_overlap": int(chunk_overlap),
        "source_document_count": len(source_documents),
        "chunk_count": len(prepared_chunks),
    }
    temp_manifest.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    temp_chunks.replace(target_chunks)
    temp_faiss.replace(target_faiss)
    temp_manifest.replace(target_manifest)

    if target_dir.resolve() == INDEX_DIR.resolve():
        clear_pipeline_cache()

    return manifest