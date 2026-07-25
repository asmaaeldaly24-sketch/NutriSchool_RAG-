from __future__ import annotations

import json
import logging
import os
import re
import time
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, Sequence

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq

from prompting import SYSTEM_PROMPT, build_generation_prompt
from retrieve_context import RetrievedChunk, build_context, retrieve_context


LOGGER = logging.getLogger("nutrischool.rag_pipeline")
PROJECT_ROOT = Path(__file__).resolve().parent
load_dotenv(PROJECT_ROOT / ".env")

LLM_MODEL = os.getenv("RAG_LLM_MODEL", "llama-3.3-70b-versatile")
MAX_PROFILE_CHARACTERS = max(300, int(os.getenv("RAG_MAX_PROFILE_CHARACTERS", "2200")))
MAX_QUESTION_CHARACTERS = max(200, int(os.getenv("RAG_MAX_QUESTION_CHARACTERS", "1800")))
LOG_PATH = PROJECT_ROOT / "logs" / "rag_events.jsonl"
ARABIC_RE = re.compile(r"[\u0600-\u06FF]")
JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)
AGE_RE = re.compile(r"\bAge:\s*(\d{1,2})\s*years?\b", re.IGNORECASE)


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


@lru_cache(maxsize=1)
def load_llm() -> ChatGroq:
    api_key = os.getenv("GROQ_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is missing from the .env file.")
    return ChatGroq(
        api_key=api_key,
        model=LLM_MODEL,
        temperature=0.0,
        max_tokens=1000,
        max_retries=1,
        timeout=90,
    )


def detect_language(text: str) -> str:
    arabic = len(ARABIC_RE.findall(text or ""))
    latin = len(re.findall(r"[A-Za-z]", text or ""))
    return "ar" if arabic >= latin else "en"


def clean_text(value: object, maximum: int = 1200) -> str:
    text = " ".join(str(value or "").split()).strip()
    return text[:maximum]


def safe_citations(value: object, source_count: int) -> tuple[int, ...]:
    if not isinstance(value, (list, tuple, set)):
        return ()
    output: list[int] = []
    for item in value:
        try:
            number = int(item)
        except (TypeError, ValueError):
            continue
        if 1 <= number <= source_count and number not in output:
            output.append(number)
    return tuple(output)


def extract_json_object(text: str) -> dict[str, Any]:
    clean = str(text or "").strip().replace("```json", "").replace("```", "")
    try:
        payload = json.loads(clean)
    except json.JSONDecodeError:
        match = JSON_BLOCK_RE.search(clean)
        if not match:
            raise ValueError("The language model did not return a JSON object.")
        payload = json.loads(match.group(0))
    if not isinstance(payload, dict):
        raise ValueError("The language model JSON response must be an object.")
    return payload


def reference_for_chunk(chunk: RetrievedChunk, number: int) -> CitationReference:
    return CitationReference(
        number=number,
        title=clean_text(chunk.book_title, 100),
        page=str(chunk.page_number),
        section=clean_text(chunk.hierarchy_path or chunk.chapter_title, 180),
    )


def parse_answer(
    payload: Mapping[str, Any],
    sources: Sequence[RetrievedChunk],
    fallback_language: str,
    retrieval_query: str,
) -> NutritionAnswer:
    source_count = len(sources)
    status = str(payload.get("status", "ok")).strip().lower()
    if status not in {"ok", "insufficient_evidence", "out_of_scope"}:
        status = "insufficient_evidence"
    language = str(payload.get("language", fallback_language)).strip().lower()
    if language not in {"ar", "en"}:
        language = fallback_language

    summary: list[CitedText] = []
    for item in payload.get("summary", []) if isinstance(payload.get("summary"), list) else []:
        if not isinstance(item, Mapping):
            continue
        text = clean_text(item.get("text"), 900)
        citations = safe_citations(item.get("citations"), source_count)
        if text and citations:
            summary.append(CitedText(text=text, citations=citations))

    rows: list[GuidanceRow] = []
    for item in payload.get("rows", []) if isinstance(payload.get("rows"), list) else []:
        if not isinstance(item, Mapping):
            continue
        recommendation = clean_text(item.get("recommendation"), 600)
        citations = safe_citations(item.get("citations"), source_count)
        if not recommendation or not citations:
            continue
        rows.append(
            GuidanceRow(
                recommendation=recommendation,
                suitable_foods=clean_text(item.get("suitable_foods"), 500),
                quantity_frequency=clean_text(item.get("quantity_frequency"), 350),
                practical_notes=clean_text(item.get("practical_notes"), 500),
                warnings=clean_text(item.get("warnings"), 500),
                citations=citations,
            )
        )

    closing_payload = payload.get("closing_note")
    closing_note: CitedText | None = None
    if isinstance(closing_payload, Mapping):
        closing_text = clean_text(closing_payload.get("text"), 600)
        closing_citations = safe_citations(closing_payload.get("citations"), source_count)
        if closing_text and closing_citations:
            closing_note = CitedText(closing_text, closing_citations)

    if status == "ok" and not summary and not rows:
        status = "insufficient_evidence"

    return NutritionAnswer(
        status=status,
        language=language,
        title=clean_text(payload.get("title"), 160) or "School Nutrition Guidance",
        summary=tuple(summary[:6]),
        rows=tuple(rows[:8]),
        closing_note=closing_note,
        medical_notice=clean_text(payload.get("medical_notice"), 600)
        or "This educational guidance does not replace individualized care from a pediatrician or registered dietitian.",
        references=tuple(reference_for_chunk(chunk, index) for index, chunk in enumerate(sources, start=1)),
        retrieval_query=retrieval_query,
    )


def status_answer(status: str, language: str, retrieval_query: str = "") -> NutritionAnswer:
    message = (
        "The indexed school-nutrition references did not provide enough evidence for a safe answer."
        if status == "insufficient_evidence"
        else "This assistant is limited to nutrition questions for school-age children and adolescents."
    )
    return NutritionAnswer(
        status=status,
        language=language,
        title="School Nutrition Guidance",
        summary=(CitedText(message, ()),),
        medical_notice="This educational tool does not replace a pediatrician or registered dietitian.",
        retrieval_query=retrieval_query,
    )


def profile_age(profile: str) -> int | None:
    match = AGE_RE.search(profile or "")
    return int(match.group(1)) if match else None


def safe_log(payload: Mapping[str, Any]) -> None:
    try:
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with LOG_PATH.open("a", encoding="utf-8") as file:
            file.write(json.dumps(dict(payload), ensure_ascii=False, default=str) + "\n")
    except OSError:
        pass


def answer(
    question: str,
    profile: str = "",
    filters: Mapping[str, Any] | None = None,
) -> NutritionAnswer:
    del filters
    clean_question = clean_text(question, MAX_QUESTION_CHARACTERS)
    if not clean_question:
        raise ValueError("Question cannot be empty.")

    language = detect_language(clean_question)
    safe_profile = str(profile or "").strip()[:MAX_PROFILE_CHARACTERS]
    age = profile_age(safe_profile)
    if age is not None and age < 5:
        return status_answer("out_of_scope", language)

    started = time.perf_counter()
    retrieval = retrieve_context(clean_question, top_k=35, final_k=5)
    sources: list[RetrievedChunk] = list(retrieval.get("chunks", []))
    retrieval_query = " | ".join(retrieval.get("queries", []))

    if not sources:
        return status_answer("insufficient_evidence", language, retrieval_query)

    evidence = build_context(sources, maximum_characters=8000)
    prompt = build_generation_prompt(clean_question, safe_profile, evidence)
    last_error: Exception | None = None

    for attempt in range(2):
        try:
            response = load_llm().invoke(
                [
                    SystemMessage(content=SYSTEM_PROMPT),
                    HumanMessage(
                        content=prompt
                        + ("\nReturn fewer claims with direct citations." if attempt else "")
                    ),
                ]
            )
            result = parse_answer(
                extract_json_object(str(response.content or "")),
                sources=sources,
                fallback_language=language,
                retrieval_query=retrieval_query,
            )
            if result.status != "ok" or result.summary or result.rows:
                safe_log(
                    {
                        "event": "answer",
                        "status": result.status,
                        "language": result.language,
                        "source_count": len(sources),
                        "latency_ms": round((time.perf_counter() - started) * 1000, 2),
                    }
                )
                return result
        except Exception as error:
            last_error = error
            LOGGER.warning("Answer generation attempt %s failed: %s", attempt + 1, error)

    if last_error:
        LOGGER.error("Answer generation failed after retries: %s", last_error)
    return status_answer("insufficient_evidence", language, retrieval_query)