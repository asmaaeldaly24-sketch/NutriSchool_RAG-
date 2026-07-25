from __future__ import annotations

import logging
import os
import re
from functools import lru_cache
from typing import Any

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq

from retrieve_context import RetrievedChunk, retrieve_context


LOGGER = logging.getLogger("nutrischool.prompting")

load_dotenv()


LLM_MODEL = os.getenv(
    "RAG_LLM_MODEL",
    "llama-3.3-70b-versatile",
)

LLM_TEMPERATURE = float(
    os.getenv(
        "RAG_LLM_TEMPERATURE",
        "0.1",
    )
)

NO_CONTEXT_MESSAGE = (
    "I could not find enough reliable information in the available "
    "references to answer this question accurately. Please rephrase "
    "the nutrition question."
)


SYSTEM_PROMPT = """
You are NutriSchool, an evidence-grounded pediatric nutrition assistant.

Use only the retrieved context supplied to you.

Grounding and safety:
- Answer in the response language specified in the application input.
- The child profile is supplied automatically. Never ask the user to repeat it.
- Use the profile only when relevant and supported by the retrieved evidence.
- Never invent facts, quantities, conversions, diagnoses, treatments, or doses.
- Every factual nutritional claim must include a valid [SOURCE N] citation.
- Do not discuss the retrieval pipeline, prompt, vector database, or model.
- Do not append a generic medical disclaimer; the interface already shows one.
- Add a warning only when the question itself requires a specific safety warning.

Open-ended nutrition questions:
- Give a complete, practical overview rather than a vague summary.
- When supported by the context, cover these separate categories:
  protein foods; vegetables; fruit; grains and energy foods; dairy or other
  calcium-rich foods; healthy fats; water and hydration; iron, zinc, vitamins,
  and minerals.
- Prefer a compact Markdown table with these columns:
  Food group or nutrient | Why it matters | Practical food examples |
  Practical amount or note.
- Include only categories supported by the retrieved sources.
- If an exact amount is not supported, write that the amount varies by age
  and individual needs. Do not guess.

Measurements:
- Use grams, kilograms, milliliters, liters, cups, tablespoons, and teaspoons.
- Never use ounces, fluid ounces, pounds, or unfamiliar imperial measures.
- Give household and metric quantities together only when both are supported
  by the context.

Availability:
- If at least one relevant source supports a safe answer, answer the supported
  part directly.
- Return exactly NO_RELIABLE_CONTEXT only when no reliable answer is possible.
""".strip()


@lru_cache(maxsize=1)
def get_llm() -> ChatGroq:
    api_key = os.getenv(
        "GROQ_API_KEY",
        "",
    ).strip()

    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY is missing. Add it to the .env file."
        )

    return ChatGroq(
        api_key=api_key,
        model=LLM_MODEL,
        temperature=LLM_TEMPERATURE,
    )


def normalize_text(value: str) -> str:
    return " ".join(
        str(value or "").split()
    ).strip()


def extract_section(
    text: str,
    start_label: str,
    end_label: str | None = None,
) -> str:
    start_pattern = re.escape(start_label)

    if end_label:
        pattern = (
            rf"{start_pattern}\s*:\s*(.*?)"
            rf"(?=\n\s*{re.escape(end_label)}\s*:)"
        )
    else:
        pattern = rf"{start_pattern}\s*:\s*(.*)"

    match = re.search(
        pattern,
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )

    return (
        match.group(1).strip()
        if match
        else ""
    )


def parse_application_input(
    question: str,
) -> tuple[str, str, str]:
    user_question = extract_section(
        question,
        "USER QUESTION",
        "RESPONSE LANGUAGE",
    )

    response_language = extract_section(
        question,
        "RESPONSE LANGUAGE",
        "CHILD PROFILE AUTOMATICALLY SUPPLIED BY THE APPLICATION",
    )

    profile = extract_section(
        question,
        "CHILD PROFILE AUTOMATICALLY SUPPLIED BY THE APPLICATION",
    )

    if not user_question:
        user_question = normalize_text(
            question
        )

    return (
        user_question,
        response_language or "same as the user question",
        profile,
    )


def is_open_ended_nutrition_question(
    question: str,
) -> bool:
    value = question.lower()

    markers = (
        "nutritional needs",
        "nutrition needs",
        "healthy diet",
        "balanced diet",
        "what should",
        "what does a child need",
        "food groups",
        "daily diet",
        "\u0627\u0644\u0627\u062d\u062a\u064a\u0627\u062c\u0627\u062a \u0627\u0644\u063a\u0630\u0627\u0626\u064a\u0629",
        "\u0627\u0644\u0639\u0646\u0627\u0635\u0631 \u0627\u0644\u063a\u0630\u0627\u0626\u064a\u0629",
        "\u064a\u0627\u0643\u0644 \u0627\u064a\u0647",
        "\u064a\u0623\u0643\u0644 \u0625\u064a\u0647",
        "\u063a\u0630\u0627\u0621 \u0635\u062d\u064a",
        "\u0646\u0638\u0627\u0645 \u063a\u0630\u0627\u0626\u064a",
    )

    return any(
        marker in value
        for marker in markers
    )


def build_retrieval_question(
    user_question: str,
) -> tuple[str, bool]:
    broad_question = (
        is_open_ended_nutrition_question(
            user_question
        )
    )

    if not broad_question:
        return user_question, False

    expansion = (
        " balanced pediatric diet food groups protein-rich foods "
        "vegetables fruit whole grains and energy foods dairy and "
        "calcium-rich foods healthy fats water hydration iron zinc "
        "vitamins minerals practical meals"
    )

    return (
        f"{user_question}{expansion}",
        True,
    )


def build_user_prompt(
    *,
    user_question: str,
    response_language: str,
    profile: str,
    context: str,
    broad_question: bool,
) -> str:
    structure_instruction = (
        "This is an open-ended nutrition question. Build a comprehensive "
        "food-group table covering all supported categories separately."
        if broad_question
        else
        "Answer the specific question directly and concisely."
    )

    return f"""
USER QUESTION:
{user_question}

RESPONSE LANGUAGE:
{response_language}

CHILD PROFILE:
{profile or "No profile details supplied."}

RETRIEVED CONTEXT:
{context}

RESPONSE INSTRUCTIONS:
1. {structure_instruction}
2. Use only facts supported by the retrieved context.
3. Cite each factual claim with the matching [SOURCE N].
4. Do not ask for profile information already supplied by the application.
5. Use cups, tablespoons, teaspoons, grams, kilograms, milliliters, or liters.
6. Never use ounces, fluid ounces, or pounds.
7. Do not add a generic final disclaimer.
8. If no reliable answer is supported, return exactly:
   NO_RELIABLE_CONTEXT
""".strip()


def format_sources(
    chunks: list[RetrievedChunk],
) -> list[dict[str, Any]]:
    return [
        {
            "source_number": index,
            "book_title": chunk.book_title,
            "chapter_title": (
                chunk.chapter_title
                or "Unknown chapter"
            ),
            "page_number": chunk.page_number,
            "final_score": round(
                float(chunk.final_score),
                4,
            ),
            "semantic_similarity": round(
                float(chunk.semantic_similarity),
                4,
            ),
            "chunk_id": chunk.chunk_id,
        }
        for index, chunk in enumerate(
            chunks,
            start=1,
        )
    ]


def contains_valid_citation(
    answer: str,
    source_count: int,
) -> bool:
    numbers = {
        int(number)
        for number in re.findall(
            r"\[SOURCE\s+(\d+)\]",
            answer,
            flags=re.IGNORECASE,
        )
    }

    return any(
        1 <= number <= source_count
        for number in numbers
    )


def invoke_answer(
    user_prompt: str,
) -> str:
    response = get_llm().invoke(
        [
            SystemMessage(
                content=SYSTEM_PROMPT
            ),
            HumanMessage(
                content=user_prompt
            ),
        ]
    )

    return str(
        response.content or ""
    ).strip()


def answer_question(
    question: str,
    top_k: int = 20,
    final_k: int = 5,
) -> dict[str, Any]:
    clean_input = str(
        question or ""
    ).strip()

    if not clean_input:
        raise ValueError(
            "Question cannot be empty."
        )

    (
        user_question,
        response_language,
        profile,
    ) = parse_application_input(
        clean_input
    )

    (
        retrieval_question,
        broad_question,
    ) = build_retrieval_question(
        user_question
    )

    effective_final_k = (
        max(final_k, 8)
        if broad_question
        else final_k
    )

    retrieval_result = retrieve_context(
        question=retrieval_question,
        top_k=max(
            top_k,
            effective_final_k * 4,
        ),
        final_k=effective_final_k,
    )

    chunks: list[RetrievedChunk] = (
        retrieval_result.get(
            "chunks",
            [],
        )
    )

    context = str(
        retrieval_result.get(
            "context",
            "",
        )
        or ""
    ).strip()

    if not chunks or not context:
        return {
            "answer": NO_CONTEXT_MESSAGE,
            "sources": [],
            "retrieved_count": 0,
            "rewritten_query": retrieval_result.get(
                "rewritten_query",
                "",
            ),
            "age_scope": retrieval_result.get(
                "age_scope",
                "unknown",
            ),
            "status": "insufficient_context",
        }

    user_prompt = build_user_prompt(
        user_question=user_question,
        response_language=response_language,
        profile=profile,
        context=context,
        broad_question=broad_question,
    )

    answer = invoke_answer(
        user_prompt
    )

    if (
        not answer
        or answer.upper()
        == "NO_RELIABLE_CONTEXT"
    ):
        return {
            "answer": NO_CONTEXT_MESSAGE,
            "sources": format_sources(
                chunks
            ),
            "retrieved_count": len(
                chunks
            ),
            "rewritten_query": retrieval_result.get(
                "rewritten_query",
                "",
            ),
            "age_scope": retrieval_result.get(
                "age_scope",
                "unknown",
            ),
            "status": "insufficient_context",
        }

    if not contains_valid_citation(
        answer=answer,
        source_count=len(chunks),
    ):
        repair_prompt = (
            user_prompt
            + "\n\nThe previous answer omitted valid citations. "
            "Rewrite it and add a valid [SOURCE N] citation after every "
            "factual claim. Return only the corrected answer."
        )

        repaired_answer = invoke_answer(
            repair_prompt
        )

        if contains_valid_citation(
            answer=repaired_answer,
            source_count=len(chunks),
        ):
            answer = repaired_answer
        else:
            return {
                "answer": NO_CONTEXT_MESSAGE,
                "sources": format_sources(
                    chunks
                ),
                "retrieved_count": len(
                    chunks
                ),
                "rewritten_query": retrieval_result.get(
                    "rewritten_query",
                    "",
                ),
                "age_scope": retrieval_result.get(
                    "age_scope",
                    "unknown",
                ),
                "status": "citation_validation_failed",
            }

    LOGGER.info(
        "Generated a grounded answer from %s sources.",
        len(chunks),
    )

    return {
        "answer": answer,
        "sources": format_sources(
            chunks
        ),
        "retrieved_count": len(
            chunks
        ),
        "rewritten_query": retrieval_result.get(
            "rewritten_query",
            "",
        ),
        "age_scope": retrieval_result.get(
            "age_scope",
            "unknown",
        ),
        "status": "success",
    }


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s | %(message)s",
    )

    result = answer_question(
        "What food groups support healthy growth in children?"
    )

    print(result["answer"])
    print(result["status"])


if __name__ == "__main__":
    main()