from __future__ import annotations

import os

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq


load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()

QUERY_REWRITER_MODEL = os.getenv(
    "QUERY_REWRITER_MODEL",
    "llama-3.3-70b-versatile",
)


SYSTEM_PROMPT = """
You rewrite pediatric-nutrition questions for retrieval over English medical books.

Rules:
- Return one concise English search query only.
- Preserve the user's exact population and age group.
- Do not introduce prematurity, disease, supplements, or treatment unless the
  user explicitly mentioned them.
- Expand useful pediatric nutrition terminology.
- Do not answer the question.
- Do not add explanations, labels, quotes, or bullet points.
""".strip()


def get_query_rewriter() -> ChatGroq:
    if not GROQ_API_KEY:
        raise RuntimeError(
            "GROQ_API_KEY is missing from the .env file."
        )

    return ChatGroq(
        api_key=GROQ_API_KEY,
        model=QUERY_REWRITER_MODEL,
        temperature=0.0,
    )


def rewrite_query(question: str) -> str:
    clean_question = " ".join(
        str(question or "").split()
    ).strip()

    if not clean_question:
        raise ValueError(
            "Question cannot be empty."
        )

    response = get_query_rewriter().invoke(
        [
            SystemMessage(
                content=SYSTEM_PROMPT
            ),
            HumanMessage(
                content=clean_question
            ),
        ]
    )

    rewritten = " ".join(
        str(response.content or "").split()
    ).strip()

    if not rewritten:
        return clean_question

    return rewritten


def main() -> None:
    question = (
        "ما أهم الاحتياجات الغذائية "
        "للطفل أثناء النمو؟"
    )

    rewritten = rewrite_query(question)

    print("\nQUERY REWRITING REPORT")
    print("=" * 60)
    print(f"Original: {question}")
    print(f"Rewritten: {rewritten}")


if __name__ == "__main__":
    main()