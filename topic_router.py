from __future__ import annotations

from book_catalog import (
    TOPICS,
    detect_topic_names as base_detect_topic_names,
    normalize_for_match,
)


SPECIAL_TOPIC_PHRASES = {
    "school_meals": (
        "breakfast",
        "breakfast skipping",
        "school breakfast",
        "lunch box",
        "lunchbox",
        "school meal",
        "school meals",
        "\u0644\u0627\u0646\u0634 \u0628\u0648\u0643\u0633",
        "\u0648\u062c\u0628\u0627\u062a \u062e\u0641\u064a\u0641\u0629",
        "\u0648\u062c\u0628\u0629 \u062e\u0641\u064a\u0641\u0629",
        "\u0648\u062c\u0628\u0627\u062a \u0627\u0644\u0645\u062f\u0631\u0633\u0629",
        "\u062a\u062e\u0637\u064a \u0627\u0644\u0641\u0637\u0648\u0631",
    ),

    "bone": (
        "calcium",
        "vitamin d",
        "bone mass",
        "bone mineral",
        "\u0643\u0627\u0644\u0633\u064a\u0648\u0645",
        "\u0641\u064a\u062a\u0627\u0645\u064a\u0646 \u062f",
        "\u0643\u062a\u0644\u0629 \u0639\u0638\u0645\u064a\u0629",
        "\u0643\u062a\u0644\u0629 \u0627\u0644\u0639\u0638\u0627\u0645",
        "\u0628\u0646\u0627\u0621 \u0627\u0644\u0639\u0638\u0627\u0645",
    ),

    "sports": (
        "athlete",
        "athletes",
        "adolescent athlete",
        "adolescent athletes",
        "caffeine",
        "energy drink",
        "energy drinks",
        "\u0643\u0627\u0641\u064a\u064a\u0646",
        "\u0645\u0634\u0631\u0648\u0628\u0627\u062a \u0627\u0644\u0637\u0627\u0642\u0629",
        "\u0631\u064a\u0627\u0636\u064a",
        "\u0627\u0644\u0631\u064a\u0627\u0636\u064a",
    ),

    "assessment": (
        "dietary history",
        "diet history",
        "growth chart",
        "growth charts",
        "body mass index",
        "bmi",
        "anthropometry",
        "anthropometric",
        "nutritional assessment",
        "\u062a\u0627\u0631\u064a\u062e \u0627\u0644\u063a\u0630\u0627\u0621",
        "\u0645\u062e\u0637\u0637\u0627\u062a \u0627\u0644\u0646\u0645\u0648",
        "\u0645\u0624\u0634\u0631 \u0643\u062a\u0644\u0629 \u0627\u0644\u062c\u0633\u0645",
        "\u0627\u0644\u0642\u064a\u0627\u0633\u0627\u062a \u0627\u0644\u062c\u0633\u0645\u064a\u0629",
        "\u062a\u0642\u064a\u064a\u0645 \u0627\u0644\u062d\u0627\u0644\u0629 \u0627\u0644\u063a\u0630\u0627\u0626\u064a\u0629",
    ),
}


ENERGY_DRINK_PHRASES = (
    "energy drink",
    "energy drinks",
    "\u0645\u0634\u0631\u0648\u0628 \u0637\u0627\u0642\u0629",
    "\u0645\u0634\u0631\u0648\u0628\u0627\u062a \u0627\u0644\u0637\u0627\u0642\u0629",
)


ANCHOR_TOPICS = {
    "bone",
    "sports",
    "assessment",
}


def contains_phrase(
    value: object,
    phrase: object,
) -> bool:
    value_tokens = normalize_for_match(value).split()
    phrase_tokens = normalize_for_match(phrase).split()

    if not value_tokens or not phrase_tokens:
        return False

    phrase_length = len(phrase_tokens)

    if phrase_length > len(value_tokens):
        return False

    return any(
        value_tokens[index:index + phrase_length]
        == phrase_tokens
        for index in range(
            len(value_tokens) - phrase_length + 1
        )
    )


def enhanced_detect_topic_names(
    value: object,
) -> tuple[str, ...]:
    topics = list(
        base_detect_topic_names(value)
    )

    for topic_name, phrases in (
        SPECIAL_TOPIC_PHRASES.items()
    ):
        matched = any(
            contains_phrase(value, phrase)
            for phrase in phrases
        )

        if matched and topic_name not in topics:
            topics.append(topic_name)

    has_energy_drink_phrase = any(
        contains_phrase(value, phrase)
        for phrase in ENERGY_DRINK_PHRASES
    )

    if has_energy_drink_phrase:
        topics = [
            topic
            for topic in topics
            if topic != "energy"
        ]

        if "sports" not in topics:
            topics.append("sports")

    return tuple(
        dict.fromkeys(topics)
    )


def useful_topics(
    question: object,
) -> list[str]:
    topics = list(
        enhanced_detect_topic_names(question)
    )

    if (
        len(topics) > 1
        and "adolescent" in topics
    ):
        topics.remove("adolescent")

    return topics


def title_matches_topic(
    chunk: object,
    topic_name: str,
) -> bool:
    hierarchy = normalize_for_match(
        " ".join(
            [
                str(
                    getattr(
                        chunk,
                        "chapter_title",
                        "",
                    )
                    or ""
                ),
                str(
                    getattr(
                        chunk,
                        "section_title",
                        "",
                    )
                    or ""
                ),
                str(
                    getattr(
                        chunk,
                        "hierarchy_path",
                        "",
                    )
                    or ""
                ),
            ]
        )
    )

    definition = TOPICS.get(topic_name)

    if definition is None:
        return False

    for preferred_title in (
        definition.preferred_titles
    ):
        normalized_title = normalize_for_match(
            preferred_title
        )

        if (
            normalized_title
            and normalized_title in hierarchy
        ):
            return True

    return False


def choose_top1(
    question: object,
    stages: dict[str, list],
):
    topics = useful_topics(question)
    candidates = []

    for priority, stage_name in enumerate(
        (
            "hybrid",
            "dense",
            "reranked",
        )
    ):
        chunks = stages.get(stage_name, [])

        if not chunks:
            continue

        chunk = chunks[0]

        matched_topics = [
            topic
            for topic in topics
            if title_matches_topic(
                chunk,
                topic,
            )
        ]

        candidates.append(
            {
                "stage": stage_name,
                "priority": priority,
                "chunk": chunk,
                "matched_topics": matched_topics,
                "match_count": len(
                    matched_topics
                ),
            }
        )

    if not candidates:
        return None

    return max(
        candidates,
        key=lambda item: (
            item["match_count"],
            -item["priority"],
        ),
    )


def find_topic_anchor(
    topic_name: str,
    stages: dict[str, list],
):
    candidates = []

    for priority, stage_name in enumerate(
        (
            "hybrid",
            "dense",
            "reranked",
        )
    ):
        chunks = stages.get(stage_name, [])

        for rank, chunk in enumerate(
            chunks[:20],
            start=1,
        ):
            if title_matches_topic(
                chunk,
                topic_name,
            ):
                candidates.append(
                    (
                        rank,
                        priority,
                        stage_name,
                        chunk,
                    )
                )
                break

    if not candidates:
        return None

    candidates.sort(
        key=lambda item: (
            item[0],
            item[1],
        )
    )

    return candidates[0][3]


def build_topic_anchor_ranking(
    question: object,
    *,
    dense_stage: list,
    hybrid_stage: list,
    reranked_stage: list,
    final_k: int = 5,
) -> list:
    stages = {
        "dense": list(dense_stage or []),
        "hybrid": list(hybrid_stage or []),
        "reranked": list(
            reranked_stage or []
        ),
    }

    ordered = []

    selected = choose_top1(
        question,
        stages,
    )

    if selected is not None:
        ordered.append(
            selected["chunk"]
        )

    for topic in useful_topics(question):
        if topic not in ANCHOR_TOPICS:
            continue

        anchor = find_topic_anchor(
            topic,
            stages,
        )

        if anchor is not None:
            ordered.append(anchor)

    for stage_name in (
        "hybrid",
        "dense",
        "reranked",
    ):
        ordered.extend(
            stages[stage_name]
        )

    final = []
    seen_chunk_ids = set()
    seen_hierarchies = set()

    for chunk in ordered:
        chunk_id = str(
            getattr(
                chunk,
                "chunk_id",
                "",
            )
        )

        hierarchy_key = normalize_for_match(
            getattr(
                chunk,
                "hierarchy_path",
                "",
            )
            or getattr(
                chunk,
                "chapter_title",
                "",
            )
        )

        if (
            chunk_id
            and chunk_id in seen_chunk_ids
        ):
            continue

        if (
            hierarchy_key
            and hierarchy_key
            in seen_hierarchies
        ):
            continue

        if chunk_id:
            seen_chunk_ids.add(chunk_id)

        if hierarchy_key:
            seen_hierarchies.add(
                hierarchy_key
            )

        final.append(chunk)

        if len(final) >= max(
            1,
            int(final_k),
        ):
            break

    return final
