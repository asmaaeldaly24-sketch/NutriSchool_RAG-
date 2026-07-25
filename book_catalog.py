from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Final


AAP_CHAPTER_TITLES: Final[dict[int, str]] = {
    1: "Nutrition for the 21st Century - Integrating Nutrigenetics, Nutrigenomics, and Microbiomics",
    2: "Development of Gastrointestinal Function",
    3: "Breastfeeding",
    4: "Formula Feeding of Term Infants",
    5: "Nutritional Needs of the Preterm Infant",
    6: "Complementary Feeding",
    7: "Feeding the Child",
    8: "Adolescent Nutrition",
    9: "Nutrition in School, Preschool, and Child Care",
    10: "Pediatric Global Nutrition",
    11: "Nutritional Aspects of Vegetarian Diets",
    12: "Sports Nutrition",
    13: "Fast Foods, Organic Foods, Fad Diets, and Herbs, Herbals, and Botanicals",
    14: "Energy",
    15: "Protein",
    16: "Carbohydrate and Dietary Fiber",
    17: "Fats and Fatty Acids",
    18: "Calcium, Phosphorus, and Magnesium",
    19: "Iron",
    20: "Trace Elements",
    21: "Vitamins",
    22: "Parenteral Nutrition",
    23: "Enteral Feeding for Nutritional Support",
    24: "Assessment of Nutritional Status",
    25: "Pediatric Feeding and Swallowing Disorders",
    26: "Malnutrition, Undernutrition, and Failure to Thrive",
    27: "Chronic Diarrheal Disease",
    28: "Oral Therapy for Acute Diarrhea",
    29: "Inborn Errors of Metabolism",
    30: "Nutrition Therapy for Children and Adolescents With Type 1 and Type 2 Diabetes Mellitus",
    31: "Hypoglycemia in Infants and Children",
    32: "Dyslipidemia",
    33: "Pediatric Obesity",
    34: "Food Allergy",
    35: "Nutrition and Immunity",
    36: "Nutritional Support of Children With Developmental Disabilities",
    37: "Nutrition of Children Who Are Critically Ill",
    38: "Eating Disorders in Children and Adolescents",
    39: "Nutrition for Children With Sickle Cell Disease and Thalassemia",
    40: "Nutrition in Renal Disease",
    41: "Nutritional Management of Children With Cancer",
    42: "Nutrition in the Management of Chronic Autoimmune Inflammatory Bowel Diseases in Children",
    43: "Liver Disease",
    44: "Cardiac Disease",
    45: "Nutrition in Children With Short Bowel Syndrome",
    46: "Nutrition in Cystic Fibrosis",
    47: "The Ketogenic Diet",
    48: "Diet, Nutrition, and Oral Health",
    49: "Preventing Food Insecurity - Available Community Nutrition Programs",
    50: "Federal Regulation of Foods, Infant Formulas, and Food Labeling",
    51: "Food Safety: Infectious Disease",
    52: "Food Safety: Pesticides, Industrial Chemicals, Toxins, Preservatives, Irradiation, and Food Biotechnology",
}


PNP_CANONICAL_SECTIONS: Final[dict[str, str]] = {
    "1.1": "Child Growth",
    "1.2": "Nutritional Assessment",
    "1.2.1": "Clinical Evaluation and Anthropometry",
    "1.2.2": "Dietary History and Dietary Assessment",
    "1.2.3": "Technical Measurements of Body Composition",
    "1.2.4": "Use of Laboratory Measurements in Nutritional Assessment",
    "1.3": "Physiology of Nutrition",
    "1.3.1": "Nutrient Intake Values: Concepts and Applications",
    "1.3.2": "Energy Requirements of Infants, Children, and Adolescents",
    "1.3.3": "Protein",
    "1.3.4": "Digestible and Non-Digestible Carbohydrates",
    "1.3.5": "Dietary Lipid Intake",
    "1.3.6": "Fluid and Electrolytes",
    "1.4": "Micronutrient Needs in Children and Adolescents",
    "1.4.1": "Physical Activity, Health, and Nutrition",
    "1.4.2": "Early Nutrition Impact on Long-Term Health",
    "1.4.3": "Food Safety",
    "1.4.5": "Gut Microbiota Development in Infants and Children",
    "1.4.7": "Nutrition, Brain Development, and Mental Health",
    "2.6": "Dietary Needs and Challenges in Toddlers and Young Children",
    "2.7": "Adolescent Nutrition: Issues and Actions",
    "2.9": "Vegetarian and Vegan Diets",
    "3.2": "Iron: Nutritional Deficiency and Excess",
    "3.3": "Micronutrient Deficiencies",
    "3.4": "Enteral Nutrition Support",
    "3.5": "Parenteral Nutrition Support",
    "3.6": "Management of Child and Adolescent Obesity",
    "3.7": "Acute and Prolonged Childhood Diarrhea",
    "3.11": "Celiac Disease",
    "3.12": "Food Intolerance and Allergy",
    "3.13": "Constipation and the Efficacy of Fluid, Fat, Fibers, and Prebiotics",
    "3.18": "Nutrition in Children with Diabetes Mellitus",
    "3.22": "Nutrition in Cystic Fibrosis",
    "3.24": "Nutritional Management in Children with Chronic Kidney Disease",
    "3.25": "Diet in Children with Malignant Disease",
    "5.2": "Reference Nutrient Intakes of Infants, Children, and Adolescents",
}


AAP_SCHOOL_EXCLUDED_CHAPTERS: Final[frozenset[int]] = frozenset({2, 3, 4, 5, 6})
PNP_SCHOOL_EXCLUDED_PREFIXES: Final[tuple[str, ...]] = (
    "2.1",
    "2.2",
    "2.3",
    "2.4",
    "2.5",
    "2.6",
    "3.15",
    "3.16",
    "5.3",
)


REFERENCE_MARKERS: Final[tuple[str, ...]] = (
    "references",
    "bibliography",
    "doi:",
    "published online",
    "available at:",
    "accessed ",
    "et al.",
)


@dataclass(frozen=True)
class TopicDefinition:
    name: str
    retrieval_phrase: str
    aliases: tuple[str, ...]
    preferred_titles: tuple[str, ...]


TOPICS: Final[dict[str, TopicDefinition]] = {
    "school_meals": TopicDefinition(
        name="school_meals",
        retrieval_phrase="school meals breakfast lunch snacks vending packed lunch child care nutrition",
        aliases=(
            "school meal",
            "school lunch",
            "school breakfast",
            "packed lunch",
            "\u0648\u062c\u0628\u0629 \u0645\u062f\u0631\u0633\u064a\u0629",
            "\u0648\u062c\u0628\u0627\u062a \u0627\u0644\u0645\u062f\u0631\u0633\u0629",
            "\u0641\u0637\u0627\u0631 \u0627\u0644\u0645\u062f\u0631\u0633\u0629",
        ),
        preferred_titles=("Nutrition in School, Preschool, and Child Care",),
    ),
    "adolescent": TopicDefinition(
        name="adolescent",
        retrieval_phrase="adolescent nutrition puberty growth nutrient requirements",
        aliases=("adolescent", "teen", "puberty", "\u0645\u0631\u0627\u0647\u0642", "\u0645\u0631\u0627\u0647\u0642\u0629", "\u0627\u0644\u0628\u0644\u0648\u063a"),
        preferred_titles=("Adolescent Nutrition", "Adolescent Nutrition: Issues and Actions"),
    ),
    "sports": TopicDefinition(
        name="sports",
        retrieval_phrase="sports nutrition young athlete hydration carbohydrate recovery energy drinks",
        aliases=("sports", "athlete", "exercise", "\u0631\u064a\u0627\u0636\u0629", "\u0631\u064a\u0627\u0636\u064a", "\u062a\u0645\u0631\u064a\u0646"),
        preferred_titles=("Sports Nutrition", "Physical Activity, Health, and Nutrition"),
    ),
    "vegetarian": TopicDefinition(
        name="vegetarian",
        retrieval_phrase=(
            "vegetarian vegan child adolescent protein vitamin B12 "
            "iron zinc calcium vitamin D"
        ),
        aliases=(
            "vegetarian",
            "vegan",
            "vegetarian diet",
            "vegan diet",
            "نباتي",
            "نباتية",
            "النباتي",
            "النباتية",
            "نباتي صرف",
            "النباتي الصرف",
            "نباتيون",
            "النباتيون",
            "نباتيون صرف",
            "النباتيون الصرف",
            "نظام نباتي",
            "حمية نباتية",
        ),
        preferred_titles=(
            "Nutritional Aspects of Vegetarian Diets",
            "Vegetarian and Vegan Diets",
        ),
    ),
    "energy": TopicDefinition(
        name="energy",
        retrieval_phrase="energy requirements total energy expenditure growth physical activity school-age children",
        aliases=("energy", "calorie", "caloric", "\u0637\u0627\u0642\u0629", "\u0633\u0639\u0631\u0627\u062a"),
        preferred_titles=("Energy", "Energy Requirements of Infants, Children, and Adolescents"),
    ),
    "protein": TopicDefinition(
        name="protein",
        retrieval_phrase="protein requirements quality amino acids food sources growing children",
        aliases=("protein", "amino acid", "\u0628\u0631\u0648\u062a\u064a\u0646", "\u0627\u062d\u0645\u0627\u0636 \u0627\u0645\u064a\u0646\u064a\u0629"),
        preferred_titles=("Protein",),
    ),
    "carbohydrate": TopicDefinition(
        name="carbohydrate",
        retrieval_phrase="carbohydrate dietary fiber school-age children constipation glycemic",
        aliases=("carbohydrate", "fiber", "fibre", "\u0643\u0631\u0628\u0648\u0647\u064a\u062f\u0631\u0627\u062a", "\u0623\u0644\u064a\u0627\u0641", "\u0627\u0644\u064a\u0627\u0641"),
        preferred_titles=("Carbohydrate and Dietary Fiber", "Digestible and Non-Digestible Carbohydrates"),
    ),
    "fat": TopicDefinition(
        name="fat",
        retrieval_phrase="dietary fats fatty acids omega-3 child adolescent nutrition",
        aliases=("fat", "fatty acid", "lipid", "omega", "\u062f\u0647\u0648\u0646", "\u0627\u062d\u0645\u0627\u0636 \u062f\u0647\u0646\u064a\u0629"),
        preferred_titles=("Fats and Fatty Acids", "Dietary Lipid Intake"),
    ),
    "iron": TopicDefinition(
        name="iron",
        retrieval_phrase="iron deficiency anemia cognition school performance adolescent girls",
        aliases=("iron", "anemia", "anaemia", "\u062d\u062f\u064a\u062f", "\u0627\u0646\u064a\u0645\u064a\u0627", "\u0641\u0642\u0631 \u0627\u0644\u062f\u0645"),
        preferred_titles=("Iron", "Iron: Nutritional Deficiency and Excess"),
    ),
    "bone": TopicDefinition(
        name="bone",
        retrieval_phrase="calcium phosphorus magnesium vitamin D bone mineral adolescence peak bone mass",
        aliases=("calcium", "phosphorus", "magnesium", "bone", "\u0643\u0627\u0644\u0633\u064a\u0648\u0645", "\u0641\u0648\u0633\u0641\u0648\u0631", "\u0645\u063a\u0646\u064a\u0633\u064a\u0648\u0645", "\u0639\u0638\u0627\u0645"),
        preferred_titles=("Calcium, Phosphorus, and Magnesium", "Micronutrient Needs in Children and Adolescents"),
    ),
    "obesity": TopicDefinition(
        name="obesity",
        retrieval_phrase="pediatric obesity assessment staged family-based weight management physical activity screen time",
        aliases=("obesity", "overweight", "weight management", "\u0633\u0645\u0646\u0629", "\u0632\u064a\u0627\u062f\u0629 \u0627\u0644\u0648\u0632\u0646"),
        preferred_titles=("Pediatric Obesity", "Management of Child and Adolescent Obesity"),
    ),
    "diabetes": TopicDefinition(
        name="diabetes",
        retrieval_phrase="pediatric diabetes nutrition therapy carbohydrate counting meal planning exercise hypoglycemia",
        aliases=("diabetes", "diabetic", "\u0633\u0643\u0631", "\u0633\u0643\u0631\u064a"),
        preferred_titles=(
            "Nutrition Therapy for Children and Adolescents With Type 1 and Type 2 Diabetes Mellitus",
            "Nutrition in Children with Diabetes Mellitus",
        ),
    ),
    "food_allergy": TopicDefinition(
        name="food_allergy",
        retrieval_phrase="food allergy elimination diet nutrient adequacy school meals cross-contact allergen safety",
        aliases=(
            "food allergy",
            "allergen",
            "allergens",
            "elimination diet",
            "cross-contact",
            "\u062d\u0633\u0627\u0633\u064a\u0629 \u0627\u0644\u0637\u0639\u0627\u0645",
            "\u062d\u0633\u0627\u0633\u064a\u0629 \u0637\u0639\u0627\u0645",
            "\u062d\u0633\u0627\u0633\u064a\u0629 \u063a\u0630\u0627\u0626\u064a\u0629",
            "\u0645\u0633\u0628\u0628 \u0627\u0644\u062d\u0633\u0627\u0633\u064a\u0629",
            "\u0645\u0633\u0628\u0628\u0627\u062a \u0627\u0644\u062d\u0633\u0627\u0633\u064a\u0629",
            "\u062d\u0645\u064a\u0629 \u0627\u0633\u062a\u0628\u0639\u0627\u062f",
            "\u0646\u0638\u0627\u0645 \u0627\u0633\u062a\u0628\u0639\u0627\u062f",
            "\u0627\u0633\u062a\u0628\u0639\u0627\u062f \u063a\u0630\u0627\u0626\u064a",
            "\u062a\u0644\u0627\u0645\u0633 \u0639\u0631\u0636\u064a",
            "\u0627\u0644\u062a\u0644\u0627\u0645\u0633 \u0627\u0644\u0639\u0631\u0636\u064a",
        ),
        preferred_titles=("Food Allergy", "Food Intolerance and Allergy"),
    ),
    "assessment": TopicDefinition(
        name="assessment",
        retrieval_phrase="pediatric nutritional assessment growth chart BMI anthropometry dietary history laboratory",
        aliases=("nutritional assessment", "growth chart", "BMI", "anthropometry", "\u062a\u0642\u064a\u064a\u0645 \u063a\u0630\u0627\u0626\u064a", "\u0645\u0624\u0634\u0631 \u0643\u062a\u0644\u0629 \u0627\u0644\u062c\u0633\u0645"),
        preferred_titles=("Assessment of Nutritional Status", "Nutritional Assessment"),
    ),
    "eating_disorder": TopicDefinition(
        name="eating_disorder",
        retrieval_phrase="eating disorders adolescents warning signs nutritional rehabilitation anorexia bulimia",
        aliases=("eating disorder", "anorexia", "bulimia", "\u0627\u0636\u0637\u0631\u0627\u0628 \u0627\u0644\u0627\u0643\u0644", "\u0641\u0642\u062f\u0627\u0646 \u0627\u0644\u0634\u0647\u064a\u0629"),
        preferred_titles=("Eating Disorders in Children and Adolescents",),
    ),
    "dyslipidemia": TopicDefinition(
        name="dyslipidemia",
        retrieval_phrase="pediatric dyslipidemia dietary management cholesterol saturated fat fiber",
        aliases=("dyslipidemia", "cholesterol", "lipid disorder", "\u062f\u0647\u0648\u0646 \u0627\u0644\u062f\u0645", "\u0643\u0648\u0644\u064a\u0633\u062a\u0631\u0648\u0644"),
        preferred_titles=("Dyslipidemia",),
    ),
    "oral_health": TopicDefinition(
        name="oral_health",
        retrieval_phrase="diet nutrition oral health dental caries sugary drinks children adolescents",
        aliases=("oral health", "dental caries", "tooth decay", "\u062a\u0633\u0648\u0633", "\u0635\u062d\u0629 \u0627\u0644\u0641\u0645", "\u0645\u0634\u0631\u0648\u0628\u0627\u062a \u0633\u0643\u0631\u064a\u0629"),
        preferred_titles=("Diet, Nutrition, and Oral Health",),
    ),
    "food_safety": TopicDefinition(
        name="food_safety",
        retrieval_phrase="school packed lunch food safety cross contamination temperature infectious disease",
        aliases=(
            "food safety",
            "cross contamination",
            "packed lunch safety",
            "lunch box safety",
            "safe temperature",
            "safe temperatures",
            "\u0633\u0644\u0627\u0645\u0629 \u0627\u0644\u063a\u0630\u0627\u0621",
            "\u0633\u0644\u0627\u0645\u0629 \u0627\u0644\u0648\u062c\u0628\u0627\u062a",
            "\u0633\u0644\u0627\u0645\u0629 \u0627\u0644\u0644\u0627\u0646\u0634 \u0628\u0648\u0643\u0633",
            "\u0644\u0627\u0646\u0634 \u0628\u0648\u0643\u0633",
            "\u062a\u0644\u0648\u062b \u062a\u0628\u0627\u062f\u0644\u064a",
            "\u0627\u0644\u062a\u0644\u0648\u062b \u0627\u0644\u062a\u0628\u0627\u062f\u0644\u064a",
            "\u062f\u0631\u062c\u0627\u062a \u062d\u0631\u0627\u0631\u0629 \u0622\u0645\u0646\u0629",
            "\u062f\u0631\u062c\u0629 \u062d\u0631\u0627\u0631\u0629 \u0622\u0645\u0646\u0629",
        ),
        preferred_titles=("Food Safety: Infectious Disease", "Food Safety"),
    ),
    "food_insecurity": TopicDefinition(
        name="food_insecurity",
        retrieval_phrase="food insecurity screening school breakfast lunch nutrition assistance programs",
        aliases=("food insecurity", "school meal program", "\u0627\u0646\u0639\u062f\u0627\u0645 \u0627\u0644\u0627\u0645\u0646 \u0627\u0644\u063a\u0630\u0627\u0626\u064a", "\u0646\u0642\u0635 \u0627\u0644\u063a\u0630\u0627\u0621"),
        preferred_titles=("Preventing Food Insecurity", "Nutrition in School, Preschool, and Child Care"),
    ),
    "food_labeling": TopicDefinition(
        name="food_labeling",
        retrieval_phrase="food labeling allergens nutrition facts school-age children family food choices",
        aliases=(
            "food label",
            "nutrition label",
            "allergen label",
            "ingredient label",
            "\u0628\u0637\u0627\u0642\u0629 \u063a\u0630\u0627\u0626\u064a\u0629",
            "\u0628\u0637\u0627\u0642\u0629 \u0627\u0644\u063a\u0630\u0627\u0621",
            "\u0628\u0637\u0627\u0642\u0629 \u0627\u0644\u0637\u0639\u0627\u0645",
            "\u0628\u0637\u0627\u0642\u0629 \u0627\u0644\u0645\u0643\u0648\u0646\u0627\u062a",
            "\u0645\u0644\u0635\u0642 \u063a\u0630\u0627\u0626\u064a",
            "\u0645\u0644\u0635\u0642 \u0627\u0644\u0637\u0639\u0627\u0645",
            "\u0627\u0644\u0628\u064a\u0627\u0646\u0627\u062a \u0627\u0644\u063a\u0630\u0627\u0626\u064a\u0629",
        ),
        preferred_titles=("Federal Regulation of Foods", "Food Allergy"),
    ),
    "fast_food": TopicDefinition(
        name="fast_food",
        retrieval_phrase="fast food fad diets herbs botanicals adolescents nutritional risks",
        aliases=("fast food", "fad diet", "herbal", "\u0648\u062c\u0628\u0627\u062a \u0633\u0631\u064a\u0639\u0629", "\u062d\u0645\u064a\u0629 \u0631\u0627\u0626\u062c\u0629", "\u0627\u0639\u0634\u0627\u0628"),
        preferred_titles=("Fast Foods, Organic Foods, Fad Diets",),
    ),
}


def canonical_aap_chapter(number: int | str) -> str:
    chapter_number = int(number)
    title = AAP_CHAPTER_TITLES.get(chapter_number, "")
    return f"Chapter {chapter_number}: {title}" if title else f"Chapter {chapter_number}"


def canonical_pnp_section(number: str, extracted_title: str = "") -> str:
    canonical = PNP_CANONICAL_SECTIONS.get(number)
    title = canonical or " ".join(str(extracted_title or "").split()).strip()
    return f"{number} {title}".strip()


_ARABIC_CHARACTER_MAP = str.maketrans(
    {
        "\u0623": "\u0627",
        "\u0625": "\u0627",
        "\u0622": "\u0627",
        "\u0649": "\u064a",
        "\u0624": "\u0648",
        "\u0626": "\u064a",
    }
)

_ARABIC_COMBINED_PREFIXES = (
    "\u0648\u0627\u0644",
    "\u0641\u0627\u0644",
    "\u0628\u0627\u0644",
    "\u0643\u0627\u0644",
)

_ARABIC_LL_PREFIX = "\u0644\u0644"
_ARABIC_AL_PREFIX = "\u0627\u0644"


def _clean_match_text(value: object) -> str:
    import unicodedata

    text = str(value or "").lower().strip()
    text = unicodedata.normalize("NFKD", text)

    cleaned_characters = []

    for character in text:
        codepoint = ord(character)

        if unicodedata.category(character) == "Mn":
            continue

        if codepoint == 0x0640:
            continue

        cleaned_characters.append(character)

    cleaned = "".join(cleaned_characters)

    return cleaned.translate(
        _ARABIC_CHARACTER_MAP
    )


def _contains_arabic(token: str) -> bool:
    return any(
        0x0600 <= ord(character) <= 0x06FF
        for character in token
    )


def _normalize_match_token(token: str) -> str:
    token = _clean_match_text(token)

    if not token:
        return ""

    if _contains_arabic(token):
        prefix_removed = False

        for prefix in _ARABIC_COMBINED_PREFIXES:
            if (
                token.startswith(prefix)
                and len(token) >= len(prefix) + 3
            ):
                token = token[len(prefix):]
                prefix_removed = True
                break

        if not prefix_removed:
            if (
                token.startswith(_ARABIC_LL_PREFIX)
                and len(token) >= 5
            ):
                token = token[
                    len(_ARABIC_LL_PREFIX):
                ]

            elif (
                token.startswith(_ARABIC_AL_PREFIX)
                and len(token) >= 5
            ):
                token = token[
                    len(_ARABIC_AL_PREFIX):
                ]

    return token.strip()


def normalize_for_match(value: object) -> str:
    cleaned = _clean_match_text(value)

    tokens = []
    current_token = []

    for character in cleaned:
        if character.isalnum():
            current_token.append(character)
        else:
            if current_token:
                tokens.append(
                    "".join(current_token)
                )
                current_token = []

    if current_token:
        tokens.append(
            "".join(current_token)
        )

    normalized_tokens = [
        _normalize_match_token(token)
        for token in tokens
    ]

    return " ".join(
        token
        for token in normalized_tokens
        if token
    )


def _contains_normalized_phrase(
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

    for index in range(
        len(value_tokens) - phrase_length + 1
    ):
        candidate = value_tokens[
            index:index + phrase_length
        ]

        if candidate == phrase_tokens:
            return True

    return False


def detect_topic_names(
    value: object,
) -> tuple[str, ...]:
    detected = []

    for topic_name, definition in TOPICS.items():
        matched = any(
            _contains_normalized_phrase(
                value,
                alias,
            )
            for alias in definition.aliases
        )

        if matched:
            detected.append(topic_name)

    return tuple(detected)


def preferred_title_score(query: object, hierarchy: object) -> float:
    hierarchy_normalized = normalize_for_match(hierarchy)
    topics = detect_topic_names(query)
    score = 0.0

    for topic in topics:
        definition = TOPICS[topic]
        for preferred in definition.preferred_titles:
            preferred_normalized = normalize_for_match(preferred)
            if preferred_normalized and preferred_normalized in hierarchy_normalized:
                score = max(score, 1.0)
            else:
                preferred_tokens = set(preferred_normalized.split())
                hierarchy_tokens = set(hierarchy_normalized.split())
                if preferred_tokens:
                    overlap = len(preferred_tokens & hierarchy_tokens) / len(preferred_tokens)
                    score = max(score, overlap)

    return min(1.0, score)


def school_scope_allows(book_id: str, chapter_number: str, hierarchy: str) -> bool:
    if book_id == "aap8":
        try:
            return int(chapter_number) not in AAP_SCHOOL_EXCLUDED_CHAPTERS
        except (TypeError, ValueError):
            return False

    if book_id == "pnp3":
        section = str(chapter_number or "").strip()
        normalized_hierarchy = normalize_for_match(hierarchy)
        if any(section == prefix or section.startswith(f"{prefix}.") for prefix in PNP_SCHOOL_EXCLUDED_PREFIXES):
            return False
        infant_markers = (
            "breastfeeding",
            "formula feeding",
            "complementary feeding",
            "preterm infant",
            "feeding my baby",
        )
        return not any(marker in normalized_hierarchy for marker in infant_markers)

    return True