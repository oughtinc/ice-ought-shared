import re

from collections.abc import Callable
from dataclasses import dataclass
from typing import Final
from typing import Optional

Column = {}
Source = {}
QAResult = {}

# In testing, I found that the model is too hesitant to answer for many questions,
# so these biases improved recall
# without hurting precision
# https://ought-inc.slack.com/archives/C0291QNPSBY/p1648606550270729
pro_answer_bias: Final = -3.4

neutral_answer_bias: Final = 0

# for Organism and user custom noun phrases,
# I found in testing that the model is too eager to answer,
# so this bias struck the best balance between precision and recall
# https://ought-inc.slack.com/archives/C0291QNPSBY/p1649201236916799
anti_answer_bias: Final = 3

strong_pro_answer_bias: Final = -6

# Population Summary Few-shot prompt

POPULATION_PROMPT = """Combine the following pieces of information about a population into a single sentence.
 You MUST use ALL of the information provided.

Number of participants: 37
Region: Italy
Organism: humans
Age: 19 to 50 years old
Sex: female
Based on the information provided, the aggregated sentence is: 37 women aged 19 to 50 years old in Italy
Information used: Number of participants, Region, Organism, Age, Sex

Number of participants: 10
Organism: humans
Region: United States
Health conditions: major depression
Based on the information provided, the aggregated sentence is: 10 people in United States with major depression
Information used: Number of participants, Organism, Region, Health conditions,

Number of participants: 10
Characteristics: triathletes
Organism: humans
Region: France
Age: 15 to 35 years old
Based on the information provided, the aggregated sentence is: 10 triathletes in France aged 15 to 35 years old
Information used: Number of participants, Characteristics, Organism, Region, Age"""


def generate_population_summary_prompt(sub_results: dict[str, QAResult]) -> str:
    if len(sub_results) == 0:
        raise ValueError("No sub-results provided")

    population_details = "\n".join(
        [
            f"{key}: {sub_result.answer}"
            for key, sub_result in sub_results.items()
            if sub_result.answer is not None
        ]
    )
    return (
        POPULATION_PROMPT
        + f"""

{population_details}
Based on the information provided, the aggregated sentence is:"""
    )


# Intervention Summary Few-shot prompt

INTERVENTION_PROMPT = """
Combine the following pieces of information about an intervention into a single sentence.
You MUST use ALL of the information provided.

Intervention: 400-mg albendazole every 3 months
Dose: albendazole: 400 mg\npraziquantel: 40 mg/kg
Duration: 3 years
Aggregated sentence: 400 mg of albendazole and 40 mg/kg of praziquantel every 3 months for 3 years
Information used: Intervention, Dose, Duration

Intervention: school based deworming
Aggregated sentence: School based deworming
Information used: Intervention

Intervention: deworming drugs given to children without screening for infection
Duration: 2 years
Aggregated sentence: Deworming drugs given to children without screening for infection for 2 years
Information used: Intervention, Dose, Duration
""".strip()


def generate_intervention_summary_prompt(sub_results: dict[str, QAResult]) -> str:
    if len(sub_results) == 0:
        raise ValueError("No sub-results provided")

    intervention_details = "\n".join(
        [
            f"{key}: {sub_result.answer}"
            for key, sub_result in sub_results.items()
            if sub_result.answer is not None
        ]
    )
    return (
        POPULATION_PROMPT
        + f"""

{intervention_details}
Aggregated sentence:"""
    )


class AnswerStrategy:
    ...


@dataclass
class DidSearchReturnResultAnswerStrategy(AnswerStrategy):
    yes_answer: str


class LMAnswerStrategy(AnswerStrategy):
    ...


@dataclass
class InstructAnswerStrategy(LMAnswerStrategy):
    answer_prefix: Optional[str]
    answer_bias: float
    process_answer: Optional[Callable[[str], str]] = None


class RephrasedQuestionAnswerStrategy(AnswerStrategy):
    ...


class SearchStrategy:
    ...


class LMSearchStrategy(SearchStrategy):
    ...


class RephrasedQuestionSearchStrategy(SearchStrategy):
    ...


@dataclass
class RegexSearchStrategy(SearchStrategy):
    patterns: list[re.Pattern]


@dataclass
class QAColumnConfig:
    search_strategy: SearchStrategy
    answer_strategy: AnswerStrategy
    question: str
    is_numerical: bool = False
    sources: frozenset[Source] = frozenset(["abstract", "body"])
    ignore_section_pattern: Optional[re.Pattern[str]] = re.compile(
        "acknowledgement", re.IGNORECASE
    )

    def __post_init__(self):
        if (
            isinstance(self.search_strategy, RegexSearchStrategy)
            and len(self.sources) > 1
        ):
            raise TypeError(
                "RegexSearchStrategy can only be used with a single source, "
                f"but was used with sources: {self.sources!r}"
            )

        if (
            isinstance(self.answer_strategy, DidSearchReturnResultAnswerStrategy)
            and "abstract" in self.sources
        ):
            raise TypeError(
                "DidSearchReturnResultAnswerStrategy cannot be used on the abstract"
            )


@dataclass
class CompositionalAnswerStrategy(AnswerStrategy):
    answer_bias: float
    sub_questions: dict[str, Column]
    create_aggregation_prompt: Callable[[dict[str, QAResult]], str]
    openai_stop: Optional[tuple[str]] = None


abstract_only: frozenset[Source] = frozenset(["abstract"])
body_only: frozenset[Source] = frozenset(["body"])

qa_column_configs: dict[str, QAColumnConfig] = {
    "intervention": QAColumnConfig(
        sources=abstract_only,
        question="What treatment did the authors test?",
        search_strategy=LMSearchStrategy(),
        answer_strategy=InstructAnswerStrategy(
            answer_prefix="The treatment the authors tested was",
            answer_bias=pro_answer_bias,
        ),
    ),
    "outcome-measured": QAColumnConfig(
        sources=abstract_only,
        question="What were the outcome variables?",
        search_strategy=LMSearchStrategy(),
        answer_strategy=InstructAnswerStrategy(
            answer_prefix="The outcome variables were:\n*",
            answer_bias=pro_answer_bias,
            process_answer=lambda answer: f"* {answer}".replace("*", "•"),
        ),
    ),
    "outcome": QAColumnConfig(
        sources=abstract_only,
        question="What were the outcome variables?",
        search_strategy=LMSearchStrategy(),
        answer_strategy=InstructAnswerStrategy(
            answer_prefix="The outcome variables were:\n*",
            answer_bias=pro_answer_bias,
            process_answer=lambda answer: f"* {answer}".replace("*", "•"),
        ),
    ),
    "duration": QAColumnConfig(
        question="What was the duration of the intervention in this paper?",
        search_strategy=LMSearchStrategy(),
        answer_strategy=InstructAnswerStrategy(
            answer_prefix="The duration of the intervention in this paper was",
            answer_bias=pro_answer_bias,
        ),
        is_numerical=True,
    ),
    "participant-age": QAColumnConfig(
        question="\
What was the age of the participants in this paper? \
(e.g. children, 40-55, 20 +/- 2 years, middle-aged)",
        search_strategy=LMSearchStrategy(),
        answer_strategy=InstructAnswerStrategy(
            answer_prefix="The age of the participants in this paper was",
            answer_bias=pro_answer_bias,
        ),
        is_numerical=True,
    ),
    "participant-count": QAColumnConfig(
        question="What was the number of participants in this paper?",
        search_strategy=LMSearchStrategy(),
        answer_strategy=InstructAnswerStrategy(
            answer_prefix="The number of participants in this paper was",
            answer_bias=pro_answer_bias,
        ),
        is_numerical=True,
    ),
    "final_n": QAColumnConfig(
        question="What was the number of participants in the final analysis of this paper?",
        search_strategy=LMSearchStrategy(),
        answer_strategy=InstructAnswerStrategy(
            answer_prefix="The number of participants in the final analysis was",
            answer_bias=pro_answer_bias,
        ),
        is_numerical=True,
    ),
    "region": QAColumnConfig(
        question="What was the country or region where this study was done?",
        search_strategy=LMSearchStrategy(),
        answer_strategy=InstructAnswerStrategy(
            answer_prefix="The country or region where this study was done was",
            answer_bias=pro_answer_bias,
        ),
    ),
    "country": QAColumnConfig(
        question="What was the country or region where this study was done?",
        search_strategy=LMSearchStrategy(),
        answer_strategy=InstructAnswerStrategy(
            answer_prefix="The country or region where this study was done was",
            answer_bias=pro_answer_bias,
        ),
    ),
    "organism": QAColumnConfig(
        question="What organism was studied in this paper? (e.g. rats, humans, arabidopsis)",
        search_strategy=LMSearchStrategy(),
        answer_strategy=InstructAnswerStrategy(
            answer_prefix="The organism studied was", answer_bias=neutral_answer_bias
        ),
    ),
    "study-count": QAColumnConfig(
        question="How many studies were included in this final review or meta-analysis?",
        search_strategy=LMSearchStrategy(),
        answer_strategy=InstructAnswerStrategy(
            answer_prefix="The number of studies included was",
            # I found in testing that a very strong pro-answer bias was best:
            # https://deepnote.com/project/QA-iteration-starting-23-Mar-820-PM-D5oXII1ETTm75jHAFIOuWw/%2Ftest_prompts.ipynb
            answer_bias=strong_pro_answer_bias,
        ),
        is_numerical=True,
    ),
    "population-characteristics": QAColumnConfig(
        question="What groups of participants did they study in this paper?",
        search_strategy=LMSearchStrategy(),
        answer_strategy=InstructAnswerStrategy(
            answer_prefix="The groups of participants they studied in this paper were",
            answer_bias=pro_answer_bias,
        ),
    ),
    "detailed-study-type": QAColumnConfig(
        sources=abstract_only,
        question="What was the study design?",
        search_strategy=LMSearchStrategy(),
        answer_strategy=InstructAnswerStrategy(
            answer_prefix="The study design was", answer_bias=pro_answer_bias
        ),
    ),
    "preregistered": QAColumnConfig(
        sources=body_only,
        question="Was the study pre-registered?",
        answer_strategy=DidSearchReturnResultAnswerStrategy(
            yes_answer="Preregistered",
        ),
        search_strategy=RegexSearchStrategy(
            patterns=[re.compile(r"pre-? ?(?:regist|analy)", re.IGNORECASE)],
        ),
        ignore_section_pattern=None,
    ),
    "multiple-comparisons": QAColumnConfig(
        sources=body_only,
        question="Does the study adjust for multiple comparisons?",
        answer_strategy=DidSearchReturnResultAnswerStrategy(
            yes_answer="Adjusts for multiple comparisions",
        ),
        search_strategy=RegexSearchStrategy(
            patterns=[
                re.compile(pattern, re.IGNORECASE)
                for pattern in [
                    r"multiple[- ](?:comparison|hypoth)",
                    # from https://bmcmedresmethodol.biomedcentral.com/articles/10.1186/s12874-019-0754-4:
                    r"Bonferroni",
                    r"Holm method",
                    r"Hochberg method",
                    r"Dubey/Armitage-Parmar",
                    r"Stepdown-minP",
                ]
            ],
        ),
        ignore_section_pattern=None,
    ),
    "itt": QAColumnConfig(
        sources=body_only,
        question="Does the study include intent-to-treat analysis?",
        answer_strategy=DidSearchReturnResultAnswerStrategy(
            yes_answer="Intent-to-treat analysis included",
        ),
        search_strategy=RegexSearchStrategy(
            patterns=[
                re.compile(r"intent(?:ion)?[- ]to[- ]treat", re.IGNORECASE),
                re.compile(r"ITT"),
            ],
        ),
        ignore_section_pattern=None,
    ),
    "query-relevant-summary": QAColumnConfig(
        sources=abstract_only,
        question="Question-relevant summary",
        answer_strategy=RephrasedQuestionAnswerStrategy(),
        search_strategy=RephrasedQuestionSearchStrategy(),
    ),
    "limitations": QAColumnConfig(
        sources=body_only,
        question="What were the limitations of this study?",
        answer_strategy=InstructAnswerStrategy(
            answer_prefix="The limitations of this study were",
            answer_bias=neutral_answer_bias,  # this may work well with pro_answer_bias, we haven't tested
        ),
        search_strategy=LMSearchStrategy(),
    ),
    "funding_source": QAColumnConfig(
        sources=body_only,
        question="Who funded this study?",
        answer_strategy=InstructAnswerStrategy(
            answer_prefix="The study was funded by",
            answer_bias=neutral_answer_bias,  # this may work well with pro_answer_bias, we haven't tested
        ),
        search_strategy=LMSearchStrategy(),
    ),
    "dose": QAColumnConfig(
        question="""What substances were administered to participants, and what were the doses? e.g.:

ibuprofen: 200 mg
placebo: dose not mentioned""",
        search_strategy=LMSearchStrategy(),
        answer_strategy=InstructAnswerStrategy(
            answer_prefix="The substances (with dosages) in this paper were:\n",
            answer_bias=pro_answer_bias,
        ),
        is_numerical=True,
    ),
    # # Start compositional population
    # "population-breakdown": QAColumnConfig(
    #     # TODO (jason) compositional columns should not have a root question. This right
    #     # now is only used in logging. This should perhaps be `CompositionalColumnConfig`
    #     question="What is the population summary?",
    #     search_strategy=LMSearchStrategy(),
    #     answer_strategy=CompositionalAnswerStrategy(
    #         answer_bias=strong_pro_answer_bias,
    #         sub_questions={
    #             "Number of participants": Column(
    #                 type="predefined", value="participant-count"
    #             ),
    #             "Age": Column(type="predefined", value="participant-age"),
    #             "Organism": Column(type="predefined", value="organism"),
    #             "Region": Column(type="predefined", value="region"),
    #             "Sex": Column(type="predefined", value="Population - Sex"),
    #             "Health conditions": Column(
    #                 type="predefined", value="Population - Health Conditions"
    #             ),
    #             "Occupation": Column(
    #                 type="predefined", value="Population - Occupation"
    #             ),
    #             "Characteristics": Column(
    #                 type="predefined", value="population-characteristics"
    #             ),
    #         },
    #         create_aggregation_prompt=generate_population_summary_prompt,
    #         openai_stop=("Information used",),
    #     ),
    # ),
    "Population - Sex": QAColumnConfig(
        question="""What sex(es) were the participants? E.g. "female", "male", "female and male?""",
        search_strategy=LMSearchStrategy(),
        answer_strategy=InstructAnswerStrategy(
            answer_prefix="Participant's sex(es) were",
            answer_bias=pro_answer_bias,  # TODO extremely_strong_pro_answer_bias
        ),
    ),
    "sex": QAColumnConfig(
        question="""What sex(es) were the participants? E.g. "female", "male", "female and male?""",
        search_strategy=LMSearchStrategy(),
        answer_strategy=InstructAnswerStrategy(
            answer_prefix="Participant's sex(es) were",
            answer_bias=pro_answer_bias,  # TODO extremely_strong_pro_answer_bias
        ),
    ),
    "health_conditions": QAColumnConfig(
        question="""What, if any, health condition(s) did participants have? E.g. "cancer", "depression?""",
        search_strategy=LMSearchStrategy(),
        answer_strategy=InstructAnswerStrategy(
            answer_prefix="Participant's health condition(s) was/were",
            answer_bias=pro_answer_bias,  # TODO extremely_strong_pro_answer_bias
        ),
    ),
    "Population - Health Conditions": QAColumnConfig(
        question="""What, if any, health condition(s) did participants have? E.g. "cancer", "depression?""",
        search_strategy=LMSearchStrategy(),
        answer_strategy=InstructAnswerStrategy(
            answer_prefix="Participant's health condition(s) was/were",
            answer_bias=pro_answer_bias,  # TODO extremely_strong_pro_answer_bias
        ),
    ),
    "Population - Occupation": QAColumnConfig(
        question="""What were the participants' occupations?""",
        search_strategy=LMSearchStrategy(),
        answer_strategy=InstructAnswerStrategy(
            answer_prefix="The participants' occupations were",
            answer_bias=pro_answer_bias,
        ),
    ),
    "occupation": QAColumnConfig(
        question="""What were the participants' occupations?""",
        search_strategy=LMSearchStrategy(),
        answer_strategy=InstructAnswerStrategy(
            answer_prefix="The participants' occupations were",
            answer_bias=pro_answer_bias,
        ),
    ),
    # End compositional population
    # Start compositional intervention
    # "intervention-summary": QAColumnConfig(
    #     # TODO (jason) compositional columns should not have a root question. This right
    #     # now is only used in logging. This should perhaps be `CompositionalColumnConfig`
    #     question="What is the intervention summary?",
    #     search_strategy=LMSearchStrategy(),
    #     answer_strategy=CompositionalAnswerStrategy(
    #         answer_bias=strong_pro_answer_bias,
    #         sub_questions={
    #             "Intervention": Column(type="predefined", value="intervention"),
    #             "Dose": Column(type="predefined", value="dose"),
    #             "Duration": Column(type="predefined", value="duration"),
    #         },
    #         create_aggregation_prompt=generate_intervention_summary_prompt,
    #         openai_stop=("Information used",),
    #     ),
    # ),
    # End compositional intervention
    "adherence": QAColumnConfig(
        question="What was the adherence, take-up, or coverage in the study?",
        search_strategy=LMSearchStrategy(),
        answer_strategy=InstructAnswerStrategy(
            answer_prefix="The adherence/take-up/coverage can be described as",
            answer_bias=pro_answer_bias,
        ),
    ),
    "timing": QAColumnConfig(
        question="In what time period did the study take place?",
        search_strategy=LMSearchStrategy(),
        answer_strategy=InstructAnswerStrategy(
            answer_prefix="The time period in which the study took place was",
            answer_bias=pro_answer_bias,
        ),
    ),
    "n_at_start": QAColumnConfig(
        question="How many participants began the study?",
        search_strategy=LMSearchStrategy(),
        answer_strategy=InstructAnswerStrategy(
            answer_prefix="The number of participants who began the study was",
            answer_bias=pro_answer_bias,
        ),
    ),
    "Age of participants": QAColumnConfig(
        question="\
What was the age of the participants in this paper? \
(e.g. children, 40-55, 20 +/- 2 years, middle-aged)",
        search_strategy=LMSearchStrategy(),
        answer_strategy=InstructAnswerStrategy(
            answer_prefix="The age of the participants in this paper was",
            answer_bias=pro_answer_bias,
        ),
    ),
    "age": QAColumnConfig(
        question="\
What was the age of the participants in this paper? \
(e.g. children, 40-55, 20 +/- 2 years, middle-aged)",
        search_strategy=LMSearchStrategy(),
        answer_strategy=InstructAnswerStrategy(
            answer_prefix="The age of the participants in this paper was",
            answer_bias=pro_answer_bias,
        ),
    ),
    "Placebo": QAColumnConfig(
        question="""What placebo (sham treatment) did the study use, if any?""",
        search_strategy=LMSearchStrategy(),
        answer_strategy=InstructAnswerStrategy(
            answer_prefix="The placebo used was",
            answer_bias=pro_answer_bias,
        ),
    ),
}
