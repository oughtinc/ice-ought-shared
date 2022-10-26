import json

from functools import partial

import transformers

from ice.recipe import recipe
from ice.contrib.ought_shared.paragraph_synthesis.synthesize import _get_reference
from ice.contrib.ought_shared.paragraph_synthesis.synthesize import Abstract
from ice.contrib.ought_shared.paragraph_synthesis.synthesize import num_tokens
from ice.contrib.ought_shared.paragraph_synthesis.synthesize import synthesize_from_df

tokenizer = transformers.GPT2Tokenizer.from_pretrained("gpt2")

MAX_TOKENS = 2040
FT_MODEL = "davinci:ft-ought-experiments:synthesisv2-2022-10-17-23-02-55"

PAPERS = """Paper {i}: {title}
Reference: {reference}
Abstract: {abstract}"""

PROMPT = """This is an article about how to write an ideal academic summary.

Overall question: {question}
Let's write a summary step by step.

Below are 4 relevant papers.

{paper_str}

Let's summarize each paper individually first.

Summaries (starting with paper 1):"""


def truncate(string: str, max_tokens: int) -> str:
    return tokenizer.decode(tokenizer.encode(string)[:max_tokens])


def n_tokens(string: str) -> int:
    return len(tokenizer.encode(string))


def _create_paper_str(
    titles: list[str],
    citations: list[str],
    abstracts: list[str],
    abstract_max_tokens: int,
) -> str:
    # Use davinci:ft-ought-experiments:synthesisv2-2022-10-17-23-02-55
    return "\n\n".join(
        [
            PAPERS.format(
                i=i + 1,
                title=title,
                reference=citation,
                abstract=truncate(abstract, abstract_max_tokens),
            )
            for i, (title, citation, abstract) in enumerate(
                zip(titles, citations, abstracts)
            )
        ]
    )


def _create_prompt_ft(
    query: str, titles: list[str], abstracts: list[str], citations: list[str]
) -> str:
    # Use fine-tuned model
    abstract_max_tokens = 1000
    paper_str = _create_paper_str(
        titles, citations, abstracts, abstract_max_tokens=abstract_max_tokens
    )
    prompt = PROMPT.format(question=query, paper_str=paper_str)

    while n_tokens(prompt) > 1700 and abstract_max_tokens > 0:
        paper_str = _create_paper_str(
            titles, citations, abstracts, abstract_max_tokens=abstract_max_tokens
        )
        prompt = PROMPT.format(question=query, paper_str=paper_str)
        abstract_max_tokens -= 10

    return prompt


async def synthesize_ft(
    question: str, abstracts: list[Abstract], create_prompt_fn=_create_prompt_ft
) -> str:
    prompt = create_prompt_fn(
        query=question,
        titles=[abstract.title for abstract in abstracts],
        abstracts=[abstract.text for abstract in abstracts],
        citations=[
            _get_reference(abstract.authors, abstract.year) for abstract in abstracts
        ],
    )

    remaining_tokens = MAX_TOKENS - num_tokens(prompt)

    completion = await recipe.agent(FT_MODEL).complete(
        prompt=prompt, max_tokens=remaining_tokens, stop="<|endoftext|>"
    )

    return completion


synthesize_ft_from_df = partial(synthesize_from_df, synthesize_fn=synthesize_ft)
synthesize_ft_from_df.__name__ = "synthesize_ft_from_df"
