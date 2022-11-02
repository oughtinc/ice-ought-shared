NA_PHRASE = "not mentioned in the title or excerpt"

def make_excerpt_prompt(
    *,
    qa_question: str,
    answer_prefix: str | None,
    title: str,
    excerpt: str,
) -> str:
    combined_na_phrase = (
        f"The answer to the question is {NA_PHRASE}"
        if answer_prefix is None
        else f"{answer_prefix} {NA_PHRASE}"
    )

    full_answer_prefix = (
        "Answer:" if answer_prefix is None else f"Answer: {answer_prefix}"
    )

    return f"""Answer the question "{qa_question}" based on the excerpt from a research paper. \
Try to answer, but say "{combined_na_phrase}" if you don't know how to answer. \
Include everything that the paper excerpt has to say about the answer. \
Make sure everything you say is supported by the excerpt. \
The excerpt may cite other papers; \
answer about the paper you're reading the excerpt from, not the papers that it cites. \
Answer in one phrase or sentence:

Paper title: {title}

Paper excerpt: {excerpt}

Question: {qa_question}

{full_answer_prefix}"""