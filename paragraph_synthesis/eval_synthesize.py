import pandas as pd

from ice.recipe import Recipe
from ice.recipe import recipe
from ice.contrib.ought_shared.eval.eval_vs_gs import run_over_gs
from ice.contrib.ought_shared.paragraph_synthesis.synthesize_compositional import (
    synthesize_compositional,
)
from ice.contrib.ought_shared.utils import reorder_columns
from ice.recipes.abstract_qa import Abstract
from pathlib import Path
import json

RECIPE_TO_RUN = synthesize_compositional
GS_FILENAME = "ice/contrib/ought_shared/paragraph_synthesis/paragraph_synthesis_gs.csv"
SPLITS = ["validation"]

DATA_PATH = Path("ice/contrib/ought_shared/paragraph_synthesis/data/")

async def eval_synthesize():
    gs_df = pd.read_csv(GS_FILENAME)
    gs_df = gs_df[gs_df["is_gs"] == True].reset_index()
    gs_df["abstracts"] = gs_df["papers"].apply(lambda papers: [
        Abstract(
            title=paper["title"],
            authors=paper["authors"],
            year=paper["year"],
            text=paper["abstract"],
        )
        for paper in json.loads(papers)
    ])

    answers_df = await run_over_gs(RECIPE_TO_RUN, gs_df, SPLITS)
    answers_df["question"] = answers_df["document_id"]
    gs_df.columns = [f"{column}_gs" for column in gs_df.columns]
    gs_df["question"] = gs_df["question_gs"]
    merged_df = answers_df.merge(
        gs_df, on="question", how="left", suffixes=("_answer", "_gs")
    )
    merged_df = merged_df.sort_values(["question", "technique"])
    rating_cols_to_add = [
        "citedness (0-1)",
        "hallucination (0 = none, 2 = substantive)",
        "answer_rating",
        "failure_modes",
        "notes",
    ]
    for col in rating_cols_to_add:
        merged_df[col] = ""
    merged_df = reorder_columns(
        merged_df,
        [
            "technique",
            "question",
            "answer",
            "answer_gs",
        ]
        + rating_cols_to_add
        + [
            "citation_1_gs",
            "abstract_1_gs",
            "paper_1_answer_gs",
            "citation_2_gs",
            "abstract_2_gs",
            "paper_2_answer_gs",
            "citation_3_gs",
            "abstract_3_gs",
            "paper_3_answer_gs",
            "citation_4_gs",
            "abstract_4_gs",
            "paper_4_answer_gs",
        ],
    )

    DATA_PATH.mkdir(parents=True, exist_ok=True)

    merged_df.to_csv(DATA_PATH / f"{RECIPE_TO_RUN.__name__}_eval.csv", index=False)

    return merged_df


recipe.main(eval_synthesize)
