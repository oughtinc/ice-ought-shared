from ice.recipe import Recipe, recipe
import pandas as pd
from ice.contrib.ought_shared.utils import script_run_time
from ice.contrib.anthropic.anthropic_qa import anthropic_qa_from_kwargs
from ice.utils import map_async
# from importlib import import_module

FILEPATH = "data/in_app_qa_results/qa_eval_output.csv"
RECIPE = anthropic_qa_from_kwargs

async def run_recipe_on_row(row: pd.Series, recipe_to_run: Recipe):
    return await recipe_to_run(**row)

async def run_over_csv(filepath: str, recipe: Recipe):
    validation_df = pd.read_csv(filepath).sample(n=20, random_state=42)
    validation_df = pd.concat([
        validation_df, pd.read_csv(filepath).sample(frac=1)[:10]
    ]).reset_index(drop=True)

    test_df = pd.read_csv(filepath)

    print(len(test_df))

    test_df = test_df[~(test_df["document_id"] + test_df["question_short_name"]).isin(validation_df["document_id"] + validation_df["question_short_name"])]

    print(len(test_df))

    results = await map_async(
        [row for _, row in test_df.iterrows()],
        lambda row: run_recipe_on_row(row, recipe),
        max_concurrency=5
    )
    results_df = pd.DataFrame(results)
    results_df.to_csv(f"data/{script_run_time} {recipe.__name__}.csv")

async def main():
    return await run_over_csv(FILEPATH, RECIPE)

recipe.main(main)
