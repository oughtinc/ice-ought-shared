from ice.recipe import Recipe, recipe, FunctionBasedRecipe
import pandas as pd
from ice.contrib.ought_shared.utils import script_run_time
from ice.contrib.anthropic.anthropic_qa import anthropic_qa_from_kwargs
from ice.utils import map_async

async def run_recipe_on_row(row: pd.Series, recipe_to_run: Recipe):
    return await recipe_to_run(**row)

async def run_over_csv(df: pd.DataFrame, recipe: Recipe, test: bool = False):
    if test == True:
        df = df[:3]

    results = await map_async(
        [row for _, row in df.iterrows()],
        lambda row: run_recipe_on_row(row, recipe),
        max_concurrency=5
    )
    
    results_df = pd.DataFrame(results)
    results_df.to_csv(f"data/{script_run_time} {recipe.__name__}.csv")

async def run_over_csv_cli(df: pd.DataFrame="", recipe: FunctionBasedRecipe="", test: bool=False):
    return await run_over_csv(df=df, recipe=recipe, test=test)

recipe.main(run_over_csv_cli)
