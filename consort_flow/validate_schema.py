import pandas as pd
from ice.recipes.consort_flow.types import ConsortFlow
import yaml

gs_df = pd.read_csv("gold_standards/gold_standards.csv")

gs_df = gs_df[gs_df["question_short_name"] == "consort_flow"]

for _, row in gs_df.iterrows():
    ConsortFlow.parse_obj(yaml.full_load(row["answer"]))