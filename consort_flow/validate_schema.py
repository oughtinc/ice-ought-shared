import pandas as pd
from ice.recipes.consort_flow.types import ConsortFlow
import yaml
from pydantic import ValidationError
from pprint import pprint

gs_df = pd.read_csv("gold_standards/gold_standards.csv")

gs_df = gs_df[gs_df["question_short_name"] == "consort_flow"]

for _, row in gs_df.iterrows():
    answer_yaml = yaml.full_load(row["answer"])
    try:
        ConsortFlow.parse_obj(answer_yaml)
    except ValidationError as e:
        print("validation error for ", row["document_id"], e)
        pprint(answer_yaml)