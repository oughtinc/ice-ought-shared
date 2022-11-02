import pandas as pd
from ice.recipes.consort_flow.types import ConsortFlow
from ice.contrib.ought_shared.utils import MyYAML
from pydantic import ValidationError
from pprint import pprint

yaml = MyYAML()

def validate_schema(consort_gs_df: pd.DataFrame) -> None:
    passed_validation = True
    
    for _, row in consort_gs_df.iterrows():
        # try:
        answer_yaml = yaml.load(row["answer"])
        # except ScannerError as e:
        #     passed_validation = False
        #     print("scanner error for ", row["document_id"], row["question_short_name"], e)
        try:
            ConsortFlow.parse_obj(answer_yaml)
        except ValidationError as e:
            passed_validation = False
            print("validation error for ", row["document_id"], row["question_short_name"], e)
            pprint(answer_yaml)
        
    if not passed_validation:
        raise ValueError("Schema validation failed")
    
if __name__ == "__main__":
    gs_df = pd.read_csv("gold_standards/gold_standards.csv")
    gs_df = gs_df[gs_df["question_short_name"].isin(["consort_flow", "consort_flow_with_adherence"])]
    validate_schema(gs_df)