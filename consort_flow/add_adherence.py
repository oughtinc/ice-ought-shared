# goals: 1. merge in adherence (allocation diverged)
import pandas as pd
from ice.recipes.consort_flow.types import ConsortFlow
from ice.contrib.ought_shared.utils import MyYAML
from ice.contrib.ought_shared.consort_flow.validate_schema import validate_schema

yaml = MyYAML()
yaml.indent(mapping=2, sequence=4, offset=2)

def sub_in_new_adherence(v1_row: pd.Series, adherence_df: pd.DataFrame) -> ConsortFlow:
    v1_flow: ConsortFlow = v1_row["flow_answer"]
    adherence_flow: ConsortFlow = adherence_df[adherence_df["document_id"] == v1_row["document_id"]].iloc[0]["flow_answer"]

    for v1_experiment in v1_flow.experiments:
        for v1_arm in v1_experiment.arms:
            try:
                v2_experiment = [v2_experiment for v2_experiment in adherence_flow.experiments if v2_experiment.name == v1_experiment.name][0]
            except IndexError:
                print(f"did not find experiment {v1_experiment.name} for document {v1_row['document_id']}")
            
            try:
                v2_arm = [v2_arm for v2_arm in v2_experiment.arms if v2_arm.name == v1_arm.name][0]

                v1_arm.received = v2_arm.received

                if v2_arm.received:
                    print(f"added received for arm {v1_arm.name} in experiment {v1_experiment.name} for document {v1_row['document_id']}")
            except IndexError:
                print(f"did not find arm {v1_arm.name} for document {v1_row['document_id']}")
    
    return v1_flow

if __name__ == "__main__":
    gs_df = pd.read_csv("gold_standards/gold_standards.csv")
    gs_df = gs_df[gs_df["question_short_name"].isin(["consort_flow", "consort_flow_with_adherence"])]
    validate_schema(gs_df)

    gs_df["flow_answer"] = gs_df["answer"].apply(lambda x: ConsortFlow.parse_obj(yaml.load(x)))

    v2_df = gs_df[gs_df["question_short_name"] == "consort_flow"].copy()
    adherence_df = gs_df[gs_df["question_short_name"] == "consort_flow_with_adherence"].copy()
    v2_df["consort_flow_v2"] = v2_df.apply(lambda row: sub_in_new_adherence(row, adherence_df), axis=1)

    v2_df["answer"] = v2_df["consort_flow_v2"].apply(lambda x: yaml.dump(x.dict()))

    validate_schema(v2_df)
    v2_df["question_short_name"] = "consort_flow_v2"
    v2_df.to_csv("ice/contrib/ought_shared/consort_flow/v2_flows.csv", index=False)