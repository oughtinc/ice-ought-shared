from ice.contrib.ought_shared.utils import MyYAML
from ice.recipes.consort_flow.types import ConsortFlow
import pandas as pd

RAW_FILEPATH = "ice/contrib/ought_shared/consort_flow/parse_experiments_arms_gs/experiments.yaml"

yaml = MyYAML()

if __name__ == "__main__":
    file_lines = open(RAW_FILEPATH).readlines()
    
    current_paper_name = ""
    current_paper_raw_yaml = ""
    paper_raw_yamls = {}

    for line in [line for line in file_lines if line.strip() != "" ]:
        if ".pdf" in line:
            if current_paper_raw_yaml != "":
                paper_raw_yamls[current_paper_name] = current_paper_raw_yaml
            current_paper_name = line.strip()
            current_paper_raw_yaml = ""
        else:
            if line.startswith("- "):
                line = f'- "{line[2:]}"'
            current_paper_raw_yaml += "\n" + line
    
    paper_raw_yamls[current_paper_name] = current_paper_raw_yaml

    paper_jsons = {paper_name: yaml.load(paper_raw_yaml) for paper_name, paper_raw_yaml in paper_raw_yamls.items()}

    paper_data = {paper_name: (
        ConsortFlow.parse_obj({
            "experiments": paper_json["experiments"],
        }),
        paper_json["supporting quotes"]
    ) for paper_name, paper_json in paper_jsons.items()}

    paper_rows = []
    
    for paper_name, paper_data in paper_data.items():
        paper_row = {
            "document_id": paper_name,
            "answer": yaml.dump(paper_data[0].dict()),
            "question_short_name": "consort_flow_v2",
        }

        for i, quote in enumerate(paper_data[1]):
            paper_row[f"quote_{i + 1}"] = quote

        paper_rows.append(paper_row)
    
    paper_rows_df = pd.DataFrame(paper_rows)
    paper_rows_df.to_csv("ice/contrib/ought_shared/consort_flow/parse_experiments_arms_gs/30_new_papers.csv", index=False)