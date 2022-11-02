import pandas as pd
import sys
from ruamel.yaml import YAML
from ruamel.yaml.compat import StringIO

def reorder_columns(df: pd.DataFrame, ordered_columns: list[str]) -> pd.DataFrame:
    rest_columns = [column for column in df.columns if column not in ordered_columns]
    columns = ordered_columns + rest_columns
    return df[columns]

# from https://yaml.readthedocs.io/en/latest/example.html
class MyYAML(YAML):
    def dump(self, data, stream=None, **kw):
        inefficient = False
        if stream is None:
            inefficient = True
            stream = StringIO()
        YAML.dump(self, data, stream, **kw)
        if inefficient:
            return stream.getvalue()