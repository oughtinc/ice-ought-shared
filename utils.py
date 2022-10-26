import pandas as pd

def reorder_columns(df: pd.DataFrame, ordered_columns: list[str]) -> pd.DataFrame:
    rest_columns = [column for column in df.columns if column not in ordered_columns]
    columns = ordered_columns + rest_columns
    return df[columns]