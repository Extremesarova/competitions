import pandas as pd


def column_filter(data: pd.DataFrame, filter_count: int) -> pd.DataFrame:

    column_count = (data != 0).count(axis=0)
    use_columns = column_count[column_count >= filter_count].index
    data = data[use_columns]

    return data


def group_data(
    data: pd.DataFrame,
    column: str = "cid",
    filter_count: int = 10,
    normed: bool = False,
) -> pd.DataFrame:

    df = data.groupby(["hash_id", column])["hash_id"].count()
    df = df.unstack().fillna(0).astype(int)
    df.columns = df.columns.map(lambda x: f"{column}-{x}")
    if filter_count > 0:
        df = column_filter(df, filter_count)
    if normed:
        df = df.divide(df.sum(axis=1), axis=0)

    return df
