from typing import List, Tuple
import pandas as pd


def get_time_features(
    dates: pd.Series, times: pd.Series
) -> Tuple[pd.DataFrame, List[str]]:
    hour = times.dt.hour
    weekday = dates.dt.weekday + 1

    morning = ((hour >= 7) & (hour <= 11)).astype("int")
    day = ((hour >= 12) & (hour <= 18)).astype("int")
    evening = ((hour >= 19) & (hour <= 23)).astype("int")
    night = ((hour >= 0) & (hour <= 6)).astype("int")

    weekend = ((weekday >= 6) & (weekday <= 7)).astype("int")

    feature_names = ["hour", "morning", "day", "evening", "night", "weekday", "weekend"]

    datetime_features = pd.DataFrame(
        list(
            zip(
                hour,
                morning,
                day,
                evening,
                night,
                weekday,
                weekend,
            )
        ),
        columns=feature_names,
    )
    return datetime_features, feature_names


def time_features(
    df: pd.DataFrame,
    column: str = "hour",
    prefix: str = "reg_hour",
    add_total_count: bool = True,
):

    df = df.drop_duplicates(["hash_id", column, "fulldate"])
    res = df.groupby(["hash_id", column]).size().unstack().fillna(0)
    res.columns = res.columns.map(lambda x: f"{prefix}_{x}")
    total_count = res.sum(axis=1)
    res = res.divide(total_count, axis=0)
    if add_total_count:
        res[f"{prefix}_total"] = total_count

    return res
