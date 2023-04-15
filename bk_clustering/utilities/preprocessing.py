import pandas as pd


def preprocessing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocessing pipeline for cluster dataframe.
    Includes handlic strings and numeric values.

    Args:
        df: pd.DataFrame - raw dataset

    Return:
        df: pd.DataFrame - preprocessed pandas dataFrame
    """
    df = df.apply(pd.to_numeric, errors="ignore")
    df = string_handling(df)
    df = numeric_handling(df)
    df = df.dropna(axis=1)
    columns = [x for x in df.columns if x != "class"] + ["class"]
    return df[columns]


def numeric_handling(df: pd.DataFrame) -> pd.DataFrame:
    """
    Numeric handling method. Converting string to numeric where possible

    Args:
        df - pd.DataFrame: cluster dataframe

    Return:
        df - pd.DataFrame: modified cluster dataframe
    """
    numeric_df = df.apply(pd.to_numeric, errors="coerce")
    numeric_series = numeric_df.isnull().all()
    numeric_df = numeric_df[numeric_series[~numeric_series].index]

    # Define logic separating class. Possible bug {1,2,3,"noise","target"}

    if "class" in df:
        if "class" not in numeric_df.columns:
            df = pd.concat([numeric_df, df["class"]], axis=1)
        else:
            df = numeric_df
        df["class"].fillna(-1, inplace=True)
    else:
        df = numeric_df
    return df


def string_handling(
    df: pd.DataFrame,
    decoder: str = "ascii",
    create_dummy_columns: bool = True,
    thresh_dummies: int = 5,
) -> pd.DataFrame:
    """
    String handling method. Decoding and replacing string values with one-hot encoding alternative

    Args:
        df - pd.DataFrame: cluster dataframe
        decoder - str: decoder --> ascii, utf-8, etc
        create_dummy_columns - bool: should one-hot encoding to be applied
        thresh_dummies - int: if create_dummy_columns is True, maximum number of unique values to be applied.
        Reason is to avoid creating to sparse matrix

    Return:
        df - pd.DataFrame: modified cluster dataframe
    """
    for col in df.select_dtypes("object"):
        # check for binary values
        _col = df[col].str.decode(decoder)
        if not _col.isnull().all():
            df[col] = _col

        # dummy encoding
        if (
            create_dummy_columns
            and col != "class"
            and df[col].nunique() <= thresh_dummies
        ):
            _dummies = pd.get_dummies(df[col])
            _dummies.columns = [f"{col}_{value}" for value in _dummies.columns]
            df = pd.concat([df, _dummies], axis=1).drop([col], axis=1)
    return df


if __name__ == "__main__":
    pass
