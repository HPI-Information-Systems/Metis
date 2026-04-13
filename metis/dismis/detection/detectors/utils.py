from typing import Dict

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

from metis.dismis.utils.datetime import datetime_to_numeric
from metis.dismis.utils.types import COLUMN_TYPES


def force_numeric(series: pd.Series) -> pd.Series:
    """
    Attempts to convert a pandas Series to numeric, setting unconvertible values to NaN.
    """
    return pd.to_numeric(series, errors="coerce")


def split_mixed_column(col: pd.Series, colname: str) -> pd.DataFrame:
    df_out = pd.DataFrame(index=col.index)

    # Create null indicator (including real NaNs)
    df_out[f"{colname}_null"] = col.isnull().astype(int)

    # Separate numeric and non-numeric values
    numeric_values = pd.to_numeric(col, errors="coerce").mask(np.isinf)

    df_out[f"{colname}_num"] = numeric_values.fillna(
        0
    )  # Fill NaNs with 0 for numeric values
    df_out[f"{colname}_str"] = (numeric_values.isnull() & ~col.isnull()).astype(int)

    return df_out[[f"{colname}_num", f"{colname}_null", f"{colname}_str"]]


def split_datetime_column(col: pd.Series, colname: str) -> pd.DataFrame:
    df_out = pd.DataFrame(index=col.index)

    # Create null indicator (including real NaNs)
    df_out[f"{colname}_null"] = col.isnull().astype(int)

    # Separate numeric and non-numeric values
    df_out[f"{colname}_num"], _, _ = datetime_to_numeric(col)
    df_out[f"{colname}_num"] = df_out[f"{colname}_num"].mask(np.isinf)
    df_out[f"{colname}_str"] = (
        df_out[f"{colname}_num"].isnull() & ~col.isnull()
    ).astype(int)
    df_out[f"{colname}_num"].fillna(0, inplace=True)

    return df_out[[f"{colname}_num", f"{colname}_null", f"{colname}_str"]]


def encode_dataset(
    df_detect: pd.DataFrame,
    column_types: Dict[str, COLUMN_TYPES],
    embeddings: Dict[str, pd.DataFrame],
    normalize=True,
    text_embedding_dim: int | None = None,
    ohe_max_categories: int | None = None,
    ohe_min_frequency: int | None = None,
    ohe_dtype=np.float32,
) -> pd.DataFrame:
    # Make a copy to avoid modifying the original DataFrame
    df_detect = df_detect.copy()
    scaler = MinMaxScaler()
    for column in column_types.keys():
        if column_types[column] == "categorical":
            encoder_kwargs: Dict = {
                "handle_unknown": "infrequent_if_exist",
            }
            if ohe_max_categories is not None:
                encoder_kwargs["max_categories"] = ohe_max_categories
            if ohe_min_frequency is not None:
                encoder_kwargs["min_frequency"] = ohe_min_frequency
            # try:
            #     encoder = OneHotEncoder(sparse_output=True, **encoder_kwargs)
            # except TypeError:
            #     encoder = OneHotEncoder(sparse=True, **encoder_kwargs)
            encoder = OneHotEncoder(**encoder_kwargs)
            encoded = encoder.fit_transform(df_detect[[column]])
            feature_names = encoder.get_feature_names_out([column])
            if normalize:
                encoded = encoded * (1 / (2**0.5))
            ohe_features = pd.DataFrame(
                encoded.astype(ohe_dtype), columns=feature_names
            )
            df_detect = pd.concat(
                [df_detect.drop(columns=[column]), ohe_features], axis=1
            )
        elif column_types[column] in ["numeric", "date"]:
            if column_types[column] == "date":
                numeric_features = split_datetime_column(df_detect[column], column)
            else:
                numeric_features = split_mixed_column(df_detect[column], column)
            if numeric_features[f"{column}_null"].sum() == 0:
                numeric_features.drop(columns=[f"{column}_null"], inplace=True)
            if numeric_features[f"{column}_str"].sum() == 0:
                numeric_features.drop(columns=[f"{column}_str"], inplace=True)
            numeric_features[f"{column}_num"] = scaler.fit_transform(
                numeric_features[[f"{column}_num"]]
            )
            df_detect = pd.concat(
                [df_detect.drop(columns=[column]), numeric_features], axis=1
            )
        elif column_types[column] == "text":
            if column in embeddings:
                emb = embeddings[column]
                if text_embedding_dim is not None and emb.shape[1] > text_embedding_dim:
                    emb = emb[:, :text_embedding_dim]
                # L2-normalize each row, then scale by 1/2 so max pairwise
                # Euclidean distance equals 1, matching numeric and scaled OHE features.
                emb = emb.astype(np.float32)
                norms = np.linalg.norm(emb, axis=1, keepdims=True)
                emb = emb / np.maximum(norms, 1e-8) * 0.5
                emb_df = pd.DataFrame(
                    emb, columns=[f"{column}_emb{i}" for i in range(emb.shape[1])]
                )
                df_detect = pd.concat(
                    [df_detect.drop(columns=[column]), emb_df], axis=1
                )
            else:
                raise ValueError(f"No embeddings provided for text column {column}.")
        else:
            raise ValueError(
                f"Unsupported type for column {column}: {column_types[column]}"
            )
    return df_detect.astype(np.float32)
