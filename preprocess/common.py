"""Utility functions and constants used for data preprocessing"""

import pandas as pd
from tqdm import tqdm

# Dataset paths
S3_BUCKET = "[redacted]"
S3_PRETRAIN_PATH = "[redacted]"
S3_DEVELOPMENT_PATH = "[redacted]"
S3_EXTERNAL_VALIDATION_PATH = "[redacted]"
S3_OUTPUT_PREFIX = "[redacted]"


def temporal_grouper(days: int) -> int:
    """Group number of days to binned values."""
    if days == 0:
        return "0"
    elif days < 30:
        return "1_within_0_30"
    elif days < 90:
        return "2_within_30_90"
    elif days < 180:
        return "3_within_90_180"
    elif days < 360:
        return "4_within_180_360"
    elif days < 720:
        return "5_within_360_720"
    else:
        return "6_over_720"


def code_type_grouper(feature: str) -> str:
    """Group code types to either principal or auxiliary."""
    if "prin" in feature:
        return feature
    elif "Dx10" in feature:
        return "Dx10_aux"
    elif "EC10" in feature:
        return "EC10_aux"
    elif "ProcCCS" in feature:
        return "ProcCCS_aux"
    else:
        return feature


def truncate_diagnosis_codes(df: pd.DataFrame) -> pd.DataFrame:
    """Truncate diagnosis codes to 3 characters."""
    TRUNCATE_TO = 3
    dx_columns = [
        name
        for name in df.columns
        if any(sub in name for sub in ("Dx10", "EC10"))
    ]
    df[dx_columns] = df[dx_columns].astype(str)
    for dx_col in dx_columns:
        df[dx_col] = df[dx_col].apply(lambda x: x[:TRUNCATE_TO])

    return df


def chunked_melt(df, chunk_size, id_vars):
    """Melt dataframe by chunks to decrease memory utilization."""
    melted_list = []

    for i in tqdm(range(0, len(df), chunk_size)):
        # Melt dataframe chunk
        df_chunk = pd.melt(df.iloc[i : i + chunk_size], id_vars=id_vars)

        # Drop missing diagnoses and procedures
        df_chunk = df_chunk[(df_chunk.value != "nan") & (df_chunk.value != -1)]
        melted_list.append(df_chunk)

    # Recombine melted chunks
    df = pd.concat(melted_list)

    return df
