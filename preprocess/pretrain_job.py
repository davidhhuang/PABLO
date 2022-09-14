import logging
import os
import pickle

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from smart_open import open
from tqdm import tqdm

tqdm.pandas()

from preprocess.common import (
    S3_OUTPUT_PREFIX,
    S3_PRETRAIN_PATH,
    chunked_melt,
    code_type_grouper,
    temporal_grouper,
    truncate_diagnosis_codes,
)

CHUNK_SIZE = int(1e6)


def encode_column(
    df: pd.DataFrame, column_name: str, encoder_name: str
) -> pd.DataFrame:
    """Encode a column using the provided encoder."""
    vocab = {
        feat: i for i, feat in enumerate(sorted(df[column_name].unique()))
    }

    # Coalesce rows into lists of tokens
    df = df.groupby(["ID"])[column_name].progress_apply(list).reset_index()

    # Convert to numeric features
    df[column_name] = df[column_name].progress_apply(
        lambda features: list(map(lambda x: vocab[x], features))
    )

    # Save encoder
    output_path = os.path.join(
        S3_OUTPUT_PREFIX, f"output/pretrain_v2/encoders/{encoder_name}.pickle"
    )
    with open(output_path, "wb") as encoder_file:
        pickle.dump(vocab, encoder_file, protocol=pickle.HIGHEST_PROTOCOL)
    logger.info(f"Saved encoder to: {output_path}")

    return df, vocab


if __name__ == "__main__":
    logger = logging.getLogger("preprocessing")
    logger.addHandler(logging.StreamHandler())
    logger.setLevel(logging.INFO)

    # Load data
    df = pd.read_csv(S3_PRETRAIN_PATH)
    logger.info("Loaded data")

    # Convert procedure columns to integer types
    proc_cols = [name for name in df.columns if "ProcCCS" in name]
    df[proc_cols] = df[proc_cols].fillna(-1).astype(int)

    # Truncate diagnosis codes to 3 characters
    df = truncate_diagnosis_codes(df)
    logger.info("Truncated ICD codes")

    # Load mapping of ICD10 codes to clinical category
    dx_classes_df = pd.read_csv("s3://huangdh/datasets/ccs_icd10_dx.csv")
    dx_classes_df["Code_10_trunc_3"] = dx_classes_df["ICD10"].apply(
        lambda x: x[:3]
    )
    dx_classes_df = dx_classes_df[["Code_10_trunc_3", "CCS"]].copy()

    # Get most common category for each ICD code
    dx_classes_df = (
        dx_classes_df.groupby("Code_10_trunc_3")["CCS"]
        .agg(lambda x: x.mode()[0])
        .reset_index()
    )

    # Create vocab
    last_visit_dx_category_vocab = {
        feat: i for i, feat in enumerate(sorted(dx_classes_df["CCS"].unique()))
    }

    # Encode column
    dx_classes_df["CCS"] = dx_classes_df["CCS"].apply(
        lambda x: last_visit_dx_category_vocab[x]
        if x in last_visit_dx_category_vocab
        else -1
    )

    # Save encoder
    output_path = os.path.join(
        S3_OUTPUT_PREFIX,
        f"output/pretrain_v2/encoders/last_visit_dx_category_vocab.pickle",
    )
    with open(output_path, "wb") as encoder_file:
        pickle.dump(
            last_visit_dx_category_vocab,
            encoder_file,
            protocol=pickle.HIGHEST_PROTOCOL,
        )
    logger.info(f"Saved encoder to: {output_path}")

    # Get category of most recent visit prinicipal diagnoses
    last_visit_dx_prin = df[df["Visit_no"] == df["Visits"]][
        ["ID", "Dx10_prin"]
    ]
    last_visit_dx_prin = last_visit_dx_prin.merge(
        dx_classes_df,
        how="left",
        left_on="Dx10_prin",
        right_on="Code_10_trunc_3",
    )
    last_visit_dx_prin = last_visit_dx_prin.drop(
        ["Dx10_prin", "Code_10_trunc_3"], axis=1
    )
    last_visit_dx_prin.columns = ["ID", "next_visit_dx_category"]

    # Add next visit categories and remove next visits
    df = df.merge(last_visit_dx_prin, how="left", on="ID")
    df = df[df.Visit_no != df.Visits].reset_index()
    df["next_visit_dx_category"] = df["next_visit_dx_category"].fillna(-1)
    df["next_visit_dx_category"] = df["next_visit_dx_category"].astype(
        np.int16
    )
    logger.info("Added column for next visit diagnosis category")

    # Convert to ages by decade
    df["Age"] = df["Age"].fillna(-1)
    df["Age"] = df["Age"].astype(int)
    df["Age"] = df["Age"].progress_apply(lambda x: x // 10)
    logger.info("Converted age to decade")

    # Back calculate days from prediction
    df["Days_since_last_visit"] = (
        df.groupby("ID")["Days_since_last_visit"]
        .shift(-1)
        .fillna(0)
        .astype(np.int16)
    )
    df["Days_from_prediction"] = (
        df.loc[::-1, "Days_since_last_visit"].groupby(df["ID"]).cumsum()
    )
    logger.info("Back calculated days from prediction point")

    # Drop unused columns
    df = df.drop(
        [
            "index",
            "Visits",
            "Type",
            "Days_since_last_visit",
            "Type",
            "Pt_zip_inc_qrtl",
            "Adm_LOS",
            "Visit_no",
        ],
        axis=1,
    )

    # Group time frames
    df["time_from_prediction"] = df["Days_from_prediction"].progress_apply(
        temporal_grouper
    )
    df = df.drop("Days_from_prediction", axis=1)
    logger.info("Grouped time frames")

    # Melt dataframe
    df = chunked_melt(
        df,
        chunk_size=CHUNK_SIZE,
        id_vars=[
            "ID",
            "time_from_prediction",
            "next_visit_dx_category",
            "Age",
            "Sex",
            "Race",
            "Payer",
        ],
    )
    df.rename(columns={"variable": "code_type", "value": "code"}, inplace=True)
    logger.info(
        "Melted dataframe by chunks, dropped nan and -1 values, and recombined"
    )

    # Group principal and auxiliary codes together
    df["code_type"] = df["code_type"].progress_apply(code_type_grouper)

    # Sort by visit time
    df.sort_values(["ID", "time_from_prediction"], inplace=True)
    df = df.reset_index(drop=True)
    df["code"] = df["code"].astype(str)
    logger.info("Sorted by ID and time columns")

    # Encode columns
    feature_token_df, feature_vocab = encode_column(
        df, "code", "feature_vocab"
    )
    time_token_df, _ = encode_column(df, "time_from_prediction", "time_vocab")
    code_type_token_df, _ = encode_column(df, "code_type", "type_vocab")
    logger.info("Created encoded columns")

    # Get all patient IDs and demographics
    df = df.drop_duplicates(subset=["ID"], keep="first")[
        ["ID", "next_visit_dx_category", "Age", "Sex", "Race", "Payer"]
    ]
    df = chunked_melt(
        df, chunk_size=CHUNK_SIZE, id_vars=["ID", "next_visit_dx_category"]
    )
    logger.info(
        "Melted dataframe by chunks, dropped nan and -1 values, and recombined"
    )

    # Create unique column to be used by encoder
    df["demographics"] = df.progress_apply(
        lambda row: row["variable"] + "_" + str(row["value"]), axis=1
    )
    df = df.drop(["variable", "value"], axis=1)

    last_encoding_number = max(feature_vocab.values())
    for i, demo in enumerate(sorted(df.demographics.unique())):
        next_index = last_encoding_number + i + 1
        feature_vocab[demo] = next_index

    # Coalesce rows into lists of tokens
    df = (
        df.groupby(["ID", "next_visit_dx_category"])["demographics"]
        .progress_apply(list)
        .reset_index()
    )

    # Convert to numeric features
    df["demographics"] = df["demographics"].progress_apply(
        lambda features: list(map(lambda x: feature_vocab[x], features))
    )
    logger.info("Encoded demographics")

    # Save feature encoder again, overwriting
    output_path = os.path.join(
        S3_OUTPUT_PREFIX, f"output/pretrain_v2/encoders/feature_vocab.pickle"
    )
    with open(output_path, "wb") as encoder_file:
        pickle.dump(
            feature_vocab, encoder_file, protocol=pickle.HIGHEST_PROTOCOL
        )
    logger.info(f"Saved encoder to: {output_path}")

    # Merge token lists into single dataframe
    merged_df = pd.merge(feature_token_df, time_token_df, on="ID", how="inner")
    merged_df = pd.merge(merged_df, code_type_token_df, on="ID", how="inner")
    df = pd.merge(merged_df, df, on="ID", how="inner")
    logger.info("Merged coalesced rows")

    # Save preprocessed dataset
    # Split will be determined in postprocessing to avoid contamination
    dataset_output_path = os.path.join(
        S3_OUTPUT_PREFIX, "output/pretrain_v2/dataset.parquet"
    )
    df.to_parquet(dataset_output_path, index=False)
    logger.info(f"Uploaded dataset set to: {dataset_output_path}")

    logger.info("Finished job.")
