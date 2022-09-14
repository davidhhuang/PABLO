import argparse
import logging
import os
import pickle

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from smart_open import open
from tqdm import tqdm

from preprocess.jobs.common import (S3_DEVELOPMENT_PATH,
                                    S3_EXTERNAL_VALIDATION_PATH,
                                    S3_OUTPUT_PREFIX, chunked_melt,
                                    code_type_grouper, temporal_grouper,
                                    truncate_diagnosis_codes)

tqdm.pandas()


CHUNK_SIZE = int(1e6)


def encode_column_using_encoder(
    df: pd.DataFrame, column_name: str, encoder_name: str
) -> pd.DataFrame:
    """Encode a column using the given encoder and return the dataframe."""
    # Load encoder vocab
    with open(
        os.path.join(
            S3_OUTPUT_PREFIX, f"output/pretrain/encoders/{encoder_name}.pickle"
        ),
        "rb",
    ) as encoder_file:
        vocab = pickle.load(encoder_file)

    # Coalesce rows into lists of tokens
    df = df.groupby(["ID"])[column_name].progress_apply(list).reset_index()

    # Convert to numeric features, using -1 if previously unseen
    df[column_name] = df[column_name].apply(
        lambda features: list(
            map(lambda x: vocab[x] if x in vocab else -1, features)
        )
    )

    return df


def label_nat(sub_df: pd.DataFrame, nat: str) -> str:
    """Determine if patient will have NAT with 365 days of index visit."""
    days_to_nat = sub_df[f"{nat}_PredictionToEvent"].iloc[0]
    if np.isnan(days_to_nat) or days_to_nat > 365:
        return 0
    else:
        return 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--job",
        type=str,
        required=True,
        help='job type is either "finetune" or "external"',
    )
    args = parser.parse_args()

    if args.job not in ("finetune", "external"):
        raise Exception('Job must be either "finetune" or "external"')

    logger = logging.getLogger("preprocessing")
    logger.addHandler(logging.StreamHandler())
    logger.setLevel(logging.INFO)

    # Load data
    if args.job == "finetune":
        df = pd.read_csv(S3_DEVELOPMENT_PATH)
    elif args.job == "external":
        df = pd.read_csv(S3_EXTERNAL_VALIDATION_PATH)
    else:
        raise Exception('Job must be either "finetune" or "external"')
    logger.info("Loaded data")

    # Subset columns to be the same as pretrain dataset
    cols_to_use = ["ID", "Age", "Sex", "Race", "Payer", "DaysFromPrediction"]
    cols_to_use += [
        name
        for name in df.columns
        if any(
            sub in name
            for sub in [
                "Dx10",
                "EC10",
                "ProcCCS",
                "NAT_reis_broad",
                "NAT_reis_narrow",
            ]
        )
    ]
    df = df[cols_to_use].copy()
    logger.info("Subset columns")

    # Convert procedure columns to integer types
    proc_cols = [name for name in df.columns if "ProcCCS" in name]
    df[proc_cols] = df[proc_cols].fillna(-1).astype(int)

    # Convert other float columns to integers
    df["DaysFromPrediction"] = df["DaysFromPrediction"].astype(int)

    # Truncate all diagnosis codes to 3 characters
    df = truncate_diagnosis_codes(df)
    logger.info("Truncated ICD codes")

    # Convert to age by decade
    df["Age"] = df["Age"].fillna(-1)
    df["Age"] = df["Age"].astype(int)
    df["Age"] = df["Age"].apply(lambda x: x // 10)
    logger.info("Converted age to decade")

    # Add NAT label
    df_labels_by_id = (
        df.groupby("ID")[
            [
                "ID",
                "DaysFromPrediction",
                "NAT_reis_broad",
                "NAT_reis_broad_PredictionToEvent",
            ]
        ]
        .progress_apply(lambda x: label_nat(x, "NAT_reis_broad"))
        .reset_index()
    )
    df_labels_by_id.columns = ["ID", "NAT_reis_broad_label"]
    df = pd.merge(df, df_labels_by_id, on="ID", how="left")
    logger.info("Labeled NAT_reis_broad")

    # Add NAT narrow label
    df_labels_by_id = (
        df.groupby("ID")[
            [
                "ID",
                "DaysFromPrediction",
                "NAT_reis_narrow",
                "NAT_reis_narrow_PredictionToEvent",
            ]
        ]
        .progress_apply(lambda x: label_nat(x, "NAT_reis_narrow"))
        .reset_index()
    )
    df_labels_by_id.columns = ["ID", "NAT_reis_narrow_label"]
    df = pd.merge(df, df_labels_by_id, on="ID", how="left")
    logger.info("Labeled NAT_reis_narrow")

    # Exclude visits after prediction point
    df = df[df["DaysFromPrediction"] <= 0].copy()

    # Group time frames
    df["DaysFromPrediction"] = df["DaysFromPrediction"].progress_apply(
        lambda x: -1 * x
    )
    df["time_from_prediction"] = df["DaysFromPrediction"].progress_apply(
        temporal_grouper
    )
    logger.info("Grouped time frames")

    # Remove unused columns
    cols_to_remove = [
        "DaysFromPrediction",
        "NAT_reis_narrow",
        "NAT_reis_broad",
        "NAT_reis_narrow_occurs_after_pred_time",
        "NAT_reis_narrow_PredictionToEvent",
        "NAT_reis_broad_occurs_after_pred_time",
        "NAT_reis_broad_PredictionToEvent",
    ]
    df = df.drop(cols_to_remove, axis=1)

    # Melt dataframe
    id_vars = [
        "ID",
        "time_from_prediction",
        "NAT_reis_broad_label",
        "NAT_reis_narrow_label",
        "NAT_reis_broad_neg_by_pred_time",
        "NAT_reis_narrow_neg_by_pred_time",
        "Age",
        "Sex",
        "Race",
        "Payer",
    ]
    df = chunked_melt(df, chunk_size=CHUNK_SIZE, id_vars=id_vars)
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
    feature_token_df = encode_column_using_encoder(df, "code", "feature_vocab")
    time_token_df = encode_column_using_encoder(
        df, "time_from_prediction", "time_vocab"
    )
    code_type_token_df = encode_column_using_encoder(
        df, "code_type", "type_vocab"
    )
    logger.info("Created encoded columns")

    # Get all patient IDs and demographics
    df = df.drop_duplicates(subset=["ID"], keep="first")[
        [
            "ID",
            "NAT_reis_broad_label",
            "NAT_reis_narrow_label",
            "NAT_reis_broad_neg_by_pred_time",
            "NAT_reis_narrow_neg_by_pred_time",
            "Age",
            "Sex",
            "Race",
            "Payer",
        ]
    ]
    df = chunked_melt(
        df,
        chunk_size=CHUNK_SIZE,
        id_vars=[
            "ID",
            "NAT_reis_broad_label",
            "NAT_reis_narrow_label",
            "NAT_reis_broad_neg_by_pred_time",
            "NAT_reis_narrow_neg_by_pred_time",
        ],
    )
    logger.info(
        "Melted dataframe by chunks, dropped nan and -1 values, and recombined"
    )

    # Create unique column to be used by encoder
    df["demographics"] = df.progress_apply(
        lambda row: row["variable"] + "_" + str(row["value"]), axis=1
    )
    df = df.drop(["variable", "value"], axis=1)

    # Coalesce rows into lists of tokens
    df = (
        df.groupby(
            [
                "ID",
                "NAT_reis_broad_label",
                "NAT_reis_narrow_label",
                "NAT_reis_broad_neg_by_pred_time",
                "NAT_reis_narrow_neg_by_pred_time",
            ]
        )["demographics"]
        .progress_apply(list)
        .reset_index()
    )

    # Load feature vocab
    with open(
        os.path.join(
            S3_OUTPUT_PREFIX, f"output/pretrain/encoders/feature_vocab.pickle"
        ),
        "rb",
    ) as encoder_file:
        feature_vocab = pickle.load(encoder_file)

    # Convert to numeric features
    df["demographics"] = df["demographics"].progress_apply(
        lambda features: list(map(lambda x: feature_vocab[x], features))
    )
    logger.info("Encoded demographics")

    # Merge token lists into single dataframe
    merged_df = pd.merge(feature_token_df, time_token_df, on="ID", how="inner")
    merged_df = pd.merge(merged_df, code_type_token_df, on="ID", how="inner")
    df = pd.merge(merged_df, df, on="ID", how="inner")
    logger.info("Merged coalesced rows")

    if args.job == "finetune":
        # 80/10/10 split
        train, val_and_test = train_test_split(
            df, test_size=0.2, random_state=42
        )
        val, test = train_test_split(
            val_and_test, test_size=0.5, random_state=42
        )
        logger.info("Performed train/val/test split")

        # Indicator columns to drop before saving
        drop_columns = [
            "NAT_reis_broad_neg_by_pred_time",
            "NAT_reis_narrow_neg_by_pred_time",
        ]

        # Save preprocessed unrestricted datasets
        train_path = os.path.join(
            S3_OUTPUT_PREFIX, "output/finetune/unrestricted/train.parquet"
        )
        train.drop(drop_columns, axis=1).to_parquet(train_path, index=False)
        logger.info(f"Uploaded unrestricted train set to: {train_path}")

        val_path = os.path.join(
            S3_OUTPUT_PREFIX, "output/finetune/unrestricted/val.parquet"
        )
        val.drop(drop_columns, axis=1).to_parquet(val_path, index=False)
        logger.info(f"Uploaded unrestricted val set to: {val_path}")

        test_path = os.path.join(
            S3_OUTPUT_PREFIX, "output/finetune/unrestricted/test.parquet"
        )
        test.drop(drop_columns, axis=1).to_parquet(test_path, index=False)
        logger.info(f"Uploaded unrestricted test set to: {test_path}")

        # Filter out patients with previous NAT diagnoses
        test = test[test["NAT_reis_broad_neg_by_pred_time"] == 1]
        logger.info(
            "Subset test dataset to exclude patients with previous NAT"
        )

        # Save preprocessed restricted dataset
        restricted_output_path = os.path.join(
            S3_OUTPUT_PREFIX, "output/finetune/restricted/test.parquet"
        )
        test.drop(drop_columns, axis=1).to_parquet(
            restricted_output_path, index=False
        )
    elif args.job == "external":
        # Indicator columns to drop before saving
        drop_columns = [
            "NAT_reis_broad_neg_by_pred_time",
            "NAT_reis_narrow_neg_by_pred_time",
        ]

        # Save preprocessed unrestricted datasets
        output_path = os.path.join(
            S3_OUTPUT_PREFIX,
            "output/external_validation/unrestricted/dataset.parquet",
        )
        df.drop(drop_columns, axis=1).to_parquet(output_path, index=False)
        logger.info(f"Uploaded unrestricted train set to: {output_path}")

        # Filter out patients with previous NAT diagnoses
        df = df[df["NAT_reis_broad_neg_by_pred_time"] == 1]
        logger.info(
            "Subset test dataset to exclude patients with previous NAT"
        )

        # Save preprocessed restricted dataset
        restricted_output_path = os.path.join(
            S3_OUTPUT_PREFIX,
            "output/external_validation/restricted/dataset.parquet",
        )
        df.drop(drop_columns, axis=1).to_parquet(
            restricted_output_path, index=False
        )
    else:
        raise Exception('Job must be either "finetune" or "external"')

    logger.info(f"Uploaded restricted test set to: {restricted_output_path}")
    logger.info("Finished job.")
