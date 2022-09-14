import os
import pickle
from typing import Callable, Dict

import numpy as np
import wandb
from smart_open import open
from torch.utils.data import DataLoader, WeightedRandomSampler

from .constants import S3_PRETRAIN_PREPROCESSED_PATH


def load_vocab(filename: str) -> Dict[str, int]:
    """Load vocab encoder from S3"""
    s3_vocab_path = os.path.join(
        S3_PRETRAIN_PREPROCESSED_PATH, "encoders", filename
    )
    with open(s3_vocab_path, "rb") as f:
        vocab = pickle.load(f)
    return vocab


def create_dataloader(
    dataset_path: str,
    dataset_constructor: Callable,
    feature_vocab: str,
    time_vocab: str,
    code_type_vocab: str,
    config: wandb.Config,
    truncate_to: int = -1,
    weighted_sampling: bool = False,
    batch_size: int = None,
) -> DataLoader:
    """Construct a dataset and dataloader"""
    dataset = dataset_constructor(
        s3_dataset_path=dataset_path,
        cls_id=feature_vocab["CLS"],
        feature_pad_id=feature_vocab["PAD"],
        feature_mask_id=feature_vocab["MASK"],
        time_pad_id=time_vocab["PAD"],
        code_type_pad_id=code_type_vocab["PAD"],
        vocab_size=len(feature_vocab),
        max_seq_length=config.max_seq_length,
        truncate_to=truncate_to,
    )

    # Optinally use weighted random shuffling
    if weighted_sampling:
        train_labels = dataset.dataframe["NAT_reis_broad_label"].values.astype(
            int
        )
        weight = np.array([1, config.sample_weight])
        samples_weight = np.array([weight[t] for t in train_labels])
        sampler = WeightedRandomSampler(samples_weight, len(samples_weight))
        dataloader = DataLoader(
            dataset,
            batch_size=config.batch_size if batch_size is None else batch_size,
            drop_last=True,
            sampler=sampler,
            num_workers=2,
            pin_memory=True,
        )
    else:
        dataloader = DataLoader(
            dataset,
            batch_size=config.batch_size if batch_size is None else batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=2,
            pin_memory=True,
        )

    return dataloader
