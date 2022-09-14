import random
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class BaseDataset(Dataset):
    """Base dataset class for model training."""

    def __init__(
        self,
        s3_dataset_path: str,
        cls_id: int,
        feature_pad_id: int,
        feature_mask_id: int,
        time_pad_id: int,
        code_type_pad_id: int,
        vocab_size: int,
        max_seq_length: int,
        truncate_to: int = -1,
        return_ids: bool = False,
    ):
        """Initialize dataset."""
        self.dataframe = pd.read_parquet(s3_dataset_path)

        if truncate_to > 0:
            self.dataframe = self.dataframe[:truncate_to]

        self.cls_id = cls_id
        self.feature_pad_id = feature_pad_id
        self.feature_mask_id = feature_mask_id
        self.time_pad_id = time_pad_id
        self.code_type_pad_id = code_type_pad_id
        self.vocab_size = vocab_size
        self.max_seq_length = max_seq_length
        self.return_ids = return_ids

    def create_sample_dict(
        self, df_row: pd.Series, label: str
    ) -> Tuple[Dict[str, np.ndarray], int]:
        """Create a sample dictionary."""
        sample_dict = {
            "feature_tokens": self.feature_pad_id
            * np.ones(self.max_seq_length),
            "code_type_tokens": self.code_type_pad_id
            * np.ones(self.max_seq_length),
            "time_from_prediction_tokens": self.time_pad_id
            * np.ones(self.max_seq_length),
            "attention_mask": np.zeros(self.max_seq_length),
            "label": np.array([df_row[label]]),
        }

        # Add CLS token
        sample_dict["feature_tokens"][0] = self.cls_id
        sample_dict["attention_mask"][0] = 1

        # Fill in demographic tokens
        ds = df_row.demographics.size
        sample_dict["feature_tokens"][1 : 1 + ds] = df_row.demographics
        sample_dict["attention_mask"][1 : 1 + ds] = 1

        # Truncate patients with too many codes, saving most recent
        remaining_space = self.max_seq_length - ds - 1
        cs = (
            remaining_space
            if df_row.code.size > remaining_space
            else df_row.code.size
        )

        # Replace unknown codes (-1) with pad token ID
        codes = df_row.code.copy()
        codes[codes == -1] = self.feature_pad_id

        # Fill in code tokens, truncating if necessary
        sample_dict["feature_tokens"][1 + ds : 1 + ds + cs] = codes[:cs]
        sample_dict["code_type_tokens"][
            1 + ds : 1 + ds + cs
        ] = df_row.code_type[:cs]
        sample_dict["time_from_prediction_tokens"][
            1 + ds : 1 + ds + cs
        ] = df_row.time_from_prediction[:cs]
        sample_dict["attention_mask"][1 + ds : 1 + ds + cs] = 1

        # Track how many non-padded tokens are in the output
        non_padded_seq_length = 1 + ds + cs

        return sample_dict, non_padded_seq_length

    def __len__(self):
        """Get dataset size."""
        return self.dataframe.shape[0]


class PretrainDataset(BaseDataset):
    """Dataset class for pretraining (masked LM and next visit prediction)."""

    def random_masking(
        self, sample_dict: Dict[str, np.ndarray], non_padded_seq_length: int
    ) -> Dict[str, np.ndarray]:
        """Randomly mask 15% of tokens."""
        feature_tokens = sample_dict["feature_tokens"]
        feature_tokens_with_mask = np.copy(feature_tokens)
        mask_labels = np.full(feature_tokens.shape, -100)

        indices_to_replace = []
        tokens_to_replace_with = []

        # Exclude CLS token at the front
        for index in range(1, non_padded_seq_length):
            prob = random.random()

            # Mask with 15% probability
            if prob < 0.15:
                indices_to_replace.append(index)

                prob /= 0.15
                if prob < 0.8:
                    # 80% randomly change token to mask token
                    tokens_to_replace_with.append(self.feature_mask_id)
                elif prob < 0.9:
                    # 10% randomly change token to random token
                    random_token = random.randint(0, self.vocab_size - 1)
                    tokens_to_replace_with.append(random_token)
                else:
                    # 10% randomly keep current token
                    tokens_to_replace_with.append(feature_tokens[index])

        np.put(
            feature_tokens_with_mask,
            indices_to_replace,
            tokens_to_replace_with,
        )
        np.put(
            mask_labels, indices_to_replace, feature_tokens[indices_to_replace]
        )

        # Update sample dictionary
        sample_dict["feature_tokens"] = feature_tokens_with_mask
        sample_dict["mask_labels"] = mask_labels

        return sample_dict

    def __getitem__(self, idx):
        """Return a sample dictionary given an index."""
        df_row = self.dataframe.iloc[idx]

        # Create a dictionary of token IDs
        sample_dict, non_padded_seq_length = self.create_sample_dict(
            df_row=df_row, label="next_visit_dx_category"
        )

        # Perform random masking
        sample_dict = self.random_masking(sample_dict, non_padded_seq_length)

        # Convet to tensor types
        for key, arr in sample_dict.items():
            sample_dict[key] = torch.LongTensor(arr)

        return sample_dict


class FinetuneDataset(BaseDataset):
    """Dataset class for finetuning (NAT prediction)."""

    def __getitem__(self, idx):
        """Return a sample dictionary given an index."""
        df_row = self.dataframe.iloc[idx]

        # Create a dictionary of token IDs
        sample_dict, _ = self.create_sample_dict(
            df_row=df_row, label="NAT_reis_broad_label"
        )

        # Convet to tensor types
        for key, arr in sample_dict.items():
            if key == "label":
                # nn.BCEWithLogitsLoss expects a FloatTensor
                sample_dict[key] = torch.FloatTensor(arr)
            else:
                sample_dict[key] = torch.LongTensor(arr)

        if self.return_ids:
            sample_dict["ID"] = df_row["ID"]

        return sample_dict
