from typing import Tuple

import torch
import torch.nn as nn
from transformers.models.bert.modeling_bert import (
    BertEncoder,
    BertOnlyMLMHead,
    BertPooler,
    BertPreTrainedModel,
)


class CustomBertEmbeddings(nn.Module):
    """Construct embeddings from feature, time, and type IDs."""

    def __init__(self, config):
        super().__init__()
        self.feature_embeddings = nn.Embedding(
            config.feature_vocab_size,
            config.hidden_size,
            padding_idx=config.feature_pad_id,
        )
        self.time_embeddings = nn.Embedding(
            config.time_vocab_size,
            config.hidden_size,
            padding_idx=config.time_pad_id,
        )
        self.code_type_embeddings = nn.Embedding(
            config.code_type_vocab_size,
            config.hidden_size,
            padding_idx=config.code_type_pad_id,
        )
        self.layerNorm = nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps
        )
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(
        self,
        feature_ids: torch.LongTensor,
        time_ids: torch.LongTensor,
        code_type_ids: torch.LongTensor,
    ) -> torch.Tensor:
        """Embedding layer forard pass."""
        feature_embeds = self.feature_embeddings(feature_ids)
        time_embeds = self.time_embeddings(time_ids)
        code_type_embeds = self.code_type_embeddings(code_type_ids)

        # Sum embeddings together
        embeddings = feature_embeds + time_embeds + code_type_embeds

        embeddings = self.layerNorm(embeddings)
        embeddings = self.dropout(embeddings)

        return embeddings


class BertBaseModel(BertPreTrainedModel):
    """Base BERT model."""

    def __init__(self, config):
        super().__init__(config)
        self.config = config

        self.embeddings = CustomBertEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        feature_ids: torch.Tensor,
        time_ids: torch.Tensor,
        code_type_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor]:

        input_shape = feature_ids.size()
        device = feature_ids.device

        # We can provide a self-attention mask of dimensions
        # [batch_size, from_seq_length, to_seq_length] ourselves in which
        # case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = (
            self.get_extended_attention_mask(
                attention_mask, input_shape, device=device
            )
        )

        embedding_output = self.embeddings(
            feature_ids=feature_ids,
            time_ids=time_ids,
            code_type_ids=code_type_ids,
        )
        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            past_key_values=None,
            use_cache=False,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=False,
        )
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output)

        return (sequence_output, pooled_output)


class BertPretrain(BertPreTrainedModel):
    """BERT for pretraining using masked LM and next visit prediction."""

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.bert = BertBaseModel(config)

        # Masked LM head
        self.mlm_head = BertOnlyMLMHead(config)

        # Next visit diagnosis prediction head
        self.dropout = nn.Dropout(config.classifier_dropout)
        self.classifier = nn.Linear(
            config.hidden_size, config.next_visit_diagnosis_labels_size
        )

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        feature_ids: torch.Tensor,
        time_ids: torch.Tensor,
        code_type_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        mask_labels: torch.Tensor,
        labels: torch.Tensor,
    ) -> Tuple[torch.Tensor]:
        """Forward pass for masked LM and multiclass classification."""
        outputs = self.bert(
            feature_ids=feature_ids,
            time_ids=time_ids,
            code_type_ids=code_type_ids,
            attention_mask=attention_mask,
        )

        # Get output sequence and pooled output
        sequence_output = outputs[0]
        pooled_output = outputs[1]

        # Masked language modeling
        prediction_scores = self.mlm_head(sequence_output)
        loss_fct = nn.CrossEntropyLoss()  # -100 index = padding token
        masked_lm_loss = loss_fct(
            prediction_scores.view(-1, self.config.feature_vocab_size),
            mask_labels.view(-1),
        )

        # Next visit diagnosis prediction
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        loss_fct = nn.CrossEntropyLoss()
        next_visit_loss = loss_fct(logits, labels.view(-1))

        # Sum losses together
        loss = masked_lm_loss + next_visit_loss

        return (loss, masked_lm_loss, next_visit_loss)


class BertFinetune(BertPreTrainedModel):
    """BERT for binary classification of NAT"""

    def __init__(self, config):
        super().__init__(config)
        self.config = config

        self.bert = BertBaseModel(config)
        self.dropout = nn.Dropout(config.classifier_dropout)
        self.binary_classifier = nn.Linear(config.hidden_size, 1)

        # Class weighting
        self.pos_weight = torch.FloatTensor([config.pos_weight])

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        feature_ids: torch.Tensor,
        time_ids: torch.Tensor,
        code_type_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
    ) -> Tuple[torch.Tensor]:
        """Forward pass for binary classification."""
        self.pos_weight = self.pos_weight.to(feature_ids.device)

        outputs = self.bert(
            feature_ids=feature_ids,
            time_ids=time_ids,
            code_type_ids=code_type_ids,
            attention_mask=attention_mask,
        )

        # Get pooled output
        pooled_output = outputs[1]

        # Binary classification
        pooled_output = self.dropout(pooled_output)
        logits = self.binary_classifier(pooled_output)

        loss_fct = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)
        loss = loss_fct(logits, labels)

        return (loss, logits)
