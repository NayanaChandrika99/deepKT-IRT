# ABOUTME: Declares the PyTorch Lightning implementation of Wide & Deep IRT.
# ABOUTME: Provides the fusion of wide IRT component with deep clickstream encoders.

from dataclasses import dataclass
from typing import Dict, Sequence

import pytorch_lightning as pl
import torch
from torch import nn
from torchmetrics.classification import BinaryAUROC

from .features import FeatureConfig


@dataclass
class WideDeepConfig:
    """Configuration values needed by the Wide & Deep IRT module."""

    wide_units: int
    deep_units: Sequence[int]
    embedding_dim: int
    dropout: float
    activation: str
    learning_rate: float
    weight_decay: float
    ability_regularizer: float


class WideDeepIrtModule(pl.LightningModule):
    """Wide & Deep IRT model implemented with PyTorch Lightning."""

    def __init__(
        self,
        config: WideDeepConfig,
        feature_config: FeatureConfig,
        item_vocab_size: int,
        action_vocab_size: int,
        latency_bucket_count: int,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["feature_config"])
        self.config = config
        self.feature_config = feature_config

        embed_dim = config.embedding_dim
        self.action_embedding = nn.Embedding(action_vocab_size, embed_dim, padding_idx=0)
        self.latency_embedding = nn.Embedding(latency_bucket_count, embed_dim, padding_idx=0)
        self.recency_proj = nn.Linear(1, embed_dim)
        self.success_proj = nn.Linear(1, embed_dim)
        self.missing_proj = nn.Linear(1, embed_dim)
        self.history_embedding_proj = nn.Linear(feature_config.bert_dim, embed_dim)

        deep_input_dim = embed_dim * 5 + embed_dim
        layers = []
        in_dim = deep_input_dim
        for units in config.deep_units:
            layers.append(nn.Linear(in_dim, units))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(config.dropout))
            in_dim = units
        layers.append(nn.Linear(in_dim, 1))
        self.deep_layers = nn.Sequential(*layers)

        self.item_beta = nn.Parameter(torch.zeros(item_vocab_size))
        self.item_guess = nn.Parameter(torch.zeros(item_vocab_size))

        self.criterion = nn.BCELoss()
        self.val_auc = BinaryAUROC()
        self.test_auc = BinaryAUROC()

    def forward(self, batch: Dict[str, torch.Tensor]):
        """Return predicted probabilities and abilities."""

        ability = self._deep_component(batch)
        probs = self._wide_component(batch, ability)
        return probs, ability

    def training_step(self, batch, batch_idx):
        inputs, labels, _ = batch
        probs, ability = self.forward(inputs)
        loss = self.criterion(probs, labels)
        loss = loss + self.config.ability_regularizer * ability.pow(2).mean()
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, labels, _ = batch
        probs, _ = self.forward(inputs)
        loss = self.criterion(probs, labels)
        self.val_auc.update(probs, labels.int())
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def on_validation_epoch_end(self):
        auc = self.val_auc.compute()
        if not torch.isnan(auc):
            self.log("val_auc", auc, prog_bar=True)
        self.val_auc.reset()

    def test_step(self, batch, batch_idx):
        inputs, labels, _ = batch
        probs, _ = self.forward(inputs)
        self.test_auc.update(probs, labels.int())

    def on_test_epoch_end(self):
        auc = self.test_auc.compute()
        if not torch.isnan(auc):
            self.log("test_auc", auc, prog_bar=True)
        self.test_auc.reset()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        return optimizer

    def _deep_component(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        action_embed = self.action_embedding(inputs["history_actions"])
        latency_embed = self.latency_embedding(inputs["history_latency_bucket"])
        recency = self.recency_proj(inputs["history_recency"].unsqueeze(-1))
        success = self.success_proj(inputs["history_item_success_rates"].unsqueeze(-1))
        missing = self.missing_proj(inputs["history_missing"].unsqueeze(-1))
        hist_embed = self.history_embedding_proj(inputs["history_item_embeddings"])

        deep_input = torch.cat(
            [action_embed, latency_embed, recency, success, hist_embed, missing],
            dim=-1,
        )

        mask = 1.0 - inputs["mask"].unsqueeze(-1)
        transformed = self.deep_layers(deep_input)
        transformed = transformed * mask
        valid = mask.sum(dim=1).clamp_min(1.0)
        pooled = transformed.sum(dim=1) / valid
        return pooled

    def _wide_component(self, inputs: Dict[str, torch.Tensor], ability: torch.Tensor) -> torch.Tensor:
        item_idx = inputs["future_item"].long().view(-1)
        beta = self.item_beta[item_idx].unsqueeze(-1)
        guess = torch.sigmoid(self.item_guess[item_idx]).unsqueeze(-1)
        logits = ability - beta
        base_prob = torch.sigmoid(logits)
        prob = (1 - guess) * base_prob + guess
        is_mc = inputs["future_item_is_mc"].unsqueeze(-1)
        return torch.where(is_mc > 0.5, prob, base_prob)
