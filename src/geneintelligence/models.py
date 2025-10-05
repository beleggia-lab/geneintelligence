# Standard library imports
import datetime
import hashlib
import math
import string
from dataclasses import asdict, dataclass, field, fields
from typing import Optional, Dict, Tuple

# Third-party imports
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.classification import (BinaryAccuracy, BinaryPrecision,
                                         BinaryRecall, BinaryF1Score,
                                         BinaryAUROC, MulticlassAccuracy,
                                         MulticlassPrecision, MulticlassRecall,
                                         MulticlassF1Score)
from torchmetrics.regression import MeanAbsoluteError, MeanSquaredError, R2Score
# Local imports
from geneintelligence.configs import EiConfig


def create_loss_fn(loss_name: str, loss_kwargs: dict) -> nn.Module:
    """Instantiate loss function from name + kwargs."""
    return getattr(nn, loss_name)(**loss_kwargs)


class SwiGLU(nn.Module):

    def __init__(self):
        super().__init__()
        self.silu = nn.SiLU()  # Swish activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """SwiGLU activation: SiLU(x) * Linear(x)"""
        x1, x2 = torch.chunk(x, 2, dim=-1)  # Split tensor into two halves
        return self.silu(x1) * x2  # Apply activation to one half and multiply


def swiglu(x: torch.Tensor, w1: nn.Module, w2: nn.Module) -> torch.Tensor:
    return F.silu(w1(x)) * w2(x)  # Swish(x) * Linear(x)


class EiPositionalEncoding(nn.Module):

    def __init__(self, d_model: int, max_len: int, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() *
            (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # add batch dimension: (1, max_len, d_model)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch_size, seq_len, d_model)
        Returns:
            Tensor of same shape as x with positional encodings added.
        """
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len]
        return self.dropout(x)


# FlashAttention-based Multi-head Self-Attention Module
class EiFlashSelfAttention(nn.Module):

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        if self.head_dim * num_heads != d_model:
            raise ValueError("d_model must be divisible by num_heads")

        # One projection for Q, K, and V
        self.qkv_proj = nn.Linear(d_model, d_model * 3)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor,
                inverted_padding: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch_size, seq_len, d_model)
            inverted_padding: Boolean mask of shape (batch_size, seq_len) where True indicates real tokens and False indicates padding
        Returns:
            Tensor of shape (batch_size, seq_len, d_model) after attention.
        """
        batch, seq_len, _ = x.shape

        # Project to get Q, K, V in one go
        qkv = self.qkv_proj(x)  # (batch, seq_len, 3 * d_model)

        # Split into Q, K, V and reshape into multiple heads
        qkv = qkv.reshape(batch, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1,
                          4)  # (3, batch, num_heads, seq_len, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Convert the padding mask to attention mask
        # src_key_padding_mask shape: [batch, seq_len]
        # Need shape: [batch, 1, 1, seq_len] for broadcasting
        attn_mask = inverted_padding.unsqueeze(1).unsqueeze(2)

        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query=q,
            key=k,
            value=v,
            dropout_p=0,
            is_causal=False,
            attn_mask=attn_mask)
        # Reshape back: (batch, seq_len, d_model)
        attn_output = attn_output.transpose(1,
                                            2).reshape(batch, seq_len,
                                                       self.d_model)
        output = self.out_proj(attn_output)
        output = self.dropout(output)
        return output


# FeedForward network (MLP) within the Transformer block
class EiFeedForward(nn.Module):

    def __init__(self, d_model: int, ff_dim: int, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model, ff_dim)
        #self.act = nn.GELU()
        self.act = SwiGLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(ff_dim // 2, d_model)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout2(x)
        return x


# A single Transformer block
class EiTransformerBlock(nn.Module):

    def __init__(self,
                 d_model: int,
                 num_heads: int,
                 ff_dim: int,
                 dropout: float = 0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = EiFlashSelfAttention(d_model, num_heads, dropout=dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = EiFeedForward(d_model, ff_dim, dropout=dropout)

    def forward(
            self,
            x: torch.Tensor,
            inverted_padding: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Self-Attention sublayer with residual connection
        attn_output = self.attn(self.ln1(x), inverted_padding=inverted_padding)
        x = x + attn_output
        # FeedForward sublayer with residual connection
        ff_output = self.ff(self.ln2(x))
        x = x + ff_output
        return x


class ExpressionIntelligence(nn.Module):

    def __init__(self,
                 config: EiConfig,
                 device: Optional[torch.device] = None):
        super().__init__()
        if device is None:
            device = torch.device(
                'cuda' if torch.cuda.is_available() else 'cpu')
        self.token_emb = nn.Embedding(config.vocab_size,
                                      config.backbone_d_model)
        self.pos_emb = EiPositionalEncoding(config.backbone_d_model,
                                            config.max_len,
                                            config.backbone_dropout)
        self.blocks = nn.ModuleList([
            EiTransformerBlock(config.backbone_d_model, config.backbone_heads,
                               config.backbone_ff_dim, config.backbone_dropout)
            for _ in range(config.backbone_blocks)
        ])
        # Final layer normalization (optional, as in BERT)
        self.ln_f = nn.LayerNorm(config.backbone_d_model)

        self.prediction_head = nn.Sequential(
            nn.Linear(config.backbone_d_model, config.backbone_d_model),
            nn.GELU(), nn.LayerNorm(config.backbone_d_model),
            nn.Dropout(config.backbone_dropout),
            nn.Linear(config.backbone_d_model, config.vocab_size, bias=False))

        # tying the weights of the embedding and prediction head
        self.prediction_head[-1].weight = self.token_emb.weight
        self.loss_fn = create_loss_fn(config.backbone_loss,
                                      config.backbone_loss_kwargs)
        self.device = device

    def forward(self,
                tokens: torch.Tensor,
                mask_positions: Optional[torch.Tensor] = None,
                inverted_padding: Optional[torch.Tensor] = None,
                only_masked: Optional[bool] = None):
        """
        Args:
            tokens: Tensor of shape (batch_size, seq_len) containing token indices.
            mask_positions: Optional mask for attention of shape [batch_size, n_masked],
                            where True indicates that the position is masked and should be ignored.
            inverted_padding: Optional padding mask of shape [batch_size, seq_len],
                                  where False indicates that the position is padding and should be ignored.
            only_masked: If True, compute logits only for masked positions.
                        If False, compute logits for all positions.
                        If None, compute logits only for masked positions during training,
                        and for all positions during inference.
        Returns:
            Tensor of shape (batch_size, seq_len, d_model) representing encoded features.
            
        """
        x = self.token_emb(tokens)  # (batch_size, seq_len, d_model)
        x = self.pos_emb(x)
        for block in self.blocks:
            x = block(x, inverted_padding=inverted_padding)

        x = self.ln_f(x)

        if (only_masked is True) or (only_masked is not False
                                     and self.training):
            # During training: compute logits only for masked positions
            masked_outputs = x[mask_positions]
            logits = self.prediction_head(masked_outputs)
        else:
            # During inference: compute logits for all positions
            logits = self.prediction_head(x)
        return logits

    def compute_loss(
            self, batch: Dict[str,
                              torch.Tensor]) -> Tuple[torch.Tensor, float]:
        tokens = batch['masked_tokens'].to(self.device)
        mask_positions = batch['mask_positions'].to(self.device)
        inverted_padding = batch['inverted_padding'].to(self.device)
        original_tokens = batch['original_tokens'].to(self.device)

        # Get masked positions predictions
        masked_logits = self(tokens,
                             mask_positions=mask_positions,
                             inverted_padding=inverted_padding,
                             only_masked=True)
        masked_targets = original_tokens[mask_positions]
        loss = self.loss_fn(masked_logits, masked_targets)

        with torch.no_grad():
            accuracy = (masked_logits.argmax(
                dim=-1) == masked_targets).float().mean()

        return loss, accuracy


class EiCellAnalyzer(nn.Module):

    def __init__(self,
                 pretrained_model: ExpressionIntelligence,
                 config: EiConfig,
                 device: str,
                 num_metadata_tokens: int = 5):
        super().__init__()
        self.backbone = pretrained_model
        self.finetune_task = config.finetune_task
        self.classes = config.finetune_classes
        self.n_output_units = 1
        if self.finetune_task == 'classification' and len(self.classes) > 2:
            self.n_output_units = len(self.classes)

        self.d_model = config.backbone_d_model
        self.dropout = config.backbone_dropout
        self.num_heads = config.backbone_heads
        self.ff_dim = config.backbone_ff_dim
        self.num_layers = config.backbone_blocks
        self.keep_blocks = config.finetune_keep_blocks
        self.new_blocks = config.finetune_new_blocks

        self.device = device
        self.num_metadata_tokens = num_metadata_tokens
        self.loss_fn = create_loss_fn(config.finetune_loss,
                                      config.finetune_loss_kwargs)

        if self.new_blocks > 0:
            self.classification_blocks = nn.ModuleList([
                EiTransformerBlock(self.d_model,
                                   self.num_heads,
                                   self.ff_dim,
                                   dropout=self.dropout)
                for _ in range(self.new_blocks)
            ])

        self.final_head = nn.Sequential(
            nn.Linear((self.num_metadata_tokens + 2) * self.d_model,
                      self.d_model), nn.GELU(), nn.LayerNorm(self.d_model),
            nn.Dropout(self.dropout),
            nn.Linear(self.d_model, self.n_output_units))

        self.backbone.blocks = nn.ModuleList(
            self.backbone.blocks[:self.keep_blocks])

        del self.backbone.prediction_head
        del self.backbone.ln_f

        for param in self.parameters():
            param.requires_grad = True

        if config.finetune_freeze_embeddings:
            print("Freezing backbone embeddings...")
            for param in self.backbone.token_emb.parameters():
                param.requires_grad = False
            for param in self.backbone.pos_emb.parameters():
                param.requires_grad = False

        num_frozen = config.finetune_freeze_blocks
        if num_frozen > 0:
            print(
                f"Freezing the first {num_frozen} backbone transformer blocks..."
            )
            for i, block in enumerate(self.backbone.blocks):
                if i < num_frozen:
                    for param in block.parameters():
                        param.requires_grad = False
        self.to(self.device)

        # Metrics
        self.metrics = {'train': {}, 'val': {}}

    def reset_metrics(self, dataset_metrics_keyword: str):
        for metric in self.metrics[dataset_metrics_keyword].values():
            metric.reset()

    def forward(self, tokens, inverted_padding):
        x = self.backbone.token_emb(tokens)
        x = self.backbone.pos_emb(x)

        for block in self.backbone.blocks:
            x = block(x, inverted_padding=inverted_padding)

        if self.new_blocks > 0:
            for block in self.classification_blocks:
                x = block(x, inverted_padding=inverted_padding)

        # Separate metadata and gene tokens
        metadata_repr = x[:, :self.num_metadata_tokens +
                          1, :]  # shape: (batch_size, 6, d_model)
        gene_repr = x[:, self.num_metadata_tokens +
                      1:, :]  # shape: (batch_size, gene_seq_len, d_model)
        gene_mask = inverted_padding[:,
                                     self.num_metadata_tokens + 1:].unsqueeze(
                                         -1
                                     )  # shape: (batch_size, gene_seq_len, 1)

        # Mean pooling over gene tokens
        pooled_genes = (gene_repr *
                        gene_mask).sum(dim=1) / gene_mask.sum(dim=1)

        # Concatenate metadata and pooled gene vector
        combined_repr = torch.cat(
            [metadata_repr, pooled_genes.unsqueeze(1)], dim=1
        )  # shape: (batch_size, self.num_metadata_tokens + 1, d_model)
        combined_repr = combined_repr.view(
            combined_repr.size(0), -1
        )  # shape: (batch_size, (self.num_metadata_tokens + 1) * d_model)

        logits = self.final_head(combined_repr)
        return logits


class EiCellBinaryClassifier(EiCellAnalyzer):

    def __init__(self,
                 pretrained_model: ExpressionIntelligence,
                 config: EiConfig,
                 device: str,
                 num_metadata_tokens: int = 5):
        super().__init__(pretrained_model, config, device, num_metadata_tokens)

        assert self.finetune_task == 'classification', "Binary classification model initialized with finetune_task != 'classification'"
        assert len(
            self.classes
        ) == 2, "Binary classification model initialized with classes != 2"

        for subset_keyword in ['train', 'val']:
            self.metrics[subset_keyword] = {
                'accuracy': BinaryAccuracy().to(self.device),
                'precision': BinaryPrecision().to(self.device),
                'recall': BinaryRecall().to(self.device),
                'f1': BinaryF1Score().to(self.device),
                'auroc': BinaryAUROC().to(self.device),
            }

    def compute_loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        logits = self(batch['tokens'], batch['inverted_padding'])
        targets = batch['ground_truth']

        loss = self.loss_fn(logits.squeeze(-1), targets.float())
        return loss

    def compute_loss_with_metrics(
            self,
            batch: Dict[str, torch.Tensor],
            dataset_metrics_keyword: str = 'train') -> torch.Tensor:
        logits = self(batch['tokens'], batch['inverted_padding'])
        targets = batch['ground_truth']

        loss = self.loss_fn(logits.squeeze(-1), targets.float())
        probs = torch.sigmoid(logits.squeeze(-1))
        preds = (probs > 0.5).long()

        for name, metric in self.metrics[dataset_metrics_keyword].items():
            if name == 'auroc':
                metric.update(probs, targets)
            else:
                metric.update(preds, targets)

        return loss

    def predict(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        with torch.no_grad():
            logits = self(batch['tokens'], batch['inverted_padding'])
            probs = torch.sigmoid(logits.squeeze(-1))
            preds = (probs > 0.5).long()
        return preds


class EiCellMulticlassClassifier(EiCellAnalyzer):

    def __init__(self,
                 pretrained_model: ExpressionIntelligence,
                 config: EiConfig,
                 device: str,
                 num_metadata_tokens: int = 5):
        super().__init__(pretrained_model, config, device, num_metadata_tokens)

        assert self.finetune_task == 'classification', "Multiclass classification model initialized with finetune_task != 'classification'"
        assert len(
            self.classes
        ) > 2, "Multiclass classification model initialized with classes <= 2"

        for subset_keyword in ['train', 'val']:
            self.metrics[subset_keyword] = {
                'accuracy':
                MulticlassAccuracy(num_classes=self.n_output_units).to(
                    self.device),
                'precision':
                MulticlassPrecision(num_classes=self.n_output_units).to(
                    self.device),
                'recall':
                MulticlassRecall(num_classes=self.n_output_units).to(
                    self.device),
                'f1':
                MulticlassF1Score(num_classes=self.n_output_units).to(
                    self.device)
            }

    def compute_loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        logits = self(batch['tokens'], batch['inverted_padding'])
        targets = batch['ground_truth']

        loss = self.loss_fn(logits, targets)
        return loss

    def compute_loss_with_metrics(
            self,
            batch: Dict[str, torch.Tensor],
            dataset_metrics_keyword: str = 'train') -> torch.Tensor:
        logits = self(batch['tokens'], batch['inverted_padding'])
        targets = batch['ground_truth']

        loss = self.loss_fn(logits, targets)
        preds = logits.argmax(dim=-1)

        for name, metric in self.metrics[dataset_metrics_keyword].items():
            if name == 'auroc':
                metric.update(probs, targets)
            else:
                metric.update(preds, targets)

        return loss

    def predict(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        with torch.no_grad():
            logits = self(batch['tokens'], batch['inverted_padding'])
            preds = logits.argmax(dim=-1)
        return preds


class EiCellRegressor(EiCellAnalyzer):

    def __init__(self,
                 pretrained_model: ExpressionIntelligence,
                 config: EiConfig,
                 device: str,
                 num_metadata_tokens: int = 5):
        super().__init__(pretrained_model, config, device, num_metadata_tokens)

        assert self.finetune_task == 'regression', "Regression model initialized with finetune_task != 'regression'"

        for subset_keyword in ['train', 'val']:
            self.metrics[subset_keyword] = {
                'mae': MeanAbsoluteError().to(self.device),
                'rmse': MeanSquaredError(squared=False).to(self.device),
                'r2_score': R2Score().to(self.device)
            }

    def compute_loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:

        predictions = self(batch['tokens'], batch['inverted_padding'])
        targets = batch['ground_truth']

        loss = self.loss_fn(predictions.squeeze(-1), targets.float())
        return loss

    def compute_loss_with_metrics(
            self,
            batch: Dict[str, torch.Tensor],
            dataset_metrics_keyword: str = 'train') -> torch.Tensor:
        """Computes loss and updates all relevant metrics for a given batch."""
        predictions = self(batch['tokens'], batch['inverted_padding'])
        targets = batch['ground_truth']

        predictions_squeezed = predictions.squeeze(-1)

        loss = self.loss_fn(predictions_squeezed, targets.float())

        for metric in self.metrics[dataset_metrics_keyword].values():
            metric.update(predictions_squeezed, targets)

        return loss

    def predict(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Performs inference on a batch and returns the predictions."""
        with torch.no_grad():
            predictions = self(batch['tokens'], batch['inverted_padding'])
        return predictions.squeeze(-1)
