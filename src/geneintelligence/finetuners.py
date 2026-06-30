#!/usr/bin/env python3
"""
Script Name: finetuners.py
Description: Finetuning architectures and task-specific heads for the 
             Gene Intelligence model (cell-level and gene-level tasks).

Usage:
    Imported as a module. Not intended for direct execution.
"""

# Standard Library Imports
import copy
import logging
import math

# Third-Party Imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.classification import (
    BinaryAccuracy,
    BinaryAUROC,
    BinaryF1Score,
    BinaryPrecision,
    BinaryRecall,
    MulticlassAccuracy,
    MulticlassF1Score,
    MulticlassPrecision,
    MulticlassRecall,
    MultilabelAccuracy,
    MultilabelAUROC,
    MultilabelF1Score,
    MultilabelPrecision,
    MultilabelRecall,
)
from torchmetrics.regression import MeanAbsoluteError, MeanSquaredError

# Local Application Imports
from geneintelligence.configs import GiFinetuneConfig
from geneintelligence.models import (
    GiTransformerBlock,
    GeneIntelligence,
    create_loss_fn,
)

logger = logging.getLogger(__name__)


class GeneQueryAttentionPool(nn.Module):
    """
    Multi-query attention pooling over the gene block.

    A bank of `n_queries` learned query vectors cross-attends over the gene
    token representations, producing a fixed (n_queries, d_model) summary
    regardless of how many genes are present. Operates on the dense
    (B, S_genes, d_model) tensor with a boolean key-padding mask.
    """

    def __init__(self, d_model: int, n_queries: int, backbone_heads: int,
                 dropout: float) -> None:
        super().__init__()
        # Derive heads from the backbone_heads
        n_heads = backbone_heads

        self.n_queries = n_queries
        self.queries = nn.Parameter(torch.randn(n_queries, d_model) * 0.02)
        self.attn = nn.MultiheadAttention(d_model,
                                          n_heads,
                                          dropout=dropout,
                                          batch_first=True)

    def forward(self, gene_repr: torch.Tensor,
                gene_valid_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            gene_repr: (B, S_genes, d_model)
            gene_valid_mask: (B, S_genes), True where a real gene token sits.
        Returns:
            (B, n_queries * d_model)
        """
        B = gene_repr.size(0)
        q = self.queries.unsqueeze(0).expand(B, -1,
                                             -1)  # (B, n_queries, d_model)
        key_padding_mask = ~gene_valid_mask  # True = ignore

        # Guard: a cell with zero valid gene tokens has an all-True padding
        # row, which makes softmax over all -inf -> NaN. Temporarily mark such
        # rows as fully-attendable so the op is well-defined, then zero their
        # output below.
        has_genes = gene_valid_mask.any(dim=1)  # (B,)
        safe_kpm = key_padding_mask.clone()
        safe_kpm[~has_genes] = False

        pooled, _ = self.attn(
            q, gene_repr, gene_repr,
            key_padding_mask=safe_kpm)  # (B, n_queries, d_model)
        pooled = pooled * has_genes.view(B, 1, 1).to(pooled.dtype)

        return pooled.reshape(B, self.n_queries * gene_repr.size(-1))


class GiFinetunerBase(nn.Module):
    """
    Base class for all finetuning tasks.
    Handles backbone pruning, block freezing, and new block addition.
    """

    def __init__(self, pretrained_model: GeneIntelligence,
                 config: GiFinetuneConfig, device: str) -> None:
        super().__init__()
        self.backbone = copy.deepcopy(pretrained_model)
        self.finetune_task = config.finetune_task
        self.classes = config.finetune_classes
        self.device = device
        self.head_type = getattr(config, 'finetune_head_type', 'linear')

        # Determine number of output units
        if self.finetune_task == 'classification':
            if self.classes and len(self.classes) > 2:
                self.n_output_units = len(self.classes)
            else:
                self.n_output_units = 1
        elif self.finetune_task == 'regression':
            self.n_output_units = config.finetune_n_regression_targets

        # Get hyperparameters from config
        self.d_model = config.backbone_d_model
        # Use finetune_dropout for new blocks and heads
        self.dropout = config.finetune_dropout
        self.num_heads = config.backbone_heads
        self.ff_dim = config.backbone_ff_dim
        self.keep_blocks = config.finetune_keep_blocks
        self.new_blocks = config.finetune_new_blocks

        # Instantiate loss function
        self.loss_fn = create_loss_fn(config.finetune_loss,
                                      config.finetune_loss_kwargs)

        # --- Model Pruning and Architecture ---

        # 1. Prune backbone to keep_blocks
        self.backbone.blocks = nn.ModuleList(
            self.backbone.blocks[:self.keep_blocks])

        # 2. Delete pretraining prediction head
        if hasattr(self.backbone, 'prediction_head'):
            del self.backbone.prediction_head
        if hasattr(self.backbone, 'count_prediction_head'):
            del self.backbone.count_prediction_head

        # 3. Add new finetuning-specific transformer blocks
        if self.new_blocks > 0:
            self.finetune_blocks = nn.ModuleList([
                GiTransformerBlock(self.d_model,
                                   self.num_heads,
                                   self.ff_dim,
                                   dropout=self.dropout,
                                   layer_idx=self.keep_blocks + i)
                for i in range(self.new_blocks)
            ])
            self.finetune_blocks.apply(self.backbone._init_weights)
        else:
            self.finetune_blocks = nn.ModuleList()  # Empty list

        # --- Parameter Freezing ---

        # Start with all parameters as trainable
        for param in self.parameters():
            param.requires_grad = True

        # 4. Freeze embeddings if requested
        if config.finetune_freeze_embeddings:
            logger.info("Freezing backbone embeddings...")
            for param in self.backbone.token_emb.parameters():
                param.requires_grad = False
            for param in self.backbone.embeddings.parameters():
                param.requires_grad = False

        # 5. Freeze backbone blocks if requested
        num_frozen = config.finetune_freeze_blocks
        if num_frozen > 0:
            assert num_frozen <= self.keep_blocks, "Cannot freeze more blocks than we have"
            logger.info(
                f"Freezing the first {num_frozen} backbone transformer blocks..."
            )
            for i, block in enumerate(self.backbone.blocks):
                if i < num_frozen:
                    for param in block.parameters():
                        param.requires_grad = False

        # --- Metrics ---
        self.metrics = nn.ModuleDict({
            'train_metrics': nn.ModuleDict(),
            'val_metrics': nn.ModuleDict()
        })
        self.to(self.device)

    def _base_forward(self,
                      tokens: torch.Tensor,
                      valid_attention_mask: torch.Tensor,
                      counts: torch.Tensor | None = None) -> torch.Tensor:
        """
        Runs the token embeddings and all transformer blocks (kept + new).
        Returns the full sequence of token representations.
        """

        batch, seq_len = tokens.shape
        d_model = self.d_model

        # 1. Embeddings (on padded tensors)
        x = self.backbone.token_emb(tokens)
        x = self.backbone.embeddings(x, counts=counts)

        # 2. Pack
        lengths = valid_attention_mask.sum(dim=1, dtype=torch.int32)
        cu_seqlens = F.pad(lengths.cumsum(0, dtype=torch.int32), (1, 0))
        max_seqlen = seq_len

        mask_flat = valid_attention_mask.reshape(-1)
        x_packed = x.reshape(-1, d_model)[mask_flat]

        # 3. Transformer blocks on packed tensors
        for block in self.backbone.blocks:
            x_packed = block(x_packed,
                             cu_seqlens=cu_seqlens,
                             max_seqlen=max_seqlen)

        for block in self.finetune_blocks:
            x_packed = block(x_packed,
                             cu_seqlens=cu_seqlens,
                             max_seqlen=max_seqlen)

        # 4. Unpack
        x = torch.zeros(batch * seq_len,
                        d_model,
                        device=tokens.device,
                        dtype=x_packed.dtype)
        x[mask_flat] = x_packed
        x = x.reshape(batch, seq_len, d_model)

        return x  # Shape: (batch_size, seq_len, d_model)

    def reset_metrics(self, dataset_metrics_keyword: str) -> None:
        """Resets all metrics for a given dataset (e.g., 'train_metrics' or 'val_metrics')."""
        for metric in self.metrics[dataset_metrics_keyword].values():
            metric.reset()

    def forward(self,
                tokens: torch.Tensor,
                valid_attention_mask: torch.Tensor,
                counts: torch.Tensor | None = None) -> torch.Tensor:
        """Child classes must implement their specific forward pass."""
        raise NotImplementedError

    @torch.compiler.disable
    def _safe_update_metrics(self,
                             dataset_metrics_keyword: str,
                             targets: torch.Tensor,
                             preds: torch.Tensor | None = None,
                             probs: torch.Tensor | None = None) -> None:
        """
        Updates torchmetrics outside of the compiled graph to prevent recompilation 
        loops and stateful graph breaks.
        """
        for name, metric in self.metrics[dataset_metrics_keyword].items():
            # Handle specific requirements
            if name == 'auroc' and probs is not None:
                metric.update(probs, targets)
            # Handle standard metrics (accuracy, precision, recall, MAE, etc.)
            elif preds is not None:
                metric.update(preds, targets)


class GiCellAnalyzer(GiFinetunerBase):
    """Cell-level finetuning base: pools the gene block and concatenates it with metadata tokens.

    Supports 'linear', 'MLP', and 'gene_dot' head types; subclassed by the
    cell classification/regression task heads.
    """

    def __init__(self,
                 pretrained_model: GeneIntelligence,
                 config: GiFinetuneConfig,
                 device: str,
                 token_ids: torch.Tensor | None = None,
                 token_bias_init: torch.Tensor | None = None) -> None:

        super().__init__(pretrained_model, config, device)

        self.num_metadata_tokens = len(config.metadata_cols)

        self.ln_f = nn.LayerNorm(self.d_model)
        if hasattr(self.backbone, 'ln_f'):
            if self.keep_blocks == config.backbone_blocks:
                self.ln_f = self.backbone.ln_f
            else:
                del self.backbone.ln_f

        self.attention_genes = getattr(config, 'finetune_attention_genes',
                                       False)

        if self.attention_genes:
            n_queries = config.finetune_attention_genes_queries
            self.gene_pool = GeneQueryAttentionPool(
                d_model=self.d_model,
                n_queries=n_queries,
                backbone_heads=config.backbone_heads,
                dropout=self.dropout,
            )
            gene_contribution = n_queries
        else:
            self.gene_pool = None
            gene_contribution = 1  # single mean-pooled vector

        flat_dim = (self.num_metadata_tokens + 1 +
                    gene_contribution) * self.d_model

        if self.head_type == 'linear':
            self.final_head = nn.Sequential(
                nn.Linear(flat_dim, self.n_output_units))
            self.final_head.apply(self.backbone._init_weights)
            nn.init.zeros_(self.final_head[-1].weight)
        elif self.head_type == 'MLP':
            self.final_head = nn.Sequential(
                nn.Linear(flat_dim, self.d_model), nn.GELU(),
                nn.LayerNorm(self.d_model), nn.Dropout(self.dropout),
                nn.Linear(self.d_model, self.n_output_units))
            self.final_head.apply(self.backbone._init_weights)
            nn.init.zeros_(self.final_head[-1].weight)
        elif self.head_type == 'gene_dot':
            if token_ids is None:
                raise ValueError(
                    "head_type='gene_dot' requires token_ids tensor.")

            n_tokens = token_ids.numel()

            self.register_buffer('token_ids',
                                 token_ids.long(),
                                 persistent=False)

            self.cell_proj = nn.Sequential(
                nn.Linear(flat_dim, self.d_model),
                nn.GELU(),
                nn.LayerNorm(self.d_model),
                nn.Dropout(self.dropout),
                nn.Linear(self.d_model, self.d_model),
            )
            self.cell_ln = nn.LayerNorm(self.d_model)
            self.cell_proj.apply(self.backbone._init_weights)

            self.logit_scale = nn.Parameter(
                torch.tensor(1.0 / math.sqrt(self.d_model)))

            if token_bias_init is not None:
                assert token_bias_init.numel() == n_tokens, (
                    f"token_bias_init length ({token_bias_init.numel()}) "
                    f"must match token set size ({n_tokens}).")
                bias_tensor = token_bias_init.float().clone()
            else:
                bias_tensor = torch.zeros(n_tokens, dtype=torch.float32)
            self.token_bias = nn.Parameter(bias_tensor)

        else:
            raise ValueError(
                f"Unknown finetune_head_type: '{self.head_type}'. "
                f"Expected 'linear', 'MLP', or 'gene_dot'.")

        self.to(self.device)

    def forward(self,
                tokens: torch.Tensor,
                valid_attention_mask: torch.Tensor,
                counts: torch.Tensor | None = None) -> torch.Tensor:
        # 1. Get token representations from the base model
        x = self._base_forward(tokens, valid_attention_mask, counts)

        x = self.ln_f(x)

        # 2. Separate metadata and gene tokens
        metadata_repr = x[:, :self.num_metadata_tokens + 1, :]
        gene_repr = x[:, self.num_metadata_tokens + 1:, :]

        # 3. Pool gene tokens -> fixed-size representation
        if self.attention_genes:
            gene_valid_mask = valid_attention_mask[:,
                                                   self.num_metadata_tokens +
                                                   1:]
            pooled_genes = self.gene_pool(
                gene_repr, gene_valid_mask)  # (B, n_queries * d_model)
        else:
            gene_mask = valid_attention_mask[:, self.num_metadata_tokens + 1:] \
                .unsqueeze(-1).to(gene_repr.dtype)
            pooled_genes = (gene_repr * gene_mask).sum(dim=1) / torch.clamp(
                gene_mask.sum(dim=1), min=1.0)  # (B, d_model)

        # 4. Flatten metadata block and concatenate with pooled genes
        metadata_flat = metadata_repr.reshape(metadata_repr.size(0),
                                              -1)  # (B, (num_meta+1)*d_model)
        combined_repr_flat = torch.cat([metadata_flat, pooled_genes], dim=-1)

        # 5. Head
        if self.head_type == 'linear':
            logits = self.final_head(combined_repr_flat)
        elif self.head_type == 'gene_dot':
            cell_repr = self.cell_proj(combined_repr_flat)
            cell_repr = self.cell_ln(cell_repr)
            target_embeds = self.backbone.token_emb(self.token_ids)
            logits = (cell_repr
                      @ target_embeds.T) * self.logit_scale + self.token_bias

        return logits


class GiCellBinaryClassifier(GiCellAnalyzer):
    """Cell-level binary classifier (single-logit head, sigmoid/BCE)."""

    def __init__(self,
                 pretrained_model: GeneIntelligence,
                 config: GiFinetuneConfig,
                 device: str,
                 token_ids: torch.Tensor | None = None,
                 token_bias_init: torch.Tensor | None = None) -> None:
        super().__init__(pretrained_model, config, device, token_ids,
                         token_bias_init)
        assert self.finetune_task == 'classification', "Binary classification model initialized with finetune_task != 'classification'"
        assert len(
            self.classes
        ) == 2, "Binary classification model initialized with classes != 2"

        for subset_keyword in ['train_metrics', 'val_metrics']:
            self.metrics[subset_keyword].update({
                'accuracy':
                BinaryAccuracy().to(self.device),
                'precision':
                BinaryPrecision().to(self.device),
                'recall':
                BinaryRecall().to(self.device),
                'f1':
                BinaryF1Score().to(self.device),
                'auroc':
                BinaryAUROC().to(self.device),
            })

    def compute_loss(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        logits = self(batch['tokens'], batch['valid_attention_mask'],
                      batch.get('counts'))
        targets = batch['ground_truth']
        loss = self.loss_fn(logits.squeeze(-1), targets.float())
        return loss

    def compute_loss_with_metrics(
            self,
            batch: dict[str, torch.Tensor],
            dataset_metrics_keyword: str = 'train_metrics') -> torch.Tensor:
        logits = self(batch['tokens'], batch['valid_attention_mask'],
                      batch.get('counts'))
        targets = batch['ground_truth']
        loss = self.loss_fn(logits.squeeze(-1), targets.float())

        probs = torch.sigmoid(logits.squeeze(-1))
        preds = (probs > 0.5).long()

        self._safe_update_metrics(
            dataset_metrics_keyword=dataset_metrics_keyword,
            targets=targets,
            preds=preds,
            probs=probs)
        return loss

    def logits(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        with torch.no_grad():
            logits = self(batch['tokens'], batch['valid_attention_mask'],
                          batch.get('counts'))
        return logits

    def predict_with_metrics(
        self,
        batch: dict[str, torch.Tensor],
        dataset_metrics_keyword: str = 'val_metrics'
    ) -> dict[str, torch.Tensor]:
        with torch.no_grad():
            logits = self(batch['tokens'], batch['valid_attention_mask'],
                          batch.get('counts'))
            targets = batch['ground_truth']
            loss = self.loss_fn(logits.squeeze(-1), targets.float())
            probs = torch.sigmoid(logits.squeeze(-1))
            preds = (probs > 0.5).long()
            self._safe_update_metrics(
                dataset_metrics_keyword=dataset_metrics_keyword,
                targets=targets,
                preds=preds,
                probs=probs)
        return {
            'pred': preds,
            'prob': probs,
            'ground_truth': targets,
            'loss': loss
        }


class GiCellMulticlassClassifier(GiCellAnalyzer):
    """Cell-level multiclass classifier (one logit per class, cross-entropy)."""

    def __init__(self,
                 pretrained_model: GeneIntelligence,
                 config: GiFinetuneConfig,
                 device: str,
                 token_ids: torch.Tensor | None = None,
                 token_bias_init: torch.Tensor | None = None) -> None:
        super().__init__(pretrained_model, config, device, token_ids,
                         token_bias_init)
        assert self.finetune_task == 'classification', "Multiclass classification model initialized with finetune_task != 'classification'"
        assert self.classes is None or len(
            self.classes
        ) > 2, "Multiclass classification model initialized with classes <= 2"

        for subset_keyword in ['train_metrics', 'val_metrics']:
            self.metrics[subset_keyword].update({
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
            })

    def compute_loss(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        logits = self(batch['tokens'], batch['valid_attention_mask'],
                      batch.get('counts'))
        targets = batch['ground_truth']
        loss = self.loss_fn(logits, targets)
        return loss

    def compute_loss_with_metrics(
            self,
            batch: dict[str, torch.Tensor],
            dataset_metrics_keyword: str = 'train_metrics') -> torch.Tensor:
        logits = self(batch['tokens'], batch['valid_attention_mask'],
                      batch.get('counts'))
        targets = batch['ground_truth']
        loss = self.loss_fn(logits, targets)

        preds = logits.argmax(dim=-1)

        self._safe_update_metrics(
            dataset_metrics_keyword=dataset_metrics_keyword,
            targets=targets,
            preds=preds)

        return loss

    def predict_with_metrics(
        self,
        batch: dict[str, torch.Tensor],
        dataset_metrics_keyword: str = 'val_metrics'
    ) -> dict[str, torch.Tensor]:
        with torch.no_grad():
            logits = self(batch['tokens'], batch['valid_attention_mask'],
                          batch.get('counts'))
            targets = batch['ground_truth']
            loss = self.loss_fn(logits, targets)
            probs = torch.softmax(logits, dim=-1).max(dim=-1).values
            preds = logits.argmax(dim=-1)
            self._safe_update_metrics(
                dataset_metrics_keyword=dataset_metrics_keyword,
                targets=targets,
                preds=preds)
        return {
            'pred': preds,
            'prob': probs,
            'ground_truth': targets,
            'loss': loss
        }


class GiCellMultilabelClassifier(GiCellAnalyzer):
    """Cell-level multi-label classifier (independent per-class logits, BCEWithLogits)."""

    def __init__(self,
                 pretrained_model: GeneIntelligence,
                 config: GiFinetuneConfig,
                 device: str,
                 token_ids: torch.Tensor | None = None,
                 token_bias_init: torch.Tensor | None = None) -> None:

        # Initialize base class
        super().__init__(pretrained_model, config, device, token_ids,
                         token_bias_init)

        assert self.finetune_task == 'classification', "Task must be 'classification'"
        self.num_classes = len(self.classes)

        # Enforce BCEWithLogitsLoss for multi-label stability
        self.loss_fn = torch.nn.BCEWithLogitsLoss(
            **config.finetune_loss_kwargs)

        # Initialize multi-label specific metrics
        for subset_keyword in ['train_metrics', 'val_metrics']:
            self.metrics[subset_keyword].update({
                'accuracy':
                MultilabelAccuracy(num_labels=self.num_classes).to(
                    self.device),
                'precision':
                MultilabelPrecision(num_labels=self.num_classes).to(
                    self.device),
                'recall':
                MultilabelRecall(num_labels=self.num_classes).to(self.device),
                'f1':
                MultilabelF1Score(num_labels=self.num_classes).to(self.device),
                'auroc':
                MultilabelAUROC(num_labels=self.num_classes).to(self.device)
            })

    def compute_loss(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        logits = self(batch['tokens'], batch['valid_attention_mask'],
                      batch.get('counts'))
        targets = batch['ground_truth']
        return self.loss_fn(logits, targets)

    def compute_loss_with_metrics(
            self,
            batch: dict[str, torch.Tensor],
            dataset_metrics_keyword: str = 'train_metrics') -> torch.Tensor:

        logits = self(batch['tokens'], batch['valid_attention_mask'],
                      batch.get('counts'))
        targets = batch['ground_truth']
        loss = self.loss_fn(logits, targets)

        # Apply sigmoid to get independent probabilities per class
        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).long()

        self._safe_update_metrics(
            dataset_metrics_keyword=dataset_metrics_keyword,
            targets=targets.long(),
            preds=preds,
            probs=probs)
        return loss

    def predict_with_metrics(
        self,
        batch: dict[str, torch.Tensor],
        dataset_metrics_keyword: str = 'val_metrics'
    ) -> dict[str, torch.Tensor]:

        with torch.no_grad():
            logits = self(batch['tokens'], batch['valid_attention_mask'],
                          batch.get('counts'))
            targets = batch['ground_truth']
            loss = self.loss_fn(logits, targets)

            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).long()

            self._safe_update_metrics(
                dataset_metrics_keyword=dataset_metrics_keyword,
                targets=targets.long(),
                preds=preds,
                probs=probs)

        return {
            'pred': preds,
            'prob': probs,
            'ground_truth': targets,
            'loss': loss
        }


class GiCellRegressor(GiCellAnalyzer):
    """Cell-level regressor with NaN-masked loss over one or more continuous targets."""

    def __init__(self,
                 pretrained_model: GeneIntelligence,
                 config: GiFinetuneConfig,
                 device: str,
                 token_ids: torch.Tensor = None,
                 token_bias_init: torch.Tensor = None) -> None:

        super().__init__(pretrained_model, config, device, token_ids,
                         token_bias_init)
        assert self.finetune_task == 'regression', "Regressor initialized with task != 'regression'"

        # Overwrite the loss function to ensure 'none' reduction.
        self.loss_fn = create_loss_fn(config.finetune_loss,
                                      {'reduction': 'none'})

        for subset_keyword in ['train_metrics', 'val_metrics']:
            self.metrics[subset_keyword].update({
                'mae':
                MeanAbsoluteError().to(self.device),
                'rmse':
                MeanSquaredError(squared=False).to(self.device)
            })

    def compute_loss(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        logits = self(batch['tokens'], batch['valid_attention_mask'],
                      batch.get('counts'))
        targets = batch['ground_truth'].float()
        predictions = logits.squeeze(-1)

        valid_mask = ~torch.isnan(targets)

        safe_targets = torch.where(valid_mask, targets,
                                   torch.zeros_like(targets))

        unreduced_loss = self.loss_fn(predictions.float(),
                                      safe_targets.float())

        # Reduce only the valid (non-NaN) entries
        return (unreduced_loss * valid_mask).sum() / torch.clamp(
            valid_mask.sum(), min=1.0)

    def compute_loss_with_metrics(
            self,
            batch: dict[str, torch.Tensor],
            dataset_metrics_keyword: str = 'train_metrics') -> torch.Tensor:

        # 1. Single forward pass
        logits = self(batch['tokens'], batch['valid_attention_mask'],
                      batch.get('counts'))
        targets = batch['ground_truth'].float()
        predictions = logits.squeeze(-1)

        # 2. Mask out NaNs
        valid_mask = ~torch.isnan(targets)
        safe_targets = torch.where(valid_mask, targets,
                                   torch.zeros_like(targets))

        # 3. Compute loss
        unreduced_loss = self.loss_fn(predictions.float(),
                                      safe_targets.float())
        loss = (unreduced_loss * valid_mask).sum() / torch.clamp(
            valid_mask.sum(), min=1.0)

        # 4. Update metrics using detached predictions to prevent graph memory leaks
        self._safe_update_metrics(
            dataset_metrics_keyword=dataset_metrics_keyword,
            targets=targets,
            preds=predictions.detach())
        return loss

    @torch.compiler.disable
    def _safe_update_metrics(self,
                             dataset_metrics_keyword: str,
                             targets: torch.Tensor,
                             preds: torch.Tensor | None = None,
                             probs: torch.Tensor | None = None) -> None:
        valid_mask = ~torch.isnan(targets)
        if valid_mask.sum() == 0:
            return

        filtered_targets = targets[valid_mask]
        filtered_preds = preds[valid_mask]

        for metric in self.metrics[dataset_metrics_keyword].values():
            metric.update(filtered_preds, filtered_targets)

    def predict_with_metrics(
        self,
        batch: dict[str, torch.Tensor],
        dataset_metrics_keyword: str = 'val_metrics'
    ) -> dict[str, torch.Tensor]:
        with torch.no_grad():
            logits = self(batch['tokens'], batch['valid_attention_mask'],
                          batch.get('counts'))
            targets = batch['ground_truth'].float()
            predictions = logits.squeeze(-1)

            valid_mask = ~torch.isnan(targets)
            safe_targets = torch.where(valid_mask, targets,
                                       torch.zeros_like(targets))
            unreduced_loss = self.loss_fn(predictions.float(),
                                          safe_targets.float())
            loss = (unreduced_loss * valid_mask).sum() / torch.clamp(
                valid_mask.sum(), min=1.0)

            self._safe_update_metrics(
                dataset_metrics_keyword=dataset_metrics_keyword,
                targets=targets,
                preds=predictions.detach())
        return {'pred': predictions, 'ground_truth': targets, 'loss': loss}


class GiGeneAnalyzer(GiFinetunerBase):
    """
    Base class for gene-level analysis tasks (token-level classification/regression).
    This model applies a final head to *each* token representation.
    """

    def __init__(self, pretrained_model: GeneIntelligence,
                 config: GiFinetuneConfig, device: str) -> None:

        super().__init__(pretrained_model, config, device)

        # This task predicts on the full sequence, so we keep/add a final LayerNorm
        self.ln_f = nn.LayerNorm(self.d_model)

        # Create ln_f if missing
        if hasattr(self.backbone, 'ln_f'):
            if self.keep_blocks == config.backbone_blocks:
                self.ln_f = self.backbone.ln_f
            else:
                del self.backbone.ln_f

        # Per-token head: operates on each token's (d_model,) representation.
        if self.head_type == 'linear':
            self.final_head = nn.Sequential(
                nn.Linear(self.d_model, self.n_output_units))
        elif self.head_type == 'MLP':
            self.final_head = nn.Sequential(
                nn.Linear(self.d_model, self.d_model), nn.GELU(),
                nn.LayerNorm(self.d_model), nn.Dropout(self.dropout),
                nn.Linear(self.d_model, self.n_output_units))
        else:
            raise ValueError(
                f"Unknown finetune_head_type for gene analyzer: "
                f"'{self.head_type}'. Expected 'linear' or 'MLP'.")

        self.final_head.apply(self.backbone._init_weights)
        nn.init.zeros_(self.final_head[-1].weight)
        self.to(self.device)

    def forward(self,
                tokens: torch.Tensor,
                valid_attention_mask: torch.Tensor,
                counts: torch.Tensor | None = None) -> torch.Tensor:
        # 1. Get token representations from the base model
        x = self._base_forward(tokens, valid_attention_mask, counts)

        # 2. Apply final layer norm
        x = self.ln_f(x)

        # 3. Apply prediction head to every token
        logits = self.final_head(x)  # Shape: (batch, seq_len, n_output_units)
        return logits


class GiGeneBinaryClassifier(GiGeneAnalyzer):
    """
    Gene-level (token-level) binary classifier.
    Predicts a 0/1 for each gene token based on a provided map.
    """

    def __init__(self, pretrained_model: GeneIntelligence,
                 config: GiFinetuneConfig, device: str) -> None:
        super().__init__(pretrained_model, config, device)
        assert self.n_output_units == 1, "Binary classifier must have n_output_units=1"
        assert self.finetune_task == 'classification', "Binary classification model initialized with finetune_task != 'classification'"

        for subset_keyword in ['train_metrics', 'val_metrics']:
            self.metrics[subset_keyword].update({
                'accuracy':
                BinaryAccuracy().to(self.device),
                'precision':
                BinaryPrecision().to(self.device),
                'recall':
                BinaryRecall().to(self.device),
                'f1':
                BinaryF1Score().to(self.device),
                'auroc':
                BinaryAUROC().to(self.device),
            })

    def compute_loss(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Computes loss, automatically masking out non-target tokens (e.g., pad,
        metadata, or genes not in the target map) which are expected
        to be np.nan in the ground_truth tensor.
        """
        logits = self(batch['tokens'], batch['valid_attention_mask'],
                      batch.get('counts'))
        targets = batch['ground_truth'].float()

        logits_squeezed = logits.squeeze(-1)  # Shape: (batch, seq_len)

        # Create mask for valid (non-NaN) targets
        valid_mask = ~torch.isnan(targets)

        # Compute loss only on valid targets
        if valid_mask.sum() == 0:
            # Avoids error if a batch has no valid targets
            return torch.tensor(0.0, device=self.device, requires_grad=True)

        loss = self.loss_fn(logits_squeezed[valid_mask], targets[valid_mask])
        return loss

    def compute_loss_with_metrics(
            self,
            batch: dict[str, torch.Tensor],
            dataset_metrics_keyword: str = 'train_metrics') -> torch.Tensor:

        logits = self(batch['tokens'], batch['valid_attention_mask'],
                      batch.get('counts'))
        targets = batch['ground_truth'].float()  # Shape: (batch, seq_len)

        logits_squeezed = logits.squeeze(-1)  # Shape: (batch, seq_len)

        # Create mask for valid (non-NaN) targets
        valid_mask = ~torch.isnan(targets)

        if valid_mask.sum() == 0:
            return torch.tensor(0.0, device=self.device, requires_grad=True)

        # Compute loss only on valid targets
        loss = self.loss_fn(logits_squeezed[valid_mask], targets[valid_mask])

        # Compute metrics only on valid targets
        probs = torch.sigmoid(logits_squeezed[valid_mask])
        preds = (probs > 0.5).long()
        valid_targets = targets[valid_mask].long()

        self._safe_update_metrics(
            dataset_metrics_keyword=dataset_metrics_keyword,
            targets=valid_targets,
            preds=preds,
            probs=probs)
        return loss

    def predict(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Performs inference, returning a 0/1 prediction for *every* token
        in the sequence.
        """
        with torch.no_grad():
            logits = self(batch['tokens'], batch['valid_attention_mask'],
                          batch.get('counts'))
            probs = torch.sigmoid(
                logits.squeeze(-1))  # Shape: (batch, seq_len)
            preds = (probs > 0.5).long()
        return preds

