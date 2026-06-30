#!/usr/bin/env python3
"""
Script Name: models.py
Description: Core Transformer architectures, embeddings, and loss functions
             for the Gene Intelligence backbone.

Usage:
    Imported as a module. Not intended for direct execution.
"""

# Standard Library Imports
import math
from typing import Any

# Third-Party Imports
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from flash_attn import flash_attn_varlen_func
from torch.profiler import record_function

# Local Application Imports
from geneintelligence.configs import GiConfig


def create_loss_fn(loss_name: str, loss_kwargs: dict[str, Any]) -> nn.Module:
    """Instantiate loss function from name + kwargs."""
    kwargs = loss_kwargs.copy()
    vocab_size = kwargs.pop('vocab_size', 32768)
    if loss_name == 'CrossEntropyLoss':
        return nn.CrossEntropyLoss(**kwargs)
    elif loss_name == 'BCEWithLogitsLoss':
        return nn.BCEWithLogitsLoss(**kwargs)
    elif loss_name == 'MSELoss':
        return nn.MSELoss(**kwargs)
    elif loss_name in ['L1Loss', 'MAE']:
        return nn.L1Loss(**kwargs)
    else:
        raise ValueError(f"Unknown loss function: {loss_name}")


class SwiGLU(nn.Module):
    """SwiGLU activation: splits the input in half and gates one half with SiLU."""

    def __init__(self) -> None:
        super().__init__()
        self.silu = nn.SiLU()  # Swish activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """SwiGLU activation: SiLU(x) * Linear(x)"""
        x1, x2 = torch.chunk(x, 2, dim=-1)  # Split tensor into two halves
        return self.silu(x1) * x2  # Apply activation to one half and multiply


class GiEmbeddings(nn.Module):
    """Adds token-type, optional positional, and optional count signals to token embeddings."""

    def __init__(self, config: GiConfig) -> None:
        super().__init__()
        d_model = config.backbone_d_model
        max_len = config.max_len
        dropout = config.backbone_dropout
        num_metadata = len(config.metadata_cols)

        # 1. Embeddings & Layers
        self.token_type_emb = nn.Embedding(num_metadata + 2, d_model)
        self.LayerNorm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        # 2. Pre-calculate Token Type IDs (Segment IDs)
        # Shape: (1, max_len)
        # Default to '0' (Cell) for everything
        type_ids = torch.full((1, max_len), fill_value=0, dtype=torch.long)

        # Set indices 1 to 1+N to '1' (Metadata Tokens)
        for i in range(num_metadata):
            type_ids[0, 1 + i] = 1 + i  # types 1, 2, 3, 4, 5, 6 etc
        type_ids[0, 1 + num_metadata:] = num_metadata + 1

        # Register as a buffer so it moves to GPU automatically with the model
        self.register_buffer("type_ids_template", type_ids, persistent=False)
        self.use_positional_embeddings = getattr(
            config, 'backbone_use_positional_embeddings', True)

        if self.use_positional_embeddings:
            # 3. Pre-calculate Positional Encodings
            pe = torch.zeros(max_len, d_model)
            position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(
                torch.arange(0, d_model, 2).float() *
                (-math.log(10000.0) / d_model))
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            pe = pe.unsqueeze(0)
            self.register_buffer("pe", pe, persistent=False)

            if getattr(config, 'scale_positional_encodings', False):
                self.pos_scale = nn.Embedding(max_len, 1)
                nn.init.constant_(self.pos_scale.weight, math.log(math.e - 1))
                self.pos_scale._skip_init = True
            else:
                self.register_buffer("pos_scale",
                                     torch.ones(max_len, 1),
                                     persistent=False)

        self.use_counts = getattr(config, 'use_counts', False)

        if self.use_counts:
            # scalar log1p(count) -> d_model vector, added per token
            self.count_proj = nn.Linear(1, d_model, bias=False)
            # start near zero to introduce count signal gradually
            nn.init.normal_(self.count_proj.weight, std=0.001)
            self.count_proj._skip_init = True

    def forward(self,
                token_embeddings: torch.Tensor,
                counts: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            token_embeddings: (batch, seq_len, d_model)
            counts: (batch, seq_len) raw integer counts, or None
        """
        batch_size, seq_len, _ = token_embeddings.size()

        # A. Token type embeddings
        type_ids = self.type_ids_template[:, :seq_len].expand(batch_size, -1)
        x = token_embeddings + self.token_type_emb(type_ids)

        # B. Positional embeddings (optional)
        if self.use_positional_embeddings:
            pe = self.pe[:, :seq_len]
            if isinstance(self.pos_scale, nn.Embedding):
                positions = torch.arange(seq_len,
                                         device=token_embeddings.device)
                scale = F.softplus(self.pos_scale(positions))
            else:
                scale = self.pos_scale[:seq_len]
            x = x + pe * scale

        # C. Count embeddings (optional)
        if self.use_counts:
            # log1p transform to compress dynamic range, keep as float
            safe_counts = torch.clamp(counts.float(), min=0.0)
            count_signal = torch.log1p(safe_counts).unsqueeze(
                -1)  # (batch, seq_len, 1)
            x = x + self.count_proj(
                count_signal
            )  # projects (batch, seq_len, 1) -> (batch, seq_len, d_model)

        # D. Norm & Dropout
        x = self.LayerNorm(x)
        x = self.dropout(x)

        return x


class GiFlashSelfAttention(nn.Module):
    """Multi-head self-attention over packed, variable-length sequences via FlashAttention."""

    def __init__(self,
                 d_model: int,
                 num_heads: int,
                 dropout: float = 0.1) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.dropout = nn.Dropout(dropout)
        self.qkv_proj = nn.Linear(d_model, d_model * 3, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x_packed: torch.Tensor, cu_seqlens: torch.Tensor,
                max_seqlen: int) -> torch.Tensor:
        # x_packed: (total_tokens, d_model)
        total_tokens = x_packed.shape[0]

        qkv = self.qkv_proj(x_packed)
        qkv = qkv.reshape(total_tokens, 3, self.num_heads, self.head_dim)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]

        out = flash_attn_varlen_func(q,
                                     k,
                                     v,
                                     cu_seqlens_q=cu_seqlens,
                                     cu_seqlens_k=cu_seqlens,
                                     max_seqlen_q=max_seqlen,
                                     max_seqlen_k=max_seqlen,
                                     dropout_p=0.0,
                                     causal=False)
        out = self.out_proj(out.reshape(total_tokens, -1))
        return self.dropout(out)


class GiFeedForward(nn.Module):
    """FeedForward (MLP) sub-layer of a Transformer block, using a SwiGLU activation."""

    def __init__(self,
                 d_model: int,
                 ff_dim: int,
                 dropout: float = 0.1) -> None:
        super().__init__()
        self.fc1 = nn.Linear(d_model, ff_dim, bias=False)
        self.act = SwiGLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(ff_dim // 2, d_model, bias=False)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout2(x)
        return x


class GiTransformerBlock(nn.Module):
    """A single pre-norm Transformer block (attention + feedforward with residuals)."""

    def __init__(self,
                 d_model: int,
                 num_heads: int,
                 ff_dim: int,
                 dropout: float = 0.1,
                 layer_idx: int = 0) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = GiFlashSelfAttention(d_model, num_heads, dropout=dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = GiFeedForward(d_model, ff_dim, dropout=dropout)
        self.layer_idx = layer_idx
        self.attn.out_proj._is_residual_projection = True
        self.attn.out_proj._residual_depth = 2 * layer_idx + 1
        self.ff.fc2._is_residual_projection = True
        self.ff.fc2._residual_depth = 2 * layer_idx + 2

    def forward(self,
                x: torch.Tensor,
                cu_seqlens: torch.Tensor | None = None,
                max_seqlen: int | None = None) -> torch.Tensor:
        attn_output = self.attn(self.ln1(x),
                                cu_seqlens=cu_seqlens,
                                max_seqlen=max_seqlen)
        x = x + attn_output
        ff_output = self.ff(self.ln2(x))
        x = x + ff_output
        return x


class GeneIntelligence(nn.Module):
    """Transformer backbone for masked modeling of gene-expression token sequences.

    Embeds ``<cell>`` + metadata + gene tokens, runs them through stacked
    Transformer blocks, and predicts masked gene tokens (and optionally masked
    counts) via tied prediction heads.
    """

    def __init__(self,
                 config: GiConfig,
                 device: torch.device | None = None,
                 gene_metadata: pd.DataFrame | None = None) -> None:
        super().__init__()
        if device is None:
            device = torch.device(
                'cuda' if torch.cuda.is_available() else 'cpu')

        self.config = config

        self.token_emb = nn.Embedding(config.vocab_size,
                                      config.backbone_d_model,
                                      padding_idx=1)

        self.embeddings = GiEmbeddings(config)

        self.num_metadata_tokens = len(config.metadata_cols)

        self.blocks = nn.ModuleList([
            GiTransformerBlock(config.backbone_d_model,
                               config.backbone_heads,
                               config.backbone_ff_dim,
                               config.backbone_dropout,
                               layer_idx=i)
            for i in range(config.backbone_blocks)
        ])
        # Final layer normalization
        self.ln_f = nn.LayerNorm(config.backbone_d_model)

        self.prediction_head = nn.Sequential(
            nn.Linear(config.backbone_d_model, config.backbone_d_model),
            nn.GELU(), nn.LayerNorm(config.backbone_d_model),
            nn.Dropout(config.backbone_dropout),
            nn.Linear(config.backbone_d_model, config.vocab_size))

        if getattr(config, 'pretrain_mask_counts_prob', 0.0) > 0.0:
            self.count_prediction_head = nn.Sequential(
                nn.Linear(config.backbone_d_model, config.backbone_d_model),
                nn.GELU(), nn.LayerNorm(config.backbone_d_model),
                nn.Dropout(config.backbone_dropout),
                nn.Linear(config.backbone_d_model, 1))

        # --- Dynamic Loss Balancing Initialization ---
        if getattr(config, 'pretrain_dynamic_loss_weighting', False):
            # We learn eta. Effective sigma = softplus(eta) + floor.
            self.eta_tokens = nn.Parameter(torch.tensor(1.5))
            self.eta_counts = nn.Parameter(torch.tensor(0.5))

        self.apply(self._init_weights)
        # tying the weights of the embedding and prediction head
        self.prediction_head[-1].weight = self.token_emb.weight
        if gene_metadata is not None:
            self._initialize_prediction_biases(config, gene_metadata, device)

        loss_kwargs = config.backbone_loss_kwargs.copy(
        ) if config.backbone_loss_kwargs else {}
        loss_kwargs['vocab_size'] = config.vocab_size
        self.loss_fn = create_loss_fn(config.backbone_loss, loss_kwargs)
        self.device = device
        self.to(device)

    def _init_weights(self, module: nn.Module) -> None:
        std = 0.02
        if getattr(module, '_skip_init', False):
            return

        if isinstance(module, nn.Linear):
            if getattr(module, '_is_residual_projection', False):
                depth = getattr(module, '_residual_depth', 1)
                scale = 1 / math.sqrt(depth)
                std *= scale

            torch.nn.init.trunc_normal_(module.weight,
                                        mean=0.0,
                                        std=std,
                                        a=-2 * std,
                                        b=2 * std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

        elif isinstance(module, nn.Embedding):
            torch.nn.init.trunc_normal_(module.weight,
                                        mean=0.0,
                                        std=std,
                                        a=-2 * std,
                                        b=2 * std)

            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def _initialize_prediction_biases(self, config: GiConfig,
                                      gene_metadata: pd.DataFrame,
                                      device: torch.device) -> None:

        nnz = torch.tensor(gene_metadata['nnz'].values,
                           dtype=torch.float32,
                           device=device)
        gene_token_ids = torch.tensor(gene_metadata['token_id'].values,
                                      dtype=torch.long,
                                      device=device)
        bias_vector = torch.full((config.vocab_size, ),
                                 -1e4,
                                 dtype=torch.float32,
                                 device=device)

        # Log-Frequency for standard MLM (Softmax/CrossEntropy)
        total_gene_tokens = nnz.sum()
        f = torch.clamp(nnz / total_gene_tokens, min=1e-9)
        target_biases = torch.log(f)

        bias_vector[gene_token_ids] = target_biases

        with torch.no_grad():
            self.prediction_head[-1].bias.copy_(bias_vector)

        if hasattr(self, 'count_prediction_head'):
            total_counts = torch.tensor(gene_metadata['countsum'].values,
                                        dtype=torch.float32,
                                        device=device).sum()

            # total_gene_tokens is the sum of nnz (total non-zero entries in the dataset)
            avg_raw_count = total_counts / total_gene_tokens
            empirical_mean_log1p_count = torch.log1p(avg_raw_count).item()

            with torch.no_grad():
                # Set the final linear layer's bias to the empirical mean and the weights to 0
                self.count_prediction_head[-1].bias.fill_(
                    empirical_mean_log1p_count)
                self.count_prediction_head[-1].weight.fill_(0.0)

    def forward(
        self,
        tokens: torch.Tensor,
        counts: torch.Tensor,
        mask_positions: torch.Tensor | None = None,
        count_mask_positions: torch.Tensor | None = None,
        valid_attention_mask: torch.Tensor | None = None,
        only_masked: bool | None = None
    ) -> torch.Tensor | None | tuple[torch.Tensor | None, torch.Tensor | None]:
        """Run the backbone and prediction head(s).

        With ``only_masked`` set (or during training) returns predictions for
        masked positions only; otherwise returns logits for the full sequence.
        When a count head is present, returns a ``(token_logits, count_preds)``
        tuple.
        """

        batch, seq_len = tokens.shape
        x = self.token_emb(tokens)
        x = self.embeddings(x, counts=counts)

        lengths = valid_attention_mask.sum(dim=1, dtype=torch.int32)
        cu_seqlens = F.pad(lengths.cumsum(0, dtype=torch.int32), (1, 0))

        max_seqlen = seq_len

        mask_flat = valid_attention_mask.reshape(-1)
        x_packed = x.reshape(-1, self.token_emb.embedding_dim)[mask_flat]

        # Transformer Blocks
        for block in self.blocks:
            x_packed = block(x_packed,
                             cu_seqlens=cu_seqlens,
                             max_seqlen=max_seqlen)

        # Check if we only need masked tokens
        is_training_masked = (only_masked is True) or (only_masked is not False
                                                       and self.training)
        if is_training_masked and (mask_positions is not None
                                   or count_mask_positions is not None):
            logits = None
            count_preds = None

            # 1. Extract and predict only the masked tokens for genes
            if mask_positions is not None:
                packed_mask = mask_positions.reshape(-1)[mask_flat]
                masked_outputs = self.ln_f(x_packed[packed_mask])
                logits = self.prediction_head(masked_outputs)

            # 2. Independently extract and predict only the masked tokens for counts
            if hasattr(self, 'count_prediction_head'
                       ) and count_mask_positions is not None:
                packed_count_mask = count_mask_positions.reshape(-1)[mask_flat]
                count_masked_outputs = self.ln_f(x_packed[packed_count_mask])
                count_preds = self.count_prediction_head(
                    count_masked_outputs).squeeze(-1)

            if hasattr(self, 'count_prediction_head'):
                return logits, count_preds
            return logits

        else:
            x = torch.zeros(batch * seq_len,
                            self.token_emb.embedding_dim,
                            device=tokens.device,
                            dtype=x_packed.dtype)
            x[mask_flat] = x_packed
            x = x.reshape(batch, seq_len, -1)

            x = self.ln_f(x)
            logits = self.prediction_head(x)

            if hasattr(self, 'count_prediction_head'):
                count_preds = self.count_prediction_head(x).squeeze(-1)
                return logits, count_preds

            return logits

    def compute_loss(
        self, batch: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
               torch.Tensor]:
        """Compute masked-token CE loss and optional count-regression loss.

        Returns ``(ce_loss, reg_loss, reg_mae, accuracy, top_k_accuracy)``.
        """

        # Extract base tensors
        tokens = batch.get('masked_tokens',
                           batch.get('tokens')).to(self.device,
                                                   non_blocking=True)
        valid_attention_mask = batch['valid_attention_mask'].to(
            self.device, non_blocking=True)
        counts = batch.get('masked_counts',
                           batch.get('counts')).to(self.device,
                                                   non_blocking=True)

        # Extract optional masks
        mask_positions = batch.get('mask_positions')
        if mask_positions is not None:
            mask_positions = mask_positions.to(self.device, non_blocking=True)

        count_mask_positions = batch.get('count_mask_positions')
        if count_mask_positions is not None:
            count_mask_positions = count_mask_positions.to(self.device,
                                                           non_blocking=True)

        predicting_counts = hasattr(self, 'count_prediction_head')

        with record_function("Model_Forward_Pass"):
            forward_out = self(tokens,
                               counts=counts,
                               mask_positions=mask_positions,
                               count_mask_positions=count_mask_positions,
                               valid_attention_mask=valid_attention_mask,
                               only_masked=True)

        if predicting_counts:
            masked_logits, count_preds = forward_out
        else:
            masked_logits = forward_out
            count_preds = None

        accuracy = torch.tensor(0.0, device=self.device)
        reg_mae = torch.tensor(0.0, device=self.device)
        top_k_accuracy = torch.tensor(0.0, device=self.device)

        with record_function("Loss_Computation"):
            ce_loss = torch.tensor(0.0, device=self.device)
            reg_loss = torch.tensor(0.0, device=self.device)

            # 1. Gene Token Loss (Raw)
            if mask_positions is not None and masked_logits is not None:
                original_tokens = batch['tokens'].to(self.device,
                                                     non_blocking=True)
                masked_targets = original_tokens[mask_positions].long()
                ce_loss = self.loss_fn(masked_logits, masked_targets)

                with torch.no_grad():
                    accuracy = (masked_logits.argmax(
                        dim=-1) == masked_targets).float().mean()

                    # 1. Get number of masked genes per cell: shape (batch_size,)
                    k_per_cell = mask_positions.sum(dim=-1)

                    # 2. Map k to each token. E.g. if cell 0 has 2 masks and cell 1 has 3:
                    # k_per_cell = [2, 3] -> repeat_interleave -> [2, 2, 3, 3, 3]
                    k_per_token = torch.repeat_interleave(
                        k_per_cell, k_per_cell)

                    # 3. Extract target logit: shape (M, 1)
                    target_logits = masked_logits.gather(
                        1, masked_targets.unsqueeze(-1))

                    # 4. Fused Rank Calculation: Fuses into a single Triton reduction kernel
                    ranks = (masked_logits > target_logits).sum(dim=-1) + 1

                    # 5. Calculate final dynamic top-k accuracy
                    top_k_accuracy = (ranks <= k_per_token).float().mean()

            # 2. Count Regression Loss (Raw)
            if predicting_counts and count_mask_positions is not None and count_preds is not None:
                original_counts = batch['counts'].to(self.device,
                                                     non_blocking=True)
                target_counts = original_counts[count_mask_positions].float()

                # Apply log1p to compress dynamic range, matching the embeddings
                target_counts_log = torch.log1p(
                    torch.clamp(target_counts, min=0.0))
                reg_loss = F.smooth_l1_loss(count_preds,
                                            target_counts_log,
                                            beta=1.0)
                with torch.no_grad():
                    reg_mae = F.l1_loss(count_preds, target_counts_log)

            # 3. Dynamic Weighting
            if getattr(self.config, 'pretrain_dynamic_loss_weighting',
                       False) and hasattr(self, 'eta_tokens'):
                # Extract floors from config
                s_t_floor = getattr(self.config,
                                    'pretrain_token_uncertainty_floor', 1.0)
                s_c_floor = getattr(self.config,
                                    'pretrain_count_uncertainty_floor', 0.1)

                # Compute effective sigmas
                sigma_t = F.softplus(self.eta_tokens) + s_t_floor
                sigma_c = F.softplus(self.eta_counts) + s_c_floor

                # Apply Bayesian Multi-Task Weighting: L / (C * sigma^2) + log(sigma^2)
                ce_loss = (ce_loss / (sigma_t**2)) + torch.log(sigma_t**2)
                reg_loss = (reg_loss /
                            (2 * sigma_c**2)) + torch.log(sigma_c**2)

        return ce_loss, reg_loss, reg_mae, accuracy, top_k_accuracy

    @torch.no_grad()
    def encode(self,
               tokens: torch.Tensor,
               valid_attention_mask: torch.Tensor,
               counts: torch.Tensor | None = None,
               layer_norm: bool = True) -> torch.Tensor:
        """Full per-token hidden states (B, S, d_model).
        Row order follows the input: <cell>, metadata, then genes in sorter order.

        Note:
            If the backbone was trained with `use_counts=True`, actual count
            tensors must be provided. Zero-filling is only mathematically
            equivalent if the model was trained with `use_counts=False`.
        """
        self.eval()
        batch, seq_len = tokens.shape
        if counts is None:
            counts = torch.zeros_like(tokens)

        x = self.token_emb(tokens)
        x = self.embeddings(x, counts=counts)

        lengths = valid_attention_mask.sum(dim=1, dtype=torch.int32)
        cu_seqlens = F.pad(lengths.cumsum(0, dtype=torch.int32), (1, 0))
        mask_flat = valid_attention_mask.reshape(-1)
        x_packed = x.reshape(-1, self.token_emb.embedding_dim)[mask_flat]

        for block in self.blocks:
            x_packed = block(x_packed,
                             cu_seqlens=cu_seqlens,
                             max_seqlen=seq_len)

        x = torch.zeros(batch * seq_len,
                        self.token_emb.embedding_dim,
                        device=tokens.device,
                        dtype=x_packed.dtype)
        x[mask_flat] = x_packed
        x = x.reshape(batch, seq_len, -1)
        if layer_norm:
            x = self.ln_f(x)
        return x
