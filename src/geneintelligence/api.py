#!/usr/bin/env python3
"""
Script Name: api.py
Description: High-level API for inference and embeddings.

Usage:
    Imported as a module. Not intended for direct execution.
"""
# Standard Library Imports
from typing import Any

# Third-Party Imports
import numpy as np
import pandas as pd
import torch
import yaml
from anndata import AnnData
from torch.utils.data import DataLoader

# Local Application Imports
from geneintelligence.configs import GiConfig
from geneintelligence.datasets import AnnDataDataset, gi_collate, get_sorter
from geneintelligence.embeddings import (
    concatenate_embedding_adata,
    pivot_obs_column_to_adata,
    pool_embedding_adata,
)
from geneintelligence.models import GeneIntelligence


class APICollate:
    """Picklable collate wrapper to ensure safe multiprocessing in spawn contexts."""

    def __init__(self, pad_token: int, return_counts: bool) -> None:
        self.pad_token = pad_token
        self.return_counts = return_counts

    def __call__(self, batch: list[Any]) -> dict[str, torch.Tensor]:
        return gi_collate(batch,
                          pad_token=self.pad_token,
                          return_counts=self.return_counts)


class GeneIntelligenceModel:
    """
    High-level API for Gene Intelligence inference and embeddings.
    """

    def __init__(self,
                 checkpoint_path: str,
                 config_path: str,
                 token_dictionary: dict[str, int] | None = None,
                 gene_metadata: pd.DataFrame | None = None,
                 gene_medians: pd.Series | None = None,
                 device: str | torch.device | None = None) -> None:
        self.device = torch.device(
            device or ('cuda' if torch.cuda.is_available() else 'cpu'))
        self.token_dictionary = token_dictionary
        self.gene_metadata = gene_metadata
        if gene_medians is not None and gene_metadata is not None:
            gene_medians = gene_medians.loc[gene_metadata.index]
        self.gene_medians = gene_medians

        # 1. Parse config and build the base model architecture
        self.config, self.model = self._build_model_from_config(config_path)

        # 2. Load weights
        checkpoint = torch.load(checkpoint_path,
                                map_location=self.device,
                                weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])

        self.model.eval()
        self.model.to(self.device)

    def _build_model_from_config(
            self, config_path: str) -> tuple[GiConfig, GeneIntelligence]:
        """Inspects the YAML and loads the base model config."""
        with open(config_path, 'r') as f:
            raw_config = yaml.unsafe_load(f)

        config = GiConfig._from_dict(raw_config)
        backbone = GeneIntelligence(config, device=self.device)
        return config, backbone

    @torch.no_grad()
    def embed_cells(self,
                    adata: AnnData,
                    batch_size: int = 32,
                    num_workers: int = 0) -> AnnData:
        """
        Extracts cell-level representations.
        Returns a new AnnData object with embeddings in .X and original metadata in .obs.
        """
        if self.token_dictionary is None or self.gene_metadata is None:
            raise ValueError(
                "token_dictionary and gene_metadata must be provided during initialization."
            )

        sorter = get_sorter(
            strategy_name=self.config.pretrain_preprocessing_strategy,
            gene_ranks=self.gene_metadata.cohort_rank,
            gene_medians=self.gene_medians,
            vocab_size=len(self.token_dictionary))

        dataset = AnnDataDataset(adata=adata,
                                 dictionary=self.token_dictionary,
                                 gene_metadata=self.gene_metadata,
                                 max_len=self.config.max_len,
                                 sorter=sorter,
                                 metadata_cols=self.config.metadata_cols,
                                 fill_metadata=True,
                                 return_counts=self.config.use_counts)

        api_collate = APICollate(pad_token=self.token_dictionary.get('<pad>'),
                                 return_counts=self.config.use_counts)

        dataloader = DataLoader(dataset,
                                batch_size=batch_size,
                                collate_fn=api_collate,
                                num_workers=num_workers)

        backbone = self.model
        metadata_len = len(self.config.metadata_cols)
        d_model = backbone.config.backbone_d_model
        use_cuda = self.device.type == 'cuda'
        all_embeddings = []

        for batch in dataloader:
            batch = {
                k: v.to(self.device, non_blocking=True)
                for k, v in batch.items()
            }
            with torch.autocast(device_type=self.device.type,
                                dtype=torch.bfloat16,
                                enabled=use_cuda):
                hidden = backbone.encode(
                    tokens=batch['tokens'].long(),
                    valid_attention_mask=batch['valid_attention_mask'],
                    counts=batch.get('counts'),
                    layer_norm=True)
            hidden = hidden.float()

            valid = batch['valid_attention_mask']
            B, S, D = hidden.shape

            # 1. CLS Token
            cls_repr = hidden[:, 0, :]

            # 2. Metadata Tokens
            if metadata_len > 0:
                meta_repr = hidden.new_zeros((B, metadata_len, D))
                meta_end = min(1 + metadata_len, S)
                if meta_end > 1:
                    meta_repr[:, :meta_end - 1, :] = (
                        hidden[:, 1:meta_end, :] *
                        valid[:, 1:meta_end].unsqueeze(-1).to(hidden.dtype))
                meta_repr = meta_repr.reshape(B, metadata_len * D)
            else:
                meta_repr = hidden.new_zeros((B, 0))

            # 3. Mean-Pooled Gene Tokens
            gene_start = min(1 + metadata_len, S)
            if gene_start < S:
                gene_hidden = hidden[:, gene_start:, :]
                gene_mask = valid[:, gene_start:]
                gene_den = gene_mask.sum(dim=1).clamp_min(1).to(
                    hidden.dtype).unsqueeze(1)
                gene_repr = (gene_hidden * gene_mask.unsqueeze(-1).to(
                    hidden.dtype)).sum(dim=1) / gene_den
            else:
                gene_repr = hidden.new_zeros((B, D))

            # Concatenate: cls_metadata_gene_mean_concat
            batch_emb = torch.cat([cls_repr, meta_repr, gene_repr], dim=1)
            all_embeddings.append(batch_emb.cpu().numpy())

        # Construct new AnnData
        X_emb = np.concatenate(all_embeddings, axis=0).astype(np.float32)
        return AnnData(X=X_emb, obs=adata.obs.copy())

    @torch.no_grad()
    def embed_genes(self,
                    adata: AnnData,
                    target_genes: list[str] | None = None,
                    batch_size: int = 32,
                    num_workers: int = 0) -> AnnData:
        """
        Extracts contextual gene-level (token) embeddings.
        Returns a new AnnData where .X contains the token embeddings.
        The .obs will contain 'original_cell_id' and 'gene_id'.
        """
        if self.token_dictionary is None or self.gene_metadata is None:
            raise ValueError(
                "token_dictionary and gene_metadata must be provided during initialization."
            )

        sorter = get_sorter(
            strategy_name=self.config.pretrain_preprocessing_strategy,
            gene_ranks=self.gene_metadata.cohort_rank,
            gene_medians=self.gene_medians,
            vocab_size=len(self.token_dictionary))

        dataset = AnnDataDataset(adata=adata,
                                 dictionary=self.token_dictionary,
                                 gene_metadata=self.gene_metadata,
                                 max_len=self.config.max_len,
                                 sorter=sorter,
                                 metadata_cols=self.config.metadata_cols,
                                 fill_metadata=True,
                                 return_counts=self.config.use_counts)

        api_collate = APICollate(pad_token=self.token_dictionary.get('<pad>'),
                                 return_counts=self.config.use_counts)

        dataloader = DataLoader(dataset,
                                batch_size=batch_size,
                                collate_fn=api_collate,
                                num_workers=num_workers)

        backbone = self.model
        token_to_gene = {v: k for k, v in self.token_dictionary.items()}

        target_token_ids = None
        if target_genes is not None:
            target_token_ids = {
                self.token_dictionary[g]
                for g in target_genes if g in self.token_dictionary
            }

        all_embeddings = []
        obs_rows = []

        gene_start_idx = 1 + len(self.config.metadata_cols)
        global_cell_idx = 0
        use_cuda = self.device.type == 'cuda'

        for batch in dataloader:
            batch = {
                k: v.to(self.device, non_blocking=True)
                for k, v in batch.items()
            }

            with torch.autocast(device_type=self.device.type,
                                dtype=torch.bfloat16,
                                enabled=use_cuda):
                hidden = backbone.encode(
                    tokens=batch['tokens'].long(),
                    valid_attention_mask=batch['valid_attention_mask'],
                    counts=batch.get('counts'),
                    layer_norm=True)

            hidden = hidden.float()

            valid_mask = batch['valid_attention_mask']
            tokens = batch['tokens']

            for i in range(hidden.size(0)):
                current_cell_id = adata.obs_names[global_cell_idx]
                global_cell_idx += 1

                valid_len = valid_mask[i].sum().item()
                if valid_len <= gene_start_idx:
                    continue

                cell_gene_tokens = tokens[
                    i, gene_start_idx:valid_len].cpu().numpy()
                cell_gene_embs = hidden[
                    i, gene_start_idx:valid_len, :].cpu().numpy()

                if target_token_ids is not None:
                    keep_mask = np.isin(cell_gene_tokens,
                                        list(target_token_ids))
                    cell_gene_tokens = cell_gene_tokens[keep_mask]
                    cell_gene_embs = cell_gene_embs[keep_mask]

                if len(cell_gene_tokens) == 0:
                    continue

                all_embeddings.append(cell_gene_embs)

                for token_id in cell_gene_tokens:
                    gene_name = token_to_gene.get(token_id,
                                                  f"UNKNOWN_{token_id}")
                    obs_rows.append({
                        "original_cell_id": current_cell_id,
                        "gene_id": gene_name
                    })

        if not all_embeddings:
            return AnnData(
                X=np.empty((0, backbone.config.backbone_d_model),
                           dtype=np.float32),
                obs=pd.DataFrame(columns=["original_cell_id", "gene_id"]))

        X_emb = np.concatenate(all_embeddings, axis=0).astype(np.float32)
        obs_df = pd.DataFrame(obs_rows)
        obs_df.index = obs_df["original_cell_id"].astype(
            str) + "_" + obs_df["gene_id"].astype(str)

        return AnnData(X=X_emb, obs=obs_df)

    @torch.no_grad()
    def embed_samples(self,
                      adata: AnnData,
                      sample_col: str,
                      groupby_col: str,
                      concatenation_order: list[str] | None = None,
                      batch_size: int = 32,
                      num_workers: int = 0,
                      concatenate_fractions: bool = False,
                      fraction_label: str = "cell_type_fraction") -> AnnData:
        """
        Extracts sample-level embeddings by first computing cell-level embeddings,
        pooling them by (sample, groupby_col), and concatenating them horizontally.
        Returns a new AnnData object with one row per unique sample.

        If concatenate_fractions=True, one extra feature per group is appended:
        the fraction of the sample's cells assigned to that group (groupby_col
        value), in the same concatenation_order as the embedding blocks. Cell
        types absent from a sample get a zero embedding block and a 0 fraction.
        """
        if sample_col not in adata.obs.columns:
            raise KeyError(
                f"Sample column '{sample_col}' not found in adata.obs")
        if groupby_col not in adata.obs.columns:
            raise KeyError(
                f"Groupby column '{groupby_col}' not found in adata.obs")

        cell_adata = self.embed_cells(adata,
                                      batch_size=batch_size,
                                      num_workers=num_workers)

        group_by_columns = [sample_col, groupby_col]
        pooled_adata = pool_embedding_adata(cell_adata,
                                            group_by_columns=group_by_columns)

        if concatenation_order is None:
            concatenation_order = sorted(
                adata.obs[groupby_col].dropna().unique().astype(str).tolist())

        sample_adata = concatenate_embedding_adata(
            adata=pooled_adata,
            group_by_columns=[sample_col],
            concatenation_axis=groupby_col,
            concatenation_order=concatenation_order)

        if not concatenate_fractions:
            return sample_adata

        count_col = f"{sample_col}_{groupby_col}_count"
        sample_total = pooled_adata.obs.groupby(
            sample_col, observed=True)[count_col].transform("sum")
        pooled_adata.obs[fraction_label] = (
            pooled_adata.obs[count_col] /
            sample_total.replace(0, np.nan)).fillna(0.0)

        # Pivot the fraction column horizontally (one column per group value),
        # using the same sample-index convention as concatenate_embedding_adata
        # so rows correspond one-to-one.
        fraction_adata = pivot_obs_column_to_adata(
            adata=pooled_adata,
            group_by_columns=[sample_col],
            concatenation_axis=groupby_col,
            concatenation_order=concatenation_order,
            target_column=fraction_label,
            fill_value=0.0)

        # Align by sample id, then concatenate features: [embeddings | fractions].
        fraction_adata = fraction_adata[list(sample_adata.obs_names)].copy()
        combined_x = np.concatenate([
            np.asarray(sample_adata.X, dtype=np.float32),
            np.asarray(fraction_adata.X, dtype=np.float32),
        ],
                                    axis=1)
        combined_var = pd.DataFrame(index=list(sample_adata.var_names) +
                                    list(fraction_adata.var_names))
        return AnnData(X=combined_x.astype(np.float32),
                       obs=sample_adata.obs.copy(),
                       var=combined_var)

    def detect_doublets(self, adata: AnnData) -> AnnData:
        """
        Detects doublets in the dataset.
        
        Note: This feature is currently under development and will be available 
        in a future release.
        """
        raise NotImplementedError(
            "Doublet detection is currently under development and is not yet available."
        )
