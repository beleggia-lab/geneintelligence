#!/usr/bin/env python3
"""
Script Name: configs.py
Description: Configuration dataclasses for the Gene Intelligence model, 
             handling pretraining and finetuning hyperparameters.

Usage:
    Imported as a module. Not intended for direct execution.
"""

# Standard Library Imports
import hashlib
import string
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Self

# Third-Party Imports
import numpy as np
import yaml


@dataclass
class GiConfig:
    """Base configuration for gene intelligence model and pretraining hyperparameters."""

    # Basic settings
    vocab_size: int = 32768
    max_len: int = 16384
    uid: str | None = field(init=False, default=None)
    metadata_cols: list[str] = field(default_factory=list)
    scale_positional_encodings: bool = True
    use_counts: bool = False
    task: str | None = None

    # Backbone (model) parameters
    backbone_d_model: int = 256
    backbone_blocks: int = 4
    backbone_heads: int = 4
    backbone_ff_dim: int = 1408
    backbone_dropout: float = 0.1
    backbone_total_parameters: int = field(init=False, default=0)
    backbone_flops_per_batch: float = field(init=False, default=0.0)
    backbone_loss: str = 'CrossEntropyLoss'
    backbone_loss_kwargs: dict[str, Any] = field(default_factory=dict)
    backbone_use_positional_embeddings: bool = False

    pretrain_dynamic_loss_weighting: bool = False
    pretrain_count_uncertainty_floor: float | None = None
    pretrain_token_uncertainty_floor: float | None = None

    # Pretraining data parameters
    pretrain_load_batch_size: int = 32
    pretrain_accumulated_batch_size: int = 256
    pretrain_mask_prob: float = 0.25
    pretrain_mask_counts_prob: float = 0.0
    pretrain_preprocessing_strategy: str = "rank_of_counts_ranks"
    pretrain_total_samples: int = field(
        init=False, default=0)  # e.g., total number of training samples

    # Pretraining optimizer/scheduler parameters
    pretrain_optimizer: str = 'AdamW'
    pretrain_max_lr: float = 0.0003
    pretrain_beta1: float = 0.9
    pretrain_beta2: float = 0.99
    pretrain_optimizer_epsilon: float = 1e-8
    pretrain_weight_decay: float = 0.001
    pretrain_scheduler: str = 'oneCycleLR'
    pretrain_div_factor: float = 100
    pretrain_final_div_factor: float = 100
    pretrain_step_strategy: str = 'fixed'
    pretrain_total_steps: int = 100000
    pretrain_warmup_strategy: str = 'fixed'
    pretrain_warmup_steps: int = 3000
    pretrain_cooldown_strategy: str = 'fixed'
    pretrain_cooldown_steps: int = 0
    pretrain_epochs: int = field(init=False, default=0)
    pretrain_dataset_size: int = field(init=False, default=0)
    pretrain_validation_value: str | None = None
    pretrain_validation_column: str | None = None
    pretrain_validation_size: int = 65536
    pretrain_test_value: str | None = None
    pretrain_test_column: str | None = None
    pretrain_test_size: int = 65536

    def calculate_pretrain_steps(self,
                                 num_parameters: int,
                                 dataset_size: int,
                                 batches_per_epoch: int | None = None) -> None:
        """Compute total_steps, warmup_steps, cooldown_steps, and epochs based on provided parameters."""
        self.pretrain_dataset_size = dataset_size
        if 'N' in self.pretrain_step_strategy:
            factor = float(self.pretrain_step_strategy.replace('N', ''))
            self.pretrain_total_steps = int(
                (num_parameters * factor) /
                (self.pretrain_accumulated_batch_size * self.max_len))
        assert self.pretrain_accumulated_batch_size % self.pretrain_load_batch_size == 0, "accumulated batch size must be a multiple of load batch size"
        grad_accum_steps = self.pretrain_accumulated_batch_size // self.pretrain_load_batch_size
        if batches_per_epoch is not None:
            steps_per_epoch = batches_per_epoch // grad_accum_steps
        else:
            steps_per_epoch = int(
                np.ceil(dataset_size / self.pretrain_accumulated_batch_size))

        self.pretrain_epochs = int(
            np.ceil(self.pretrain_total_steps / steps_per_epoch))

        if 'N' in self.pretrain_warmup_strategy:
            factor = float(self.pretrain_warmup_strategy.replace('N', ''))
            self.pretrain_warmup_steps = int(
                (num_parameters * factor) /
                (self.pretrain_accumulated_batch_size * self.max_len))

        if 'N' in self.pretrain_cooldown_strategy:
            factor = float(self.pretrain_cooldown_strategy.replace('N', ''))
            self.pretrain_cooldown_steps = int(
                (num_parameters * factor) /
                (self.pretrain_accumulated_batch_size * self.max_len))

    def generate_uid(self, length: int = 6) -> str:
        """
        Generate a UID based on the current configuration.
        Excludes the `uid` field itself from the hash.
        """
        # Use asdict() and then filter out None values and the 'uid' key.
        config_dict = {
            k: v
            for k, v in asdict(self).items() if v is not None and k != 'uid'
        }
        # Create a sorted representation
        combined_repr = repr(sorted(config_dict.items()))
        hash_bytes = hashlib.sha1(combined_repr.encode('utf-8')).digest()
        hash_int = int.from_bytes(hash_bytes, 'big')

        chars = string.digits + string.ascii_letters
        uid_chars = []
        for _ in range(length):
            hash_int, remainder = divmod(hash_int, len(chars))
            uid_chars.append(chars[remainder])
        self.uid = ''.join(uid_chars)
        return self.uid

    def save(self, output_dir: str) -> str:
        """
        Generate UID and save the full GiConfig to a YAML file.
        
        Returns:
            str: The label used for saving.
        """
        self.uid = self.generate_uid()  # Update UID based on current state
        combined_dict = asdict(self)

        label = f'{self.uid}'
        full_output_dir = f"{output_dir}/{label}/"
        Path(full_output_dir).mkdir(parents=True, exist_ok=True)
        with open(f"{full_output_dir}/{label}_config.yaml", 'w') as f:
            yaml.dump(combined_dict, f, sort_keys=False)
        return label

    @classmethod
    def load(cls, path: str | Path) -> Self:
        """Load a configuration from a YAML file."""
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        return cls._from_dict(data)

    @classmethod
    def _from_dict(cls, data: dict[str, Any]) -> Self:
        """Recursively convert a dictionary into a dataclass instance."""
        if not is_dataclass(cls):
            return data
        kwargs = {}
        for f in fields(cls):
            if not f.init:
                continue
            value = data.get(f.name)
            if value is not None:
                # If the field is a dataclass itself, recursively instantiate it
                field_type = f.type
                if is_dataclass(field_type):
                    kwargs[f.name] = field_type._from_dict(value)
                else:
                    kwargs[f.name] = value

        return cls(**kwargs)


@dataclass
class GiFinetuneConfig(GiConfig):
    """Configuration for fine-tuning the pre-trained gene intelligence backbone model."""

    # Fine-tuning parameters
    finetune_backbone_uid: str | None = None
    finetune_model: str | None = None
    finetune_backbone_training_steps: int | None = None
    finetune_max_lr: float = 0.0003
    finetune_weight_decay: float = 0.0001
    finetune_beta1: float = 0.9
    finetune_beta2: float = 0.999
    finetune_dropout: float = 0.1
    finetune_freeze_embeddings: bool = False
    finetune_freeze_blocks: int = 0
    finetune_keep_blocks: int = 0
    finetune_new_blocks: int = 0
    finetune_epochs: int = 5
    finetune_loss: str | None = None  # 'CrossEntropyLoss'   or 'BCEWithLogitsLoss' or 'MSELoss'
    finetune_loss_kwargs: dict[str, Any] = field(default_factory=dict)
    finetune_load_batch_size: int = 32
    finetune_accumulated_batch_size: int = 32
    finetune_total_samples: int = field(init=False, default=0)
    finetune_validation_value: str | None = None
    finetune_validation_column: str | None = None
    finetune_validation_size: int = 500
    finetune_classes: list[str] = field(default_factory=list)
    finetune_task: str | None = None  # 'classification' or 'regression'
    finetune_mask_metadata: bool = False
    finetune_head_type: str | None = 'linear'  # 'linear', 'MLP', or 'gene_dot'
    finetune_n_regression_targets: int | None = None
    finetune_attention_genes: bool = False
    finetune_attention_genes_queries: int = 8
