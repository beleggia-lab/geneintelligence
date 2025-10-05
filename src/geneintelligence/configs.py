# geneintelligence/configs.py

from dataclasses import dataclass, field, asdict, fields, is_dataclass
import yaml
import numpy as np
import torch.nn as nn
import hashlib
import string
from pathlib import Path
from typing import Optional

# ---------------- Unified EiConfig ---------------- #


@dataclass
class EiConfig:
    # Basic settings
    vocab_size: int = 32768
    max_len: int = 16384
    uid: Optional[str] = field(init=False, default=None)
    metadata_cols: list = None

    # Backbone (model) parameters
    backbone_d_model: int = 128
    backbone_blocks: int = 2
    backbone_heads: int = 2
    backbone_ff_dim: int = 512
    backbone_dropout: float = 0.1
    backbone_total_parameters: int = field(init=False, default=0)
    backbone_loss: str = 'CrossEntropyLoss'
    backbone_loss_kwargs: dict = field(
        default_factory=lambda: {'label_smoothing': 0.0})

    # Pretraining data parameters
    pretrain_batch_size: int = 32
    pretrain_mask_prob: float = 0.15
    pretrain_preprocessing_strategy: str = "counts_random"
    pretrain_total_samples: int = field(
        init=False, default=0)  # e.g., total number of training samples
    pretrain_persistent_workers: bool = True
    pretrain_prefetch_factor: int = 4

    # Pretraining optimizer/scheduler parameters
    pretrain_optimizer: str = 'AdamW'
    pretrain_max_lr: float = 0.0003
    pretrain_beta1: float = 0.9
    pretrain_beta2: float = 0.99
    pretrain_optimizer_epsilon: float = 1e-8
    pretrain_weight_decay: float = 0.0001
    pretrain_scheduler: str = 'oneCycleLR'
    pretrain_div_factor: float = 100
    pretrain_final_div_factor: float = 100
    pretrain_step_strategy: str = 'fixed'
    pretrain_total_steps: int = 1000000
    pretrain_warmup_strategy: str = 'fixed'
    pretrain_warmup_steps: int = 100000
    pretrain_cooldown_strategy: str = 'fixed'
    pretrain_cooldown_steps: int = 0
    pretrain_epochs: int = field(init=False, default=0)
    pretrain_dataset_size: int = field(init=False, default=0)
    pretrain_validation_value: str = None
    pretrain_validation_column: str = None
    pretrain_validation_size: int = 500
    pretrain_test_value: str = None
    pretrain_test_column: str = None
    pretrain_test_size: int = 31250

    def calculate_pretrain_steps(self, num_parameters: int,
                                 dataset_size: int) -> None:
        """Compute total_steps, warmup_steps, cooldown_steps, and epochs based on provided parameters."""
        self.pretrain_dataset_size = dataset_size
        if 'N' in self.pretrain_step_strategy:
            factor = float(
                self.pretrain_step_strategy.replace('05N',
                                                    '0.5').replace('N', ''))
            self.pretrain_total_steps = int(
                (num_parameters * factor) / self.pretrain_batch_size)

        self.pretrain_epochs = int(
            np.ceil(self.pretrain_total_steps /
                    np.ceil(dataset_size / self.pretrain_batch_size)))

        if 'N' in self.pretrain_warmup_strategy:
            factor = float(
                self.pretrain_warmup_strategy.replace('05N',
                                                      '0.5').replace('N', ''))
            self.pretrain_warmup_steps = int(
                (num_parameters * factor) / self.pretrain_batch_size)

        if 'N' in self.pretrain_cooldown_strategy:
            factor = float(
                self.pretrain_cooldown_strategy.replace('05N', '0.5').replace(
                    'N', ''))
            self.pretrain_cooldown_steps = int(
                (num_parameters * factor) / self.pretrain_batch_size)

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
        Generate UID and save the full EiConfig to a YAML file.
        The output folder is created with a label that includes the UID, 
        the total parameters and the total samples.
        
        Returns:
            str: The label used for saving.
        """
        self.uid = self.generate_uid()  # Update UID based on current state
        combined_dict = asdict(self)
        # Use unified attribute names; ensure these fields exist in your config:
        label = f'{self.uid}'
        full_output_dir = f"{output_dir}/{label}/"
        Path(full_output_dir).mkdir(parents=True, exist_ok=True)
        with open(f"{full_output_dir}/{self.uid}_config.yaml", 'w') as f:
            yaml.dump(combined_dict, f, sort_keys=False)
        return label


@dataclass
class EiFinetuneConfig(EiConfig):

    # Fine-tuning parameters
    finetune_max_lr: float = 0.0003
    finetune_weight_decay: float = 0.0001
    finetune_dropout: float = 0.1
    finetune_freeze_embeddings: bool = True
    finetune_freeze_blocks: int = 2
    finetune_keep_blocks: int = 2
    finetune_new_blocks: int = 0
    finetune_epochs: int = 10
    finetune_loss: str = None  #'CrossEntropyLoss'   or 'BCEWithLogitsLoss' or 'MSELoss'
    finetune_loss_kwargs: dict = field(default_factory=lambda: {})
    finetune_batch_size: int = 32
    finetune_total_samples: int = field(init=False, default=0)
    finetune_prefetch_factor: int = 4
    finetune_persistent_workers: bool = True
    finetune_backbone_training_steps: int = 0
    finetune_validation_value: str = None
    finetune_validation_column: str = None
    finetune_validation_size: int = 500
    finetune_classes: list = None
    finetune_task: str = None  # 'classification' or 'regression'
    finetune_num_metadata_tokens: int = 5


# ---------------- Standalone load/save functions ---------------- #


def load_config(cls, path: str) -> EiConfig:
    """Load a configuration from a YAML file and return an instance of cls."""
    with open(path, 'r') as f:
        data = yaml.safe_load(f)
    return _dict_to_dataclass(cls, data)


def _dict_to_dataclass(cls, data) -> EiConfig:
    if not is_dataclass(cls):
        return data  # for primitive types
    kwargs = {}
    for f in fields(cls):
        if not f.init:
            continue
        value = data.get(f.name)
        if value is not None:
            # Attempt recursive conversion for nested dataclasses (if any)
            kwargs[f.name] = _dict_to_dataclass(f.type, value)

    return cls(**kwargs)
