# geneintelligence/src/geneintelligence/finetuners.py
import torch
import pickle
import pandas as pd
from torch.utils.data import DataLoader
from functools import partial
from anndata import AnnData
from tqdm import tqdm

from .models import ExpressionIntelligence, EiCellBinaryClassifier, EiCellMulticlassClassifier
from .configs import EiFinetuneConfig, load_config
from .datasets import AnndataCellFinetuneDataset, RankOfCountsRanksSorter, ei_cell_finetune_collate_fn, initialize_worker


class EiTestClassificationFinetuner:
    """
    A simple fine-tuner for cell classification tasks

    Args:
        pretrained_model_path (str): Path to the pre-trained backbone model checkpoint (.pt file).
        token_dictionary_path (str): Path to the token dictionary file in pickle format.
        gene_metadata_path (str): Path to the gene metadata file in hdf5 format.

    """

    def __init__(self,
                 pretrained_model_path: str,
                 token_dictionary_path: str,
                 gene_metadata_path: str,
                 pretrain_config_path: str,
                 model_step: int = 0):

        #make sure CUDA is available, raise an error if not
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA is not available. Please check your GPU configuration.")

        self.device = torch.device("cuda")

        self.pretrained_model_path = pretrained_model_path
        self.model = None

        with open(token_dictionary_path, "rb") as f:
            self.dictionary = pickle.load(f)

        gene_metadata = pd.read_hdf(gene_metadata_path, key="gene_metadata")
        gene_ranks = gene_metadata.cohort_rank
        gene_ranks.index = gene_ranks.index.astype(str)
        self.sorter = RankOfCountsRanksSorter(gene_ranks)
        self.gene_metadata = gene_metadata

        self.config = load_config(EiFinetuneConfig, pretrain_config_path)
        self.config.finetune_backbone_training_steps = model_step

    def train(self,
              adata: AnnData,
              label_column: str,
              finetune_classes: list[str],
              finetune_task: str = 'classification',
              epochs: int = 1,
              num_workers: int = 2):
        """
        Fine-tunes the model on the provided annotated data.

        Args:
            adata (AnnData): An AnnData object with expression data and labels.
            label_column (str): The column in `adata.obs` with cell labels.
            pretrain_config_path (str): Path to the pre-training YAML config file.
            epochs (int): The number of epochs to train for.
            num_workers (int, optional): Number of DataLoader workers. Defaults to 2.
        """
        print("Starting fine-tuning process...")

        self.config.finetune_task = finetune_task
        self.config.finetune_epochs = epochs

        self.config.finetune_classes = finetune_classes
        self.class_names = self.config.finetune_classes
        print(f"Classes found: {self.class_names}")

        # These metadata columns should match those used during pre-training
        metadata_cols = [
            'assay', 'suspension_type', 'tissue_general', 'sex',
            'development_stage', 'disease'
        ]

        dataset = AnndataCellFinetuneDataset(adata=adata,
                                             dictionary=self.dictionary,
                                             sorter=self.sorter,
                                             gene_metadata=self.gene_metadata,
                                             max_len=self.config.max_len,
                                             metadata_cols=metadata_cols,
                                             ground_truth_col='ground_truth')

        collate_fn = partial(ei_cell_finetune_collate_fn,
                             pad_token=self.dictionary.get('<pad>'),
                             ground_truth_dtype=torch.long)

        dataloader = DataLoader(dataset,
                                batch_size=self.config.finetune_batch_size,
                                shuffle=True,
                                collate_fn=collate_fn,
                                num_workers=num_workers,
                                worker_init_fn=initialize_worker)

        backbone = ExpressionIntelligence(self.config,
                                          device=self.device).to(self.device)
        checkpoint = torch.load(self.pretrained_model_path,
                                map_location=self.device)
        backbone.load_state_dict(checkpoint['model_state_dict'])

        num_classes = len(self.config.finetune_classes)
        if num_classes == 2:
            model_class = EiCellBinaryClassifier
            self.config.finetune_loss = 'BCEWithLogitsLoss'
        else:
            model_class = EiCellMulticlassClassifier
            self.config.finetune_loss = 'CrossEntropyLoss'

        self.model = model_class(pretrained_model=backbone,
                                 config=self.config,
                                 device=self.device,
                                 num_metadata_tokens=len(metadata_cols))
        self.model.to(self.device)

        optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad,
                                             self.model.parameters()),
                                      lr=self.config.finetune_max_lr)

        self.model.train()
        for epoch in range(self.config.finetune_epochs):
            pbar = tqdm(
                dataloader,
                desc=f'Epoch {epoch + 1}/{self.config.finetune_epochs}',
                total=len(dataloader),
                miniters=1,
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt}{postfix}\n')

            total_loss = 0
            for batch in pbar:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                optimizer.zero_grad()
                loss = self.model.compute_loss(batch)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})

            avg_loss = total_loss / len(dataloader)
            print(
                f"\nEpoch {epoch + 1} completed. Average Loss: {avg_loss:.4f}\n",
                flush=True)

        print("Fine-tuning finished.")
