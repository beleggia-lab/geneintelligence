# Standard library imports
import math
from functools import partial
import random

# Third-party imports
import lmdb
import lz4.frame
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, BatchSampler, DataLoader, IterableDataset
import collections
import math
import abc
from anndata import AnnData
from scipy.sparse import csr_matrix
from scipy import stats
from typing import Dict, Optional, List, Any, Iterator, Callable, Tuple, Union
import multiprocessing as mp
import itertools


class GeneSorterBase(abc.ABC):
    """
    Abstract Base Class for gene sorting strategies.
    """

    @abc.abstractmethod
    def sort_genes(self, gene_indices: np.ndarray,
                   gene_counts: np.ndarray) -> np.ndarray:
        """
        Takes cell-specific gene indices and counts and returns a sorted list
        of the gene indices.
        """
        raise NotImplementedError


class RankOfCountsRanksSorter(GeneSorterBase):
    """Sorts genes by count, using pre-computed global gene ranks as a tie-breaker."""

    def __init__(self, gene_ranks: pd.Series):
        self.gene_ranks = gene_ranks.values

    def sort_genes(self, gene_indices: np.ndarray,
                   gene_counts: np.ndarray) -> np.ndarray:
        if len(gene_indices) == 0:
            return []
        # Subtract a small fraction of the rank to break ties
        ranked_counts = gene_counts - (self.gene_ranks[gene_indices] * 1e-7)
        counts_argsort = np.argsort(-ranked_counts)
        return gene_indices[counts_argsort]


class RankOfCountsMedianExpressingSorter(GeneSorterBase):
    """Sorts genes by count, using pre-computed gene medians as a tie-breaker."""

    def __init__(self, gene_medians: pd.Series):
        self.gene_medians = gene_medians.values

    def sort_genes(self, gene_indices: np.ndarray,
                   gene_counts: np.ndarray) -> np.ndarray:
        if len(gene_indices) == 0:
            return []

        # Subtract a small fraction of the median expression to break ties
        ranked_counts = gene_counts - (self.gene_medians[gene_indices] * 1e-7)
        counts_argsort = np.argsort(-ranked_counts)
        return gene_indices[counts_argsort]


class RankOfCountsRandomSorter(GeneSorterBase):
    """Sorts genes by count, using random noise as a tie-breaker."""

    def sort_genes(self, gene_indices: np.ndarray,
                   gene_counts: np.ndarray) -> np.ndarray:
        if len(gene_indices) == 0:
            return []
        # Add small random noise to break ties randomly
        noise = np.random.rand(len(gene_counts)) * 1e-7
        noisy_counts = gene_counts + noise
        counts_argsort = np.argsort(-noisy_counts)
        return gene_indices[counts_argsort]


class RankOfRankshiftSorter(GeneSorterBase):
    """Sorts genes based on the shift from their global rank."""

    def __init__(self, gene_ranks: pd.Series, vocab_size: int):
        self.gene_ranks = gene_ranks.values
        self.vocab_size = vocab_size

    def sort_genes(self, gene_indices: np.ndarray,
                   gene_counts: np.ndarray) -> np.ndarray:
        if len(gene_indices) == 0:
            return []
        # Calculate rank within the cell and shift by global rank
        cell_ranks = stats.rankdata(
            gene_counts,
            method='average') + self.vocab_size - len(gene_indices)
        rank_shift = cell_ranks - self.gene_ranks[gene_indices]
        rankshift_argsort = np.argsort(-rank_shift)
        return gene_indices[rankshift_argsort]


class RankOfFoldChangeMedianSorter(GeneSorterBase):
    """Sorts genes by fold change over the median of expressing cells."""

    def __init__(self, gene_medians: pd.Series):
        self.gene_medians = gene_medians.values

    def sort_genes(self, gene_indices: np.ndarray,
                   gene_counts: np.ndarray) -> np.ndarray:
        if len(gene_indices) == 0:
            return []
        # Calculate fold change relative to the median expression
        count_sum = np.sum(gene_counts)
        normalized_count = (gene_counts * 10000) / count_sum
        fold_change = normalized_count / self.gene_medians[gene_indices]
        fold_change_argsort = np.argsort(-fold_change)
        return gene_indices[fold_change_argsort]


class AnnDataDataset(Dataset):

    def __init__(self,
                 *,
                 adata: AnnData,
                 dictionary: Dict[str, int],
                 gene_metadata: pd.DataFrame,
                 max_len: int,
                 sorter: "GeneSorterBase",
                 metadata_cols: list,
                 included_cells: list = None,
                 **kwargs):
        super().__init__()

        if included_cells is not None:
            adata = adata[included_cells].copy()
        if not isinstance(adata.X, csr_matrix):
            adata.X = adata.X.tocsr()

        self.dictionary = dictionary
        self.max_len = max_len
        self.sorter = sorter
        self.gene_metadata = gene_metadata
        self.obs = adata.obs
        if not gene_metadata.index.equals(adata.var.index):
            raise ValueError(
                "gene_metadata.index and adata.var.index are not identical")
        try:
            gene_names = adata.var.index.to_list()
            self.gene_index_to_token_id = np.array(
                [self.dictionary[name] for name in gene_names], dtype=np.int16)
        except KeyError as e:
            raise ValueError(
                f"A gene in adata.var.index was not found in the dictionary: {e}"
            )

        self.cell_token = self.dictionary.get('<cell>')
        assert set(metadata_cols).issubset(
            set(adata.obs.columns)
        ), f"Metadata columns {set(metadata_cols) - set(adata.obs.columns)} not found in adata.obs.columns"
        self.pre_tokenized_metadata = adata.obs[metadata_cols].apply(
            lambda s: self.map_metadata(s, self.dictionary)).values

        self.X_data = adata.X.data
        self.X_indices = adata.X.indices
        self.X_indptr = adata.X.indptr

    def map_metadata(self, series: pd.Series,
                     dictionary: Dict[str, int]) -> List[int]:
        return series.map(lambda x: dictionary[x])

    def __len__(self) -> int:
        return len(self.pre_tokenized_metadata)

    def __getitem__(self, idx: int) -> List[int]:
        # 1. Get metadata tokens (already mapped to final IDs)
        metadata_tokens = list(self.pre_tokenized_metadata[idx])
        token_ids = [self.cell_token] + metadata_tokens

        # 2. Get raw gene data for the cell
        start = self.X_indptr[idx]
        end = self.X_indptr[idx + 1]
        cell_gene_indices = self.X_indices[start:end]
        cell_counts = self.X_data[start:end]

        # 3. Get the SORTED GENE INDICES from the sorter
        sorted_gene_indices = self.sorter.sort_genes(cell_gene_indices,
                                                     cell_counts)

        # 4. Convert indices to final token IDs using the pre-computed map.
        final_gene_tokens = self.gene_index_to_token_id[sorted_gene_indices]
        token_ids += final_gene_tokens.tolist()

        # 5. Concatenate and truncate
        return token_ids[:self.max_len]


class LmdbDataset(Dataset):

    def __init__(self,
                 *,
                 database_path: str,
                 cell_token_ids: pd.DataFrame,
                 dictionary: Dict[str, int],
                 max_len: int,
                 metadata_cols: list,
                 included_cells: list = None,
                 **kwargs):
        super().__init__()
        if included_cells is not None:
            cell_token_ids = cell_token_ids.loc[included_cells].copy()

        self.database_path = database_path
        self.dictionary = dictionary
        self.max_len = max_len

        self.cell_tokens = cell_token_ids.loc[:, metadata_cols].copy()
        self.nnz_filtered = cell_token_ids.nnz_filtered.copy()
        self.env = None
        self.txn = None
        self.cell_token = dictionary.get('<cell>')
        self.metadata_length = len(metadata_cols)

    def _initialize_database(self) -> None:

        self.env = lmdb.open(
            self.database_path,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
            max_readers=1,  # One reader per worker process
            map_async=True)

    def __len__(self) -> int:
        return len(self.cell_tokens)

    def __getitem__(self, idx: int) -> List[int]:

        if self.env is None:
            self._initialize_database()

        cell_metadata_tokens = self.cell_tokens.iloc[idx]

        cell_name = cell_metadata_tokens.name
        cell_idx = str(cell_name).encode('utf-8')
        with self.env.begin(write=False) as txn:
            cell_data = txn.get(cell_idx)

        if cell_data is None:
            print(
                f"\n\nWarning: Cell ID '{cell_name}' (index {idx}) not found in LMDB database.\n\n",
                flush=True)
            raise KeyError(
                f"Cell ID '{cell_name}' (index {idx}) not found in LMDB database."
            )
        data_bytes = lz4.frame.decompress(cell_data)
        token_ids = [self.cell_token] + list(
            cell_metadata_tokens.values) + list(
                np.frombuffer(data_bytes, dtype=np.int16))

        return token_ids[:self.max_len]

    def __del__(self):
        # Ensure that file handles are closed when the dataset object is deleted.
        if self.env is not None:
            self.env.close()


class EiBucketSampler(BatchSampler):

    def __init__(self,
                 dataset: LmdbDataset,
                 batch_size: int,
                 shuffle: bool = True):
        """
        Args:
            dataset (scRNAseqDataset): Dataset object.
            batch_size (int): Number of samples per batch.
            shuffle (bool): Whether to shuffle indices within buckets and the overall batches.
        """
        self.batch_size = batch_size
        self.shuffle = shuffle

        # Compute bucket_ids: ceil(log2(length)) - 7, clipped between 0 and 7.
        lengths = dataset.nnz_filtered.values.astype(
            np.float32) + dataset.metadata_length + 1
        bucket_ids = np.ceil(np.log2(lengths)) - 7
        bucket_ids = np.clip(bucket_ids, 0, 7).astype(int)
        #count the number of samples in each bucket
        self.bucket_counts = pd.Series(2**(bucket_ids +
                                           7)).value_counts().sort_index()

        batches = []
        leftover = np.empty(0, dtype=int)

        # Iterate through bucket IDs 0 to 7.
        for bucket in range(8):
            # Get indices that fall into the current bucket.
            bucket_indices = np.nonzero(bucket_ids == bucket)[0]
            if shuffle and bucket_indices.size > 0:
                bucket_indices = np.random.permutation(bucket_indices)

            # Combine leftover from previous bucket with the current bucket's indices.
            combined = np.concatenate([leftover, bucket_indices])

            # Compute number of full batches.
            num_full_batches = combined.shape[0] // batch_size
            for i in range(num_full_batches):
                batch = combined[i * batch_size:(i + 1) * batch_size]
                batches.append(batch)

            # The remainder becomes leftover for the next bucket.
            leftover = combined[num_full_batches * batch_size:]

        # If any samples remain after processing all buckets, add them as the final batch.
        if leftover.size > 0:
            batches.append(leftover)

        # Optionally shuffle the overall list of batches.
        if shuffle and len(batches) > 0:
            np.random.shuffle(batches)

        self.batches = batches

    def __iter__(self) -> Iterator[np.ndarray]:
        for batch in self.batches:
            yield batch

    def __len__(self) -> int:
        return len(self.batches)


def initialize_worker(worker_id: int) -> None:
    """Initializes a DataLoader worker by seeding and setting up the DB connection."""

    np.random.seed(worker_id)
    random.seed(worker_id)

    worker_info = torch.utils.data.get_worker_info()
    if worker_info is not None:
        dataset = worker_info.dataset
        # Check if the dataset needs a database connection
        if isinstance(dataset, (LmdbDataset, LmdbCellFinetuneDataset)):
            dataset._initialize_database()


class LmdbCellFinetuneDataset(LmdbDataset):

    def __init__(self,
                 *,
                 cell_token_ids: pd.DataFrame,
                 ground_truth_col: str,
                 mask_token_index: int = None,
                 **kwargs: Any):
        super().__init__(cell_token_ids=cell_token_ids, **kwargs)
        self.ground_truth = cell_token_ids.loc[self.cell_tokens.index,
                                               ground_truth_col].values
        self.mask_token_index = mask_token_index

    def __getitem__(self, idx: int) -> Tuple[List[int], int]:
        token_ids = super().__getitem__(idx)
        ground_truth = self.ground_truth[idx]
        if self.mask_token_index is not None:
            token_ids[self.mask_token_index] = self.dictionary.get('<mask>')
        return token_ids, ground_truth


class AnndataCellFinetuneDataset(AnnDataDataset):

    def __init__(self, *, adata: AnnData, ground_truth_col: str,
                 **kwargs: Any):
        super().__init__(adata=adata, **kwargs)
        self.ground_truth = self.obs[ground_truth_col].values

    """
    This class is used to finetune the model on a cell-level task.
    Args:
        adata (AnnData): The annotated data matrix, used to generate the
                         base token sequences for each cell.
        ground_truth_col (str): The column in the adata.obs that contains the ground truth labels.
        **kwargs: Additional keyword arguments passed to the parent AnnDataDataset,
                  such as 'dictionary', 'sorter', 'max_len', etc.
    """

    def __getitem__(self, idx: int) -> Tuple[List[int], int]:
        token_ids = super().__getitem__(idx)
        ground_truth = self.ground_truth[idx]
        return token_ids, ground_truth


class AnndataGeneFinetuneDataset(AnnDataDataset):

    def __init__(self, *, adata: AnnData,
                 token_to_value_map: Union[Dict[int, Union[int, float]],
                                           pd.Series],
                 fill_value: Union[int, float], **kwargs: Any):
        """
        Args:
            adata (AnnData): The annotated data matrix, used to generate the
                             base token sequences for each cell.
            token_to_value_map (Dict or pd.Series): The mapping from token ID to a
                                   ground truth value. For classification, values
                                   should be integers. For regression, floats.
            fill_value (int or float): The value to use for tokens that are not present
                                     in the token_to_value_map. For classification, this
                                     should be the ignore_index (e.g., -100). For regression,
                                     this should be np.nan.
            **kwargs: Additional keyword arguments passed to the parent AnnDataDataset,
                      such as 'dictionary', 'sorter', 'max_len', etc.
        """
        super().__init__(adata=adata, **kwargs)

        if 'vocab_size' not in kwargs:
            raise ValueError(
                "TokenMapDataset requires 'vocab_size' to be passed in kwargs."
            )
        vocab_size = kwargs['vocab_size']

        self.lookup_array = np.full(vocab_size,
                                    fill_value,
                                    dtype=type(fill_value))

        if isinstance(token_to_value_map, dict):
            token_ids = list(token_to_value_map.keys())
            values = list(token_to_value_map.values())
            self.lookup_array[token_ids] = values
        elif isinstance(token_to_value_map, pd.Series):
            self.lookup_array[
                token_to_value_map.index] = token_to_value_map.values
        else:
            raise TypeError(
                "token_to_value_map must be a dictionary or pandas Series.")

    def __getitem__(
            self, idx: int) -> Tuple[List[int], Union[List[int], List[float]]]:
        """
        Retrieves a tokenized cell sequence and its corresponding value sequence.

        Returns:
            Tuple containing the list of token IDs and the parallel list of values.
        """

        token_ids_list = super().__getitem__(idx)
        value_sequence = self.lookup_array[token_ids_list]

        return token_ids_list, value_sequence.tolist()


class LmdbCellInferenceDataset(IterableDataset):
    """
    An iterable dataset for sequential inference on an LMDB database.
    """

    def __init__(self,
                 *,
                 database_path: str,
                 cell_token_ids: pd.DataFrame,
                 dictionary: Dict[str, int],
                 metadata_cols: list,
                 batch_size: int,
                 max_len: int,
                 metadata_to_mask: list = None):
        super().__init__()
        self.database_path = database_path
        self.cell_token_ids = cell_token_ids
        self.batch_size = batch_size
        self.max_len = max_len
        self.metadata_to_mask = metadata_to_mask
        self.dictionary = dictionary
        self.metadata_cols = metadata_cols

        self.mask_token_id = self.dictionary.get('<mask>')
        self.indices_to_mask = []
        if self.metadata_to_mask:
            for col_to_mask in self.metadata_to_mask:
                try:
                    idx = self.metadata_cols.index(col_to_mask)
                    self.indices_to_mask.append(idx)
                except ValueError:
                    print(
                        f"Warning: Column '{col_to_mask}' not in metadata_cols, cannot mask.",
                        flush=True)

    def __iter__(self) -> Iterator[Tuple[List[List[int]], List[str]]]:
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            worker_id = 0
            num_workers = 1
            cell_ids_for_worker = self.cell_token_ids.index
        else:
            # Multi-process data loading
            worker_id = worker_info.id
            num_workers = worker_info.num_workers
            # Assign a unique slice of cell IDs to this worker
            cell_ids_for_worker = itertools.islice(self.cell_token_ids.index,
                                                   worker_id, None,
                                                   num_workers)

        buckets_sequences = collections.defaultdict(list)
        buckets_cell_ids = collections.defaultdict(list)
        cell_token = self.dictionary.get('<cell>')

        env = lmdb.open(self.database_path,
                        readonly=True,
                        lock=False,
                        readahead=True,
                        meminit=False)

        with env.begin(write=False) as txn:
            for cell_id in cell_ids_for_worker:
                key_bytes = str(cell_id).encode('utf-8')
                value_bytes = txn.get(key_bytes)

                if value_bytes is None:
                    print(
                        f"Warning: Cell ID '{cell_id}' not found in LMDB for worker {worker_id}.",
                        flush=True)
                    continue

                metadata_tokens = self.cell_token_ids.loc[
                    cell_id, self.metadata_cols].tolist()
                if self.indices_to_mask:
                    for idx in self.indices_to_mask:
                        metadata_tokens[idx] = self.mask_token_id

                decompressed_bytes = lz4.frame.decompress(value_bytes)
                gene_tokens = np.frombuffer(decompressed_bytes,
                                            dtype=np.int16).tolist()
                full_sequence = ([cell_token] + metadata_tokens +
                                 gene_tokens)[:self.max_len]
                seq_len = len(full_sequence)
                if seq_len == 0: continue

                bucket_id = math.ceil(math.log2(seq_len))
                buckets_sequences[bucket_id].append(full_sequence)
                buckets_cell_ids[bucket_id].append(cell_id)

                if len(buckets_sequences[bucket_id]) == self.batch_size:
                    yield (buckets_sequences.pop(bucket_id),
                           buckets_cell_ids.pop(bucket_id))

        # Process leftovers
        for bucket_id in sorted(buckets_sequences.keys()):
            if buckets_sequences[bucket_id]:
                yield (buckets_sequences.pop(bucket_id),
                       buckets_cell_ids.pop(bucket_id))

        env.close()

    def __len__(self):
        return len(self.cell_token_ids)


def ei_collate_batch(batch: List[List[int]], pad_token: int, mask_token: int,
                     mask_prob: float) -> Dict[str, torch.Tensor]:
    """
    Collate function for training. It handles token sequences and labels.
    """
    batch_size = len(batch)
    max_len = max(len(seq) for seq in batch)
    max_len = 2**math.ceil(math.log2(max_len))  # Pads to the next power of 2

    # Initialize tensors
    tokens_tensor = torch.full((batch_size, max_len),
                               pad_token,
                               dtype=torch.long)
    original_tokens_tensor = torch.full((batch_size, max_len),
                                        pad_token,
                                        dtype=torch.long)
    inverted_padding_tensor = torch.zeros((batch_size, max_len),
                                          dtype=torch.bool)
    mask_positions_tensor = torch.zeros((batch_size, max_len),
                                        dtype=torch.bool)

    for i, seq in enumerate(batch):
        seq_len = len(seq)

        # Assign original and masked tokens
        original_tokens_tensor[i, :seq_len] = torch.tensor(seq)
        tokens_tensor[i, :seq_len] = torch.tensor(seq)

        # Set inverted padding mask (True for actual tokens, False for padding)
        inverted_padding_tensor[i, :seq_len] = True

        # Determine mask candidates
        masked_positions = []
        if mask_prob > 0:
            mask_candidates = np.arange(
                1, seq_len)  # Avoid masking the first token
            num_to_mask = max(1, int(len(mask_candidates) * mask_prob))
            masked_positions = np.random.choice(mask_candidates,
                                                size=num_to_mask,
                                                replace=False)

        # Apply masks
        tokens_tensor[i, masked_positions] = mask_token
        mask_positions_tensor[i, masked_positions] = True

    return {
        'masked_tokens': tokens_tensor,
        'inverted_padding': inverted_padding_tensor,
        'original_tokens': original_tokens_tensor,
        'mask_positions': mask_positions_tensor
    }


def ei_cell_finetune_collate_fn(
        batch: List[Tuple[List[int], int]], pad_token: int,
        ground_truth_dtype: torch.dtype) -> Dict[str, torch.Tensor]:
    """
    Collate function for cell-level finetuning. It handles token sequences and labels.
    """
    token_seqs, ground_truth = zip(*batch)
    batch_size = len(token_seqs)
    max_len = max(len(seq) for seq in token_seqs)
    max_len = 2**math.ceil(math.log2(max_len))

    tokens_tensor = torch.full((batch_size, max_len),
                               pad_token,
                               dtype=torch.long)
    inverted_padding_tensor = torch.zeros((batch_size, max_len),
                                          dtype=torch.bool)

    for i, seq in enumerate(token_seqs):
        seq_len = len(seq)
        tokens_tensor[i, :seq_len] = torch.tensor(seq)
        inverted_padding_tensor[i, :seq_len] = True

    return {
        'tokens': tokens_tensor,
        'inverted_padding': inverted_padding_tensor,
        'ground_truth': torch.tensor(ground_truth, dtype=ground_truth_dtype)
    }


def ei_cell_inference_collate_fn_with_ids(batch: Tuple[List[List[int]],
                                                       List[str]],
                                          pad_token: int) -> Dict[str, any]:
    """
    Collate function for inference that also handles cell IDs.
    """

    token_seqs, cell_ids = batch

    batch_size = len(token_seqs)
    # Pad to the next power of 2 for efficiency
    max_len = max(len(seq) for seq in token_seqs) if batch_size > 0 else 0
    if max_len > 0:
        max_len = 2**math.ceil(math.log2(max_len))

    tokens_tensor = torch.full((batch_size, max_len),
                               pad_token,
                               dtype=torch.long)
    inverted_padding_tensor = torch.zeros((batch_size, max_len),
                                          dtype=torch.bool)

    for i, seq in enumerate(token_seqs):
        seq_len = len(seq)
        tokens_tensor[i, :seq_len] = torch.tensor(seq, dtype=torch.long)
        inverted_padding_tensor[i, :seq_len] = True

    return {
        'tokens': tokens_tensor,
        'inverted_padding': inverted_padding_tensor,
        'cell_ids': cell_ids
    }
