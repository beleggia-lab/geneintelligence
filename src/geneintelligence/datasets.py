#!/usr/bin/env python3
"""
Script Name: datasets.py
Description: PyTorch Dataset classes and data loading utilities for single-cell 
             genomics, including LMDB memory mapping, AnnData parsing, and gene sorting.

Usage:
    Imported as a module. Not intended for direct execution.
"""

# Standard Library Imports
import abc
import hashlib
import logging
import math
import os
import random
import tempfile
from collections.abc import Iterator
from typing import Any

# Third-Party Imports
import lmdb
import lz4.frame
import numpy as np
import pandas as pd
import torch
from anndata import AnnData
from scipy import stats
from scipy.sparse import csr_matrix
from torch.utils.data import Dataset, Sampler

logger = logging.getLogger(__name__)


class GeneSorterBase(abc.ABC):
    """Abstract base for gene-ordering strategies applied to a single cell."""

    count_dtype: type = np.int32

    def __init__(self, vocab_size: int) -> None:
        self.vocab_size = vocab_size

    @abc.abstractmethod
    def sort_genes(self, gene_indices: np.ndarray,
                   counts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Takes raw matrix indices and counts for a single cell.
        Returns the sorted matrix indices and the sorted counts.
        """
        pass


def _to_tp10k(counts: np.ndarray) -> np.ndarray:
    """Convert raw integer counts to tp10k (transcripts per 10k) as float32."""

    counts_f = counts.astype(np.float32, copy=False)
    total = counts_f.sum()
    if total <= 0:
        return counts_f
    return counts_f * (10000.0 / float(total))


class Tp10kRandomSorter(GeneSorterBase):
    """Random shuffle; returns tp10k-normalized counts (float32)."""

    count_dtype = np.float32

    def sort_genes(self, gene_indices: np.ndarray,
                   counts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        perm = np.random.permutation(len(gene_indices))
        tp10k = _to_tp10k(counts)
        return gene_indices[perm], tp10k[perm]


class RankOfTp10kRandomSorter(GeneSorterBase):
    """Sorts by raw count descending, random tie-breaking; returns tp10k counts."""

    count_dtype = np.float32

    def sort_genes(self, gene_indices: np.ndarray,
                   counts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        # Shuffle first to guarantee random tie-breaking under stable sort.
        perm = np.random.permutation(len(gene_indices))
        shuffled_indices = gene_indices[perm]
        shuffled_counts = counts[perm]

        sort_idx = np.argsort(-shuffled_counts, kind='stable')
        sorted_indices = shuffled_indices[sort_idx]
        sorted_counts_raw = shuffled_counts[sort_idx]

        # tp10k is invariant to gene order, so compute on the sorted vector.
        tp10k = _to_tp10k(sorted_counts_raw)
        return sorted_indices, tp10k


class RandomSorter(GeneSorterBase):
    """Completely random shuffle of genes (ignores counts entirely)."""

    def sort_genes(self, gene_indices: np.ndarray,
                   counts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        perm = np.random.permutation(len(gene_indices))
        return gene_indices[perm], counts[perm]


class RankOfCountsRanksSorter(GeneSorterBase):
    """Sorts by count descending. Ties broken by global gene rank."""

    def __init__(self, gene_ranks: pd.Series, vocab_size: int) -> None:
        super().__init__(vocab_size)
        self.global_ranks = gene_ranks.values.astype(np.float32)

    def sort_genes(self, gene_indices: np.ndarray,
                   counts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        local_ranks = self.global_ranks[gene_indices]

        score = counts - (local_ranks * 1e-6)

        sort_idx = np.argsort(-score, kind='stable')
        return gene_indices[sort_idx], counts[sort_idx]


class RankOfCountsMedianExpressingSorter(GeneSorterBase):
    """Sorts by count descending. Ties broken by global median expression."""

    def __init__(self, gene_medians: pd.Series, vocab_size: int) -> None:
        super().__init__(vocab_size)
        self.global_medians = gene_medians.values.astype(np.float32)

    def sort_genes(self, gene_indices: np.ndarray,
                   counts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        local_medians = self.global_medians[gene_indices]

        score = counts - (local_medians * 1e-6)

        sort_idx = np.argsort(-score, kind='stable')
        return gene_indices[sort_idx], counts[sort_idx]


class RankOfCountsRandomSorter(GeneSorterBase):
    """Sorts by expression count descending. Ties are broken randomly."""

    def sort_genes(self, gene_indices: np.ndarray,
                   counts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        # Shuffle first to guarantee random tie-breaking
        perm = np.random.permutation(len(gene_indices))
        shuffled_indices = gene_indices[perm]
        shuffled_counts = counts[perm]

        # Use stable sort on the negative counts for descending order.
        # Stable sort ensures the random permutation of ties is preserved.
        sort_idx = np.argsort(-shuffled_counts, kind='stable')
        return shuffled_indices[sort_idx], shuffled_counts[sort_idx]


class RankOfRankshiftSorter(GeneSorterBase):
    """Sorts by the shift between the cell's local rank and the global cohort rank."""

    def __init__(self, gene_ranks: pd.Series, vocab_size: int) -> None:
        super().__init__(vocab_size)
        self.global_ranks = gene_ranks.values.astype(np.float32)

    def sort_genes(self, gene_indices: np.ndarray,
                   counts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        local_ranks = self.global_ranks[gene_indices]

        # 1. Calculate local average ranks for the counts
        cell_ranks = stats.rankdata(counts, method='average')

        # 2. Add offset (vocab_size - number of expressed genes)
        cell_ranks = cell_ranks + self.vocab_size - len(gene_indices)

        # 3. Calculate rank shift
        rankshift = cell_ranks - local_ranks

        # Sort descending by rankshift
        sort_idx = np.argsort(-rankshift, kind='stable')
        return gene_indices[sort_idx], counts[sort_idx]


class RankOfFoldchangeMedianExpressingSorter(GeneSorterBase):
    """Sorts by Fold Change vs Global Median Expression."""

    def __init__(self, gene_medians: pd.Series, vocab_size: int) -> None:
        super().__init__(vocab_size)
        self.global_medians = gene_medians.values.astype(np.float32)

    def sort_genes(self, gene_indices: np.ndarray,
                   counts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        local_medians = self.global_medians[gene_indices]

        total_counts = counts.sum()
        tp10k = (counts * 10000.0) / total_counts

        fold_change = np.round(tp10k / local_medians, decimals=6)

        sort_idx = np.argsort(-fold_change, kind='stable')
        return gene_indices[sort_idx], counts[sort_idx]


def get_sorter(strategy_name: str,
               gene_ranks: pd.Series | None = None,
               gene_medians: pd.Series | None = None,
               vocab_size: int = 0) -> GeneSorterBase:
    """
    Factory function to return the correct Sorter class based on the strategy string.
    """
    if strategy_name == 'rank_of_counts_random':
        return RankOfCountsRandomSorter(vocab_size)

    elif strategy_name == 'rank_of_counts_ranks':
        return RankOfCountsRanksSorter(gene_ranks, vocab_size)

    elif strategy_name == 'rank_of_counts_median_expressing':
        return RankOfCountsMedianExpressingSorter(gene_medians, vocab_size)

    elif strategy_name == 'rank_of_rankshift':
        return RankOfRankshiftSorter(gene_ranks, vocab_size)

    elif strategy_name == 'rank_of_foldchange_median_expressing':
        return RankOfFoldchangeMedianExpressingSorter(gene_medians, vocab_size)

    elif strategy_name == 'random':
        return RandomSorter(vocab_size)

    elif strategy_name == 'tp10k_random':
        return Tp10kRandomSorter(vocab_size)

    elif strategy_name == 'rank_of_tp10k_random':
        return RankOfTp10kRandomSorter(vocab_size)

    else:
        raise ValueError(f"Unknown sorting strategy: '{strategy_name}'")


class AnnDataDataset(Dataset):
    """In-memory dataset that tokenizes cells from an AnnData object on the fly.

    Each item is a cell encoded as ``<cell>`` + metadata tokens + gene tokens
    (ordered by the configured sorter), optionally paired with a parallel
    counts vector.
    """

    def __init__(self,
                 *,
                 adata: AnnData,
                 dictionary: dict[str, int],
                 gene_metadata: pd.DataFrame,
                 max_len: int,
                 sorter: "GeneSorterBase",
                 metadata_cols: list[str],
                 included_cells: list | None = None,
                 fill_metadata: bool = False,
                 return_counts: bool = False) -> None:

        super().__init__()

        if included_cells is not None:
            adata = adata[included_cells].copy()
        if not isinstance(adata.X, csr_matrix):
            adata.X = adata.X.tocsr()

        self.dictionary = dictionary.copy()
        self.dictionary['unknown'] = dictionary['<mask>']
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
                [self.dictionary[name] for name in gene_names], dtype=np.int32)
        except KeyError as e:
            raise ValueError(
                f"A gene in adata.var.index was not found in the dictionary: {e}"
            )

        self.cell_token = self.dictionary.get('<cell>')
        if fill_metadata:
            for col in metadata_cols:
                if col not in adata.obs.columns:
                    adata.obs[col] = '<mask>'
        assert set(metadata_cols).issubset(
            set(adata.obs.columns)
        ), f"Metadata columns {set(metadata_cols) - set(adata.obs.columns)} not found in adata.obs.columns"
        self.pre_tokenized_metadata = (adata.obs[metadata_cols].apply(
            lambda s: self.map_metadata(s, self.dictionary)).to_numpy(
                dtype=np.int32))

        self.X_data = adata.X.data
        self.X_indices = adata.X.indices
        self.X_indptr = adata.X.indptr
        self.return_counts = return_counts
        self.count_pad_length = 1 + len(metadata_cols)
        self._count_dtype = sorter.count_dtype
        self.prefix_counts = np.zeros(1 + len(metadata_cols),
                                      dtype=self._count_dtype)
        self.metadata_length = len(metadata_cols)

    def map_metadata(self, series: pd.Series,
                     dictionary: dict[str, int]) -> pd.Series:
        s = series.astype("string")

        missing = s.isna() | s.isin(["", "nan", "NaN", "None", "<NA>"])
        if missing.any():
            examples = series[missing].head(10).tolist()
            raise ValueError(
                f"Missing metadata values in obs column {series.name!r}; "
                f"examples: {examples}")

        unknown = ~s.isin(dictionary.keys())
        if unknown.any():
            examples = s[unknown].dropna().unique()[:10].tolist()
            raise KeyError(
                f"Metadata values in obs column {series.name!r} are not in the "
                f"Gene Intelligence token dictionary; examples: {examples}")

        return s.map(dictionary).astype(np.int32)

    def __len__(self) -> int:
        return len(self.pre_tokenized_metadata)

    def __getitem__(self,
                    idx: int) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        # 1. Get metadata tokens, mapped to final IDs
        metadata_tokens = self.pre_tokenized_metadata[idx]

        # 2. Get raw gene data for the cell
        start = self.X_indptr[idx]
        end = self.X_indptr[idx + 1]
        cell_gene_indices = self.X_indices[start:end]
        cell_counts = self.X_data[start:end]

        # 3. Get the sorted gene indices from the sorter
        sorted_gene_indices, sorted_counts = self.sorter.sort_genes(
            cell_gene_indices, cell_counts)

        # 4. Convert indices to final token IDs using the pre-computed map.
        final_gene_tokens = self.gene_index_to_token_id[sorted_gene_indices]

        # 5. Assemble sequence
        prefix_tokens = np.concatenate(([self.cell_token], metadata_tokens),
                                       dtype=np.int32)
        token_ids = np.concatenate((prefix_tokens, final_gene_tokens),
                                   dtype=np.int32)

        # 6. Truncate
        token_ids = token_ids[:self.max_len]

        if self.return_counts:
            final_counts = np.concatenate((self.prefix_counts, sorted_counts),
                                          dtype=self._count_dtype,
                                          casting='no')
            final_counts = final_counts[:self.max_len]
            return token_ids, final_counts

        return token_ids


def get_or_create_mmap_cache(df: pd.DataFrame, database_folder: str,
                             prefix: str, metadata_cols: list[str]) -> str:
    """
    Creates a memory-mapped .npy cache of the 2D metadata matrix.
    Returns the path to the .npy file.
    Race-safe across concurrent local or multi-node jobs.
    """
    os.makedirs(database_folder, exist_ok=True)

    hasher = hashlib.md5()
    hasher.update(str(df.shape).encode())
    hasher.update(str(metadata_cols).encode())

    sample = pd.concat([df.head(1000), df.tail(1000)])
    hasher.update(pd.util.hash_pandas_object(sample).values.tobytes())

    df_hash = hasher.hexdigest()
    npy_path = os.path.join(database_folder, f"{prefix}_{df_hash}.npy")

    if os.path.exists(npy_path):
        logger.info("Cache hit for %s. Using existing memory-mapped file: %s",
                    prefix, npy_path)
        return npy_path

    logger.info("Cache miss for %s. Building memory-mapped file: %s...",
                prefix, npy_path)

    array_data = df[metadata_cols].to_numpy(dtype=np.int32)

    fd, temp_atomic_path = tempfile.mkstemp(
        dir=database_folder,
        prefix=f"{prefix}_{df_hash}.",
        suffix=".npy.tmp",
    )

    try:
        with os.fdopen(fd, "wb") as f:
            np.save(f, array_data)

        os.replace(temp_atomic_path, npy_path)
        logger.info("Memory-mapped file built successfully.")

    except Exception:
        try:
            os.remove(temp_atomic_path)
        except FileNotFoundError:
            pass
        raise

    return npy_path


class LmdbDataset(Dataset):
    """On-disk dataset backed by an LMDB store of compressed cell records.

    Gene indices/counts are read from LMDB (with an optional faster "hot"
    store), metadata is served from a memory-mapped cache, and each item is
    assembled into ``<cell>`` + metadata + genes.
    """

    def __init__(self,
                 database_path: str,
                 cell_token_ids: pd.DataFrame,
                 dictionary: dict[str, int],
                 gene_metadata: pd.DataFrame,
                 max_len: int,
                 metadata_cols: list[str],
                 sorter: Any,
                 hot_database_path: list[str] | None = None,
                 included_cells: list[str] | pd.Index | np.ndarray
                 | None = None,
                 return_counts: bool = False,
                 indices_to_mask: list[str] | None = None) -> None:

        # 1. Store configurations
        self.database_path = database_path
        self.hot_database_path = hot_database_path
        self.max_len = max_len
        self.metadata_cols = metadata_cols
        self.sorter = sorter
        self.return_counts = return_counts
        self.indices_to_mask = indices_to_mask

        # Dictionary tokens
        self.cell_token = dictionary.get('<cell>', 2)
        self.mask_token_id = dictionary.get('<mask>', 0)

        # 2. Determine included cells
        if included_cells is not None:
            self.mmap_row_indices = cell_token_ids.index.get_indexer(
                included_cells).astype(np.int32)
            self.lmdb_keys = np.array(included_cells, dtype='S')
        else:
            self.mmap_row_indices = np.arange(len(cell_token_ids),
                                              dtype=np.int32)
            self.lmdb_keys = cell_token_ids.index.to_numpy(dtype='S')

        # 3. Gene string index to token ID mapping
        self.gene_index_to_token_id = np.array(
            [dictionary[g] for g in gene_metadata.index], dtype=np.int32)

        # 4. Memory mapping of cell tokens
        database_folder = os.path.dirname(database_path)
        self.mmap_path = get_or_create_mmap_cache(
            df=cell_token_ids,
            database_folder=database_folder,
            prefix="cell_metadata",
            metadata_cols=metadata_cols)

        # 5. Metadata sequence calculations
        self.metadata_length = len(metadata_cols)
        self._count_dtype = sorter.count_dtype
        if self.return_counts:
            self.prefix_counts = np.zeros(1 + self.metadata_length,
                                          dtype=self._count_dtype)

        # 6. Internal state variables for the worker
        self._mmap_array = None
        self.env = None

    def _initialize_database(self) -> None:
        """Called inside `initialize_worker` on the spawned process."""
        self.env = lmdb.open(self.database_path,
                             readonly=True,
                             lock=False,
                             readahead=False,
                             max_readers=512,
                             subdir=True)
        self.txn = self.env.begin(write=False)

        # Optional hot LMDB for cells staged on faster local storage.
        # Hot is checked first; misses fall through to the cold env above.
        if self.hot_database_path is not None:
            self.hot_env = lmdb.open(self.hot_database_path,
                                     readonly=True,
                                     lock=False,
                                     readahead=True,
                                     max_readers=512,
                                     subdir=True)
            self.hot_txn = self.hot_env.begin(write=False)
        else:
            self.hot_env = None
            self.hot_txn = None

    @property
    def mmap_array(self) -> np.memmap:
        """Lazy loader: Opens the OS pointer only when the worker asks for it."""
        if self._mmap_array is None:
            self._mmap_array = np.load(self.mmap_path, mmap_mode='r')
        return self._mmap_array

    def __len__(self) -> int:
        return len(self.lmdb_keys)

    def __getitem__(self,
                    idx: int) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        # 1. Grab the raw integer and encode it directly
        cell_idx_bytes = self.lmdb_keys[idx].tobytes()

        # 2. Directly access the pre-calculated row index (O(1) array lookup, zero overhead)
        row_idx = self.mmap_row_indices[idx]
        metadata_values = self.mmap_array[row_idx].copy()

        if self.indices_to_mask:
            for i in self.indices_to_mask:
                metadata_values[i] = self.mask_token_id

        # 3. Read compressed gene data from LMDB
        # Try the hot env first; fall back to cold
        if self.hot_txn is not None:
            cell_data = self.hot_txn.get(cell_idx_bytes)
            if cell_data is None:
                cell_data = self.txn.get(cell_idx_bytes)
        else:
            cell_data = self.txn.get(cell_idx_bytes)

        if cell_data is None:
            raise KeyError(
                f"Cell ID '{cell_idx_bytes.decode('utf-8')}' not found in LMDB."
            )

        # 4. Decompress and decode LMDB payload
        data = (np.frombuffer(lz4.frame.decompress(cell_data),
                              dtype=np.uint16).reshape(2, -1).astype(np.int32))
        cell_gene_indices = data[0]
        cell_counts = data[1]

        # 5. Sort genes according to the configured strategy
        sorted_gene_indices, sorted_counts = self.sorter.sort_genes(
            cell_gene_indices, cell_counts)

        # 6. Map gene indices to token IDs
        final_gene_tokens = self.gene_index_to_token_id[sorted_gene_indices]

        # 7. Assemble sequence: <cell> + metadata + genes using numpy concatenation
        prefix_tokens = np.concatenate(([self.cell_token], metadata_values),
                                       dtype=np.int32)
        token_ids = np.concatenate((prefix_tokens, final_gene_tokens),
                                   dtype=np.int32)

        # 8. Truncate to max sequence length
        token_ids = token_ids[:self.max_len]

        # 9. Optionally build parallel counts vector
        if self.return_counts:
            final_counts = np.concatenate((self.prefix_counts, sorted_counts),
                                          dtype=self._count_dtype,
                                          casting='no')
            final_counts = final_counts[:self.max_len]
            return token_ids, final_counts

        return token_ids


class GiBucketSampler(Sampler):
    """Length-bucketed batch sampler keeping total tokens per batch roughly constant."""

    def __init__(self,
                 dataset: Dataset,
                 batch_size: int,
                 shuffle: bool = True,
                 drop_last: bool = True,
                 nnz_filtered: np.ndarray | None = None) -> None:
        """
        Token-constant bucket sampler. 
        Calculates buckets based on sequence length and yields batches 
        where total token count is roughly constant.
        """
        self.dataset = dataset
        self.base_batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.epoch = 0

        # 1. Compute bucket_ids: ceil(log2(length)) - 7, clipped between 0 and 7.
        lengths = nnz_filtered.astype(np.float32) + dataset.metadata_length + 1
        bucket_ids = np.ceil(np.log2(lengths)) - 7
        bucket_ids = np.clip(bucket_ids, 0, 7).astype(np.int32)

        # 2. Store INDICES per bucket using fast numpy masking
        self.bucket_indices = {}
        unique_buckets = np.unique(bucket_ids)
        for bucket in unique_buckets:
            indices = np.nonzero(bucket_ids == bucket)[0].astype(np.int32)
            self.bucket_indices[bucket] = indices

        # 3. Store bucket info needed for FLOPs calculation
        self.bucket_counts = pd.Series(2**(bucket_ids +
                                           7)).value_counts().sort_index()

        self.bucket_batch_sizes = {
            bucket: batch_size * (2**(7 - bucket))
            for bucket in range(8)
        }

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self) -> Iterator[list[int]]:
        rng_np = np.random.RandomState(self.epoch)
        batches = []

        # Generate batches dynamically every epoch
        for bucket, indices in self.bucket_indices.items():

            # A. Shuffle indices within the bucket
            if self.shuffle:
                indices = rng_np.permutation(indices)

            bs = self.bucket_batch_sizes[bucket]
            num_full_batches = len(indices) // bs

            # B. Vectorized batch creation
            if num_full_batches > 0:
                # Slice off the amount needed for full batches
                full_indices = indices[:num_full_batches * bs]
                # Reshape into a 2D array and cast to a list of lists
                bucket_batches = full_indices.reshape(-1, bs).tolist()
                batches.extend(bucket_batches)

            # C. Handle leftovers
            leftover = indices[num_full_batches * bs:]
            if len(leftover) > 0 and not self.drop_last:
                batches.append(leftover.tolist())

        # D. Shuffle the order of batches across all buckets
        if self.shuffle:
            # Convert to a numpy object array temporarily to shuffle lists of lists
            batches_arr = np.empty(len(batches), dtype=object)
            batches_arr[:] = batches
            rng_np.shuffle(batches_arr)
            batches = batches_arr.tolist()

        return iter(batches)

    def __len__(self) -> int:
        count = 0
        for bucket, indices in self.bucket_indices.items():
            bs = self.bucket_batch_sizes[bucket]
            if self.drop_last:
                count += len(indices) // bs
            else:
                count += (len(indices) + bs - 1) // bs
        return count


def initialize_worker(worker_id: int) -> None:
    """Initializes a DataLoader worker by seeding and setting up the DB connection."""

    worker_info = torch.utils.data.get_worker_info()
    if worker_info is not None:
        dataset = worker_info.dataset
        epoch = getattr(dataset, 'epoch', 0)
        seed = worker_id + epoch * worker_info.num_workers
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        # Check if the dataset needs a database connection
        if isinstance(dataset, LmdbDataset):
            dataset._initialize_database()


class LmdbCellFinetuneDataset(LmdbDataset):
    """LMDB-backed dataset that also returns a per-cell ground-truth label for finetuning."""

    def __init__(self,
                 *,
                 cell_token_ids: pd.DataFrame,
                 ground_truth_col: str,
                 gene_metadata: pd.DataFrame,
                 sorter: GeneSorterBase,
                 mask_token_index: int | None = None,
                 **kwargs: Any) -> None:

        super().__init__(cell_token_ids=cell_token_ids,
                         gene_metadata=gene_metadata,
                         sorter=sorter,
                         **kwargs)
        self.mask_token_index = mask_token_index
        cell_ids = [k.decode('utf-8') for k in self.lmdb_keys]
        self.ground_truth = cell_token_ids.loc[cell_ids,
                                               ground_truth_col].values

    def __getitem__(
        self, idx: int
    ) -> tuple[np.ndarray, Any] | tuple[np.ndarray, np.ndarray, Any]:
        result = super().__getitem__(idx)
        if self.return_counts:
            token_ids, counts = result
        else:
            token_ids = result

        if self.mask_token_index is not None:
            token_ids[self.mask_token_index] = self.mask_token_id

        ground_truth = self.ground_truth[idx]

        if self.return_counts:
            return token_ids, counts, ground_truth
        return token_ids, ground_truth


class AnndataCellFinetuneDataset(AnnDataDataset):
    """AnnData-backed dataset that returns a per-cell ground-truth label for finetuning."""

    def __init__(self,
                 *,
                 adata: AnnData,
                 ground_truth_col: str,
                 head_type: str = 'linear',
                 uns_ground_truth_key: str = None,
                 **kwargs: Any) -> None:
        """
        Args:
            adata: The annotated data matrix.
            ground_truth_col: The column in adata.obs that contains the target.
                For head_type='linear', this column holds the label value directly.
                For head_type='gene_dot', this column holds integer indices into
                adata.uns['labels'].
            head_type: 'linear' for scalar labels, 'gene_dot' for vector labels
                indexed via adata.uns['labels']. Must match the model's
                finetune_head_type.
            **kwargs: Passed to parent AnnDataDataset (dictionary, sorter, etc.).
        """
        super().__init__(adata=adata, **kwargs)

        self.head_type = head_type
        self.resolve_pointer = uns_ground_truth_key is not None

        if self.resolve_pointer:
            if uns_ground_truth_key not in adata.uns:
                raise KeyError(
                    f"resolve_pointer=True requires adata.uns['{uns_ground_truth_key}'] to "
                    "contain the target matrix.")
            self.label_matrix = np.asarray(adata.uns[uns_ground_truth_key],
                                           dtype=np.float32)
            self.label_indices = self.obs[ground_truth_col].values.astype(
                np.int32)
        else:
            self.ground_truth = self.obs[ground_truth_col].values

    def __getitem__(
        self, idx: int
    ) -> tuple[np.ndarray, Any] | tuple[np.ndarray, np.ndarray, Any]:
        result = super().__getitem__(idx)
        if self.return_counts:
            token_ids, counts = result
        else:
            token_ids = result

        if self.resolve_pointer:
            ground_truth = self.label_matrix[self.label_indices[idx]]
        else:
            ground_truth = self.ground_truth[idx]

        if self.return_counts:
            return token_ids, counts, ground_truth
        return token_ids, ground_truth


class AnndataGeneFinetuneDataset(AnnDataDataset):
    """AnnData-backed dataset that returns a per-token (per-gene) target sequence for finetuning."""

    def __init__(self, *, adata: AnnData,
                 token_to_value_map: dict[int, int | float] | pd.Series,
                 fill_value: int | float, **kwargs: Any) -> None:
        """
        Args:
            adata (AnnData): The annotated data matrix, used to generate the
                             base token sequences for each cell.
            token_to_value_map (Dict or pd.Series): The mapping from token ID to a
                                   ground truth value. For classification, values
                                   should be integers. For regression, floats.
            fill_value (int or float): The value to use for tokens that are not present
                                     in the token_to_value_map. It should be np.nan for regression and binary classification and a special value for multiclass classification (e.g., -100).
            **kwargs: Additional keyword arguments passed to the parent AnnDataDataset,
                      such as 'dictionary', 'sorter', 'max_len', etc.
        """
        super().__init__(adata=adata, **kwargs)

        if 'vocab_size' not in kwargs:
            raise ValueError(
                "AnndataGeneFinetuneDataset requires 'vocab_size' to be passed in kwargs."
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
        self, idx: int
    ) -> tuple[np.ndarray, list] | tuple[np.ndarray, np.ndarray, list]:
        """
        Retrieves a tokenized cell sequence and its corresponding value sequence.

        Returns:
            Tuple containing the list of token IDs, the parallel list of values,
            and optionally a parallel counts list.
        """

        result = super().__getitem__(idx)
        if self.return_counts:
            token_ids_list, counts = result
        else:
            token_ids_list = result

        value_sequence = self.lookup_array[token_ids_list]

        if self.return_counts:
            return token_ids_list, counts, value_sequence.tolist()
        return token_ids_list, value_sequence.tolist()


def gi_collate(batch: list[Any],
               pad_token: int,
               mask_token: int | None = None,
               mask_prob: float = 0.0,
               mask_counts_prob: float = 0.0,
               ground_truth_dtype: torch.dtype | None = None,
               ground_truth_pad_value: float | None = None,
               return_counts: bool = False,
               vocab_size: int | None = None,
               head_type: str = 'linear') -> dict[str, torch.Tensor]:
    """
    Collate function for pretraining, cell finetuning, gene finetuning,
    and inference.
    """
    # 1. Unpack batch layers based on configuration
    ground_truth_raw = None
    counts_batch = None

    if ground_truth_dtype is not None and return_counts:
        token_seqs, counts_batch, ground_truth_raw = zip(*batch)
    elif ground_truth_dtype is not None:
        token_seqs, ground_truth_raw = zip(*batch)
    elif return_counts:
        token_seqs, counts_batch = zip(*batch)
    else:
        token_seqs = list(batch)

    ground_truth_scalars = None
    ground_truth_seqs = None
    if ground_truth_raw is not None:
        if isinstance(ground_truth_raw[0], (list, np.ndarray)):
            ground_truth_seqs = ground_truth_raw
        else:
            ground_truth_scalars = ground_truth_raw

    # 2. Pad token sequences to next power of 2 (Using int32)
    batch_size = len(token_seqs)
    max_len = max(len(seq) for seq in token_seqs) if batch_size > 0 else 0
    if max_len > 0:
        max_len = 2**math.ceil(math.log2(max_len))

    tokens_tensor = torch.full((batch_size, max_len),
                               pad_token,
                               dtype=torch.int32)
    valid_attention_mask = torch.zeros((batch_size, max_len), dtype=torch.bool)

    for i, seq in enumerate(token_seqs):
        seq_len = len(seq)
        tokens_tensor[i, :seq_len] = torch.from_numpy(np.asarray(seq)).to(
            torch.int32)
        valid_attention_mask[i, :seq_len] = True

    result = {"valid_attention_mask": valid_attention_mask}
    # 3. Masking (pretraining only)
    if mask_prob > 0 and mask_token is not None:
        original_tokens = tokens_tensor.clone()
        mask_positions = torch.zeros((batch_size, max_len), dtype=torch.bool)

        # Create a boolean mask of valid candidates (ignore padding and <cell> token at index 0)
        mask_candidates = valid_attention_mask.clone()
        mask_candidates[:, 0] = False

        # B. Get 1D indices of all valid candidates across the batch
        valid_flat_indices = mask_candidates.view(-1).nonzero(as_tuple=True)[0]

        # C. Define a strict constant M for masking
        padded_volume = batch_size * max_len
        fixed_M = int(padded_volume * 0.75 * mask_prob)
        fixed_M = min(fixed_M, valid_flat_indices.numel())

        # D. Shuffle and pick exactly fixed_M tokens globally
        perm = torch.randperm(valid_flat_indices.numel())
        chosen_1d = valid_flat_indices[perm[:fixed_M]]

        # E. Apply masks directly using a flat view
        tokens_tensor.view(-1)[chosen_1d] = mask_token
        mask_positions.view(-1)[chosen_1d] = True

        result["masked_tokens"] = tokens_tensor
        result["tokens"] = original_tokens
        result["mask_positions"] = mask_positions

    else:
        result["tokens"] = tokens_tensor

    # 4. Ground truth
    if ground_truth_scalars is not None:
        # Scalar per cell (binary / multiclass / scalar regression).
        result["ground_truth"] = torch.tensor(ground_truth_scalars,
                                              dtype=ground_truth_dtype)
    elif ground_truth_seqs is not None:
        if ground_truth_pad_value is None:
            # Fixed-length label vector per cell (e.g., cell-level multi-target).
            result["ground_truth"] = torch.from_numpy(
                np.stack(ground_truth_seqs, axis=0)).to(ground_truth_dtype)
        else:
            # Per-token sequence (gene-level finetuning).
            # Pad along the token dimension, matching the tokens_tensor shape.
            gt_tensor = torch.full(
                (batch_size, max_len),
                ground_truth_pad_value,
                dtype=ground_truth_dtype,
            )
            for i, val_seq in enumerate(ground_truth_seqs):
                gt_tensor[i, :len(val_seq)] = torch.tensor(
                    val_seq, dtype=ground_truth_dtype)
            result["ground_truth"] = gt_tensor

    # 5. Counts (optional, any mode)
    if counts_batch is not None:
        first_dtype = counts_batch[0].dtype
        if not all(c.dtype == first_dtype for c in counts_batch):
            raise TypeError(
                "Heterogeneous count dtypes within a single batch: "
                f"{sorted({str(c.dtype) for c in counts_batch})}. "
                "All cells in a batch must come from a sorter with the same "
                "count_dtype.")

        if np.issubdtype(first_dtype, np.floating):
            torch_count_dtype = torch.float32
        elif np.issubdtype(first_dtype, np.integer):
            torch_count_dtype = torch.int32
        else:
            raise TypeError(
                f"Unsupported count dtype from sorter: {first_dtype}. "
                "Expected an integer or floating-point dtype.")

        counts_tensor = torch.zeros((batch_size, max_len),
                                    dtype=torch_count_dtype)
        for i, counts in enumerate(counts_batch):
            counts_tensor[i, :len(counts)] = torch.from_numpy(counts)

        result["counts"] = counts_tensor.clone()

        if mask_counts_prob > 0:
            count_mask_candidates = valid_attention_mask.clone()
            count_mask_candidates[:, 0] = False

            valid_flat_count_indices = count_mask_candidates.view(-1).nonzero(
                as_tuple=True)[0]

            padded_volume = batch_size * max_len
            fixed_count_M = int(padded_volume * 0.75 * mask_counts_prob)
            fixed_count_M = min(fixed_count_M,
                                valid_flat_count_indices.numel())

            count_perm = torch.randperm(valid_flat_count_indices.numel())
            count_chosen_1d = valid_flat_count_indices[
                count_perm[:fixed_count_M]]

            count_mask_positions = torch.zeros((batch_size, max_len),
                                               dtype=torch.bool)
            count_mask_positions.view(-1)[count_chosen_1d] = True

            counts_tensor.masked_fill_(count_mask_positions, -100)
            result["count_mask_positions"] = count_mask_positions
        result["masked_counts"] = counts_tensor
    else:
        result["masked_counts"] = torch.empty(0, dtype=torch.int32)

    return result
