# Gene Intelligence

![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Status: Research Preview](https://img.shields.io/badge/Status-Research_Preview-orange.svg)

**Status:** Research Preview

**Note:** This repository contains the implementation of Gene Intelligence, a transformer foundation model for single-cell RNA sequencing (scRNA-seq) data. The code is under active development, and a stable, user-friendly release is in preparation.

---

## 1. Overview

Gene Intelligence is a transformer foundation model for single-cell RNA sequencing (scRNA-seq). Each cell is encoded as a token sequence — a `<cell>` token, a set of metadata tokens, and one token per expressed gene — and the model is pre-trained on a large corpus of unlabelled single-cell data with a masked-token objective. The pre-trained backbone can then be fine-tuned for downstream tasks, including cell-type classification, gene classification, and doublet detection, or used directly to extract cell-, gene-, and sample-level embeddings.

In its count-embedding configuration, the model encodes expression by projecting log1p-transformed raw counts directly onto each gene-token embedding, using no library-size normalisation or positional encoding.

---

## 2. Core concepts

The implementation rests on three components.

### a. Gene-token preprocessing

A cell's transcriptome is an unordered set of (gene, count) pairs. The `src/geneintelligence/datasets.py` module turns this set into a token sequence and provides several ordering strategies, including random ordering and a range of count- and rank-based orderings (sorting by raw count, by transcripts-per-10k fold change, or by rank shift, with various tie-breaking rules). The count-embedding configuration used for Gene Intelligence orders genes randomly and encodes expression by projecting log1p-transformed raw counts onto each gene-token embedding, with no library-size normalisation.

### b. Transformer backbone

The core model, defined in `src/geneintelligence/models.py`, is a pre-norm Transformer encoder with:
* **Packed self-attention** uses FlashAttention on CUDA. On CPU, PyTorch scaled dot-product attention processes each cell separately, in tiles of up to 256 queries. Every query attends to the full cell.
* **SwiGLU** feed-forward layers.
* **Weight-tied** input and output embeddings.
* **Configurable** hyperparameters, managed through the `GiConfig` dataclass in `src/geneintelligence/configs.py`.

### c. Pre-training and fine-tuning

The framework follows the standard pre-train / fine-tune paradigm:
1.  **Pre-training (self-supervised):** the model predicts randomly masked gene tokens. In the count-embedding configuration, an independent subset of counts is also masked and predicted by a parallel regression head, with the two objectives combined using learned multi-task uncertainty weighting.
2.  **Fine-tuning (supervised):** the pre-trained backbone is adapted for specific downstream tasks through cell-level and gene-level classification and regression heads.

---

## 3. Repository structure

The package source lives under `src/geneintelligence/`:

* `configs.py`: dataclass configurations (`GiConfig`, `GiFinetuneConfig`) for managing model and training hyperparameters.
* `datasets.py`: data loaders, preprocessing logic, and the gene-ordering strategies. Supports both in-memory `AnnData` objects and memory-mapped `LMDB` databases for scalability.
* `models.py`: the core `GeneIntelligence` Transformer backbone, including the embeddings, attention blocks, and the masked-gene and count prediction heads.
* `finetuners.py`: fine-tuning heads built on top of the backbone for cell-level and gene-level classification and regression tasks.
* `api.py`: a high-level `GeneIntelligenceModel` interface for inference and for extracting cell-, gene-, and sample-level embeddings.
* `embeddings.py`: utilities for pooling and concatenating embeddings into cell- and sample-level `AnnData` objects.

### Pretrained model and metadata

Each model has its own folder under `models/`. The eight-layer model in `models/1OJZl1/` pairs the pretrained weights from step 406816 with their YAML configuration.

The shared `metadata/` folder contains the token dictionary (`token_dictionary.pickle`), the ordered gene panel (`gene_metadata_V1.h5`), and cell metadata for the source corpus (`cell_metadata_V1.h5`). When embedding new cells, the model reads their metadata from `cells.obs`.

---

## 4. Installation

### a. Prerequisites

* **Python 3.11+**
* **PyTorch 2.7.1 or later**
* The training and fine-tuning scripts require a CUDA-enabled GPU. Cell embedding supports CPU and CUDA.

### b. Installation steps

1.  **Clone the repository and fetch the model files**
    Install the Git LFS command-line client. From the root of your clone, run the following to fetch the model weights and corpus cell metadata.
    ```bash
    git lfs install
    git lfs pull
    ```

2.  **(Recommended) Set up a virtual environment**
    Using a virtual environment to manage dependencies is recommended. You can use your preferred tool, such as `conda` or `venv`.

    * **Using `conda`:**
        ```bash
        # Create a new environment
        conda create -n geneintelligence python=3.11 -y
        # Activate the environment
        conda activate geneintelligence
        ```

3.  **Install PyTorch**
    Install the CPU or CUDA build of PyTorch for your system. See the [official PyTorch website](https://pytorch.org/get-started/locally/) for current instructions.

4.  **For CUDA, install FlashAttention**
    CPU users can continue to step 5. The CUDA kernel builds against the installed PyTorch with build isolation disabled.
    ```bash
    pip install packaging ninja
    pip install flash-attn --no-build-isolation
    ```
    > **Note:** Building `flash-attn` from source can take several minutes and requires a matching CUDA toolchain. Installing `ninja` first speeds up the build.

5.  **Install Gene Intelligence**
    With the virtual environment activated, install the package and its remaining dependencies.
    ```bash
    pip install .
    ```

### c. Embed cells on CPU or GPU

Run this example from the repository root to load the eight-layer model and its shared metadata. The example expects a prepared `cells.h5ad` with raw counts in `.X` as an `int32` sparse CSR matrix. Its gene IDs must match the model’s gene panel in the same order. Reading the gene metadata with pandas requires PyTables (`pip install tables`).

```python
import anndata as ad
import pandas as pd

from geneintelligence import GeneIntelligenceModel

cells = ad.read_h5ad("cells.h5ad")
gene_metadata = pd.read_hdf("metadata/gene_metadata_V1.h5", key="gene_metadata")
assert cells.var_names.equals(gene_metadata.index)

model = GeneIntelligenceModel(
    checkpoint_path="models/1OJZl1/1OJZl1_406816.pt",
    config_path="models/1OJZl1/1OJZl1_config.yaml",
    token_dictionary=pd.read_pickle("metadata/token_dictionary.pickle"),
    gene_metadata=gene_metadata,
    device="cpu",  # Use "cuda" to run on a GPU.
)

embeddings = model.embed_cells(cells, batch_size=2)
embeddings.write_h5ad("cell_embeddings.h5ad")
```

The output contains one embedding per cell in `embeddings.X` and retains the input cell metadata in `embeddings.obs`. To select the device automatically, omit `device`. Gene Intelligence uses CUDA when available and CPU otherwise.

---

## 5. Roadmap

Planned work includes:
* Release additional pretrained models.
* A packaged doublet-detection workflow (the `detect_doublets` API is currently under development).
* Continued refinement of the high-level API.

---

## 6. Contact

filippo.beleggia@uk-koeln.de