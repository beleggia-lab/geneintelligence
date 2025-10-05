# gene intelligence

**Status:** Research Preview / Work in Progress - please get in touch before using

**Note:** This repository contains the core implementation for the **gene intelligence**, a planned suite of foundation models for the analysis of biological data. This code showcases the inaugural tool, the **expression intelligence**, and is currently under active development. A full manuscript and a stable, user-friendly version of this package are in preparation.

---

## 1. Abstract

Effectively learning from the scale and complexity of modern biologic datasets remains a fundamental challenge in biology. The **gene intelligence** aims to address this by developing a comprehensive suite of AI-powered tools that treat different modalities of biological data as a "language." By adapting powerful architectures from natural language processing, the **gene intelligence** will learn fundamental biological representations directly from raw data.

This repository currently features the first tool in this suite: the **expression intelligence**, a deep learning framework for single-cell RNA sequencing (scRNA-seq). The **expression intelligence** converts the unordered set of expressed genes in a cell into a biologically meaningful sequence. This "gene sentence" is then used to pre-train a large Transformer model on a massive corpus of unlabeled single-cell data. The resulting pre-trained model serves as a **foundation model** that can be fine-tuned for a wide array of downstream tasks, including cell type classification, perturbation response prediction, and biomarker discovery.

---

## 2. The gene intelligenceo

The **gene intelligence** is envisioned as a collection of specialized tools, each designed to understand a different type of biological data, that will feed into a single multimodal network. The **expression intelligence** is the first of these. Future work will expand the suite to include models for other data types, creating a unified ecosystem for deep learning in the life sciences.

---

## 3. Showcase: core concepts of the expression intelligence

The architectural and pre-training approach of the **expression intelligence** is heavily inspired by the pioneering work on **Geneformer** (Theis et al., *Nature*, 2023). Our implementation builds upon this foundation to create the first module of the broader, multimodal **gene intelligence**.

The implementation of the **expression intelligence**, is built on three pillars:

### a. Gene sorting strategies

A cell's transcriptome is an unordered set of genes and their counts. To apply sequence-based models like Transformers, we must first impose a meaningful order. The `gene_intelligence/datasets.py` module implements several strategies to sort genes based on their expression counts and global statistics, effectively creating a "grammar" for the language of the cell.

### b. The transformer backbone

The core model, defined in `gene_intelligence/models.py`, is a custom-built Transformer encoder. It leverages modern, high-performance components:
* **Flash Attention:** Utilizes `torch.nn.functional.scaled_dot_product_attention` for memory-efficient and fast attention calculations.
* **SwiGLU Activation:** Employs the SwiGLU variant of the GELU activation function in the feed-forward network for improved performance.
* **Flexible Configuration:** All model hyperparameters are managed via a unified `EiConfig` dataclass in `gene_intelligence/configs.py`.

### c. Pre-training and fine-tuning

The framework follows the established pre-train/fine-tune paradigm:
1.  **Pre-training (Self-Supervised):** The model learns to predict randomly masked genes in a cell's gene sequence, forcing it to understand the context and relationships between genes.
2.  **Fine-tuning (Supervised):** The pre-trained backbone is adapted for specific downstream tasks, such as cell type identification and gene function prediction.

---

## 4. Repository structure

* `gene_intelligence/configs.py`: Contains `dataclass` configurations for managing all model and training hyperparameters for the **expression intelligence**.
* `gene_intelligence/datasets.py`: Includes data loaders, preprocessing logic, and the gene sorting algorithms for the **expression intelligence**. Supports both in-memory `AnnData` objects and memory-mapped `LMDB` databases for scalability.
* `gene_intelligence/models.py`: Defines the core `ExpressionIntelligence` Transformer model and its fine-tuning heads.

---

## 5. Future Direction

Our immediate roadmap includes:
* Submission of a pre-print manuscript for the **expression intelligence**.
* Release of pre-trained model weights on a large corpus of data from the **[CELLxGENE](https://cellxgene.cziscience.com/)** census, a major public atlas of single-cell data.
* Development of the next **gene intelligence** tools.
* Refinement of the API for easier use by the research community.

---

## 6. Contact

filippo.beleggia@uk-koeln.de