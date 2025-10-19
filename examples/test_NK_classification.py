# geneintelligence/examples/test_NK_classification.py

# This script is a simple example of how to fine-tune a pre-trained expression intelligence model on a new dataset forcell
# It downloads the required files from a GitHub release and then fine-tunes the model.

import anndata
import os
import urllib.request
from geneintelligence.finetuners import EiTestClassificationFinetuner

DOWNLOAD_DIR = "./example_data"

REPO_URL = "https://github.com/beleggia-lab/geneintelligence/releases/download/v0.1.1a1"

REQUIRED_FILES = {
    "adata": {
        "name": "NK_test_annotated.h5ad",
        "url": f"{REPO_URL}/NK_test_annotated.h5ad"
    },
    "model": {
        "name": "Y8aJhX_step_1162919.pt",
        "url": f"{REPO_URL}/Y8aJhX_step_1162919.pt"
    },
    "config": {
        "name": "Y8aJhX_config.yaml",
        "url": f"{REPO_URL}/Y8aJhX_config.yaml"
    },
    "dictionary": {
        "name": "token_dictionary.pickle",
        "url": f"{REPO_URL}/token_dictionary.pickle"
    },
    "gene_meta": {
        "name": "gene_metadata_NK_test.h5",
        "url": f"{REPO_URL}/gene_metadata_NK_test.h5"
    }
}


def download_files_if_needed():
    """Checks for all required files and downloads them if they are missing."""
    os.makedirs(DOWNLOAD_DIR, exist_ok=True)
    all_files_present = True

    for key, file_info in REQUIRED_FILES.items():
        local_path = os.path.join(DOWNLOAD_DIR, file_info["name"])
        if not os.path.exists(local_path):
            print(f"Downloading {file_info['name']}...")
            try:
                urllib.request.urlretrieve(file_info["url"], local_path)
                print("...Download complete.")
            except Exception as e:
                print(f"Error downloading {file_info['name']}: {e}")
                all_files_present = False
        else:
            print(f'{local_path} already exists')
    return all_files_present


def main():

    # --- 1. Get the Data and Model Files ---
    if not download_files_if_needed():
        print("Could not download all required files. Exiting.")
        return

    # Construct local paths
    adata_path = os.path.join(DOWNLOAD_DIR, REQUIRED_FILES["adata"]["name"])
    model_path = os.path.join(DOWNLOAD_DIR, REQUIRED_FILES["model"]["name"])
    config_path = os.path.join(DOWNLOAD_DIR, REQUIRED_FILES["config"]["name"])
    dictionary_path = os.path.join(DOWNLOAD_DIR,
                                   REQUIRED_FILES["dictionary"]["name"])
    gene_meta_path = os.path.join(DOWNLOAD_DIR,
                                  REQUIRED_FILES["gene_meta"]["name"])

    # --- 2. Run the Fine-Tuning ---
    adata = anndata.read_h5ad(adata_path)
    pretrain_steps = model_path.split('_step_')[1].split('.pt')[0]

    try:
        finetuner = EiTestClassificationFinetuner(
            pretrained_model_path=model_path,
            token_dictionary_path=dictionary_path,
            gene_metadata_path=gene_meta_path,
            pretrain_config_path=config_path,
            model_step=pretrain_steps)

        finetuner.train(adata=adata,
                        label_column='ground_truth',
                        epochs=1,
                        finetune_task='classification',
                        finetune_classes=['non-NK', 'NK'])

    except Exception as e:
        print(f"\n--- An error occurred during the fine-tuning process ---")
        print(e)
        return


if __name__ == '__main__':
    main()
