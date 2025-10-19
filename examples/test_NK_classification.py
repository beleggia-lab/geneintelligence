# geneintelligence/examples/test_NK_classification.py
import anndata
import os
import urllib.request
import zipfile
from geneintelligence.finetuners import EiTestClassificationFinetuner

DOWNLOAD_DIR = "./example_data"
REPO_URL = "https://github.com/beleggia-lab/geneintelligence/releases/download/v0.1.1a1"

ASSET_ARCHIVE = {
    "name": "nk_test_assets.zip",
    "url": f"{REPO_URL}/nk_test_assets.zip"
}
EXPECTED_FILES = [
    "NK_test_annotated.h5ad", "Y8aJhX_step_1162919.pt", "Y8aJhX_config.yaml",
    "token_dictionary.pickle", "gene_metadata_NK_test.h5"
]


def download_and_unzip_assets():
    """Downloads and extracts the asset zip file."""
    os.makedirs(DOWNLOAD_DIR, exist_ok=True)

    # Check if all files already exist
    if all(
            os.path.exists(os.path.join(DOWNLOAD_DIR, f))
            for f in EXPECTED_FILES):
        print("All asset files already exist.")
        return True

    # Download the zip file
    zip_path = os.path.join(DOWNLOAD_DIR, ASSET_ARCHIVE["name"])
    if not os.path.exists(zip_path):
        print(f"Downloading {ASSET_ARCHIVE['name']}...")
        try:
            urllib.request.urlretrieve(ASSET_ARCHIVE["url"], zip_path)
            print("...Download complete.")
        except Exception as e:
            print(f"Error downloading {ASSET_ARCHIVE['name']}: {e}")
            return False

    # Unzip the file
    print(f"Extracting files from {zip_path}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(DOWNLOAD_DIR)
    print("...Extraction complete.")
    return True


def main():
    # --- 1. Get the Data and Model Files ---
    if not download_and_unzip_assets():
        print("Could not prepare all required asset files. Exiting.")
        return

    # Construct local paths from the expected filenames
    adata_path = os.path.join(DOWNLOAD_DIR, "NK_test_annotated.h5ad")
    model_path = os.path.join(DOWNLOAD_DIR, "Y8aJhX_step_1162919.pt")
    config_path = os.path.join(DOWNLOAD_DIR, "Y8aJhX_config.yaml")
    dictionary_path = os.path.join(DOWNLOAD_DIR, "token_dictionary.pickle")
    gene_meta_path = os.path.join(DOWNLOAD_DIR, "gene_metadata_NK_test.h5")

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
