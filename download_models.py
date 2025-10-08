# download_models.py

import os
import requests
import time
from tqdm import tqdm
import json

# --- Configuration ---
MODELS_DIR = "./models"
HF_HUB_URL = "https://huggingface.co"

MODELS_TO_DOWNLOAD = {
    "siglip": {
        "repo_id": "google/siglip-base-patch16-512",
        "required_files": ["config.json", "model.safetensors", "preprocessor_config.json",
         "tokenizer_config.json", "spiece.model",
          "tokenizer.json", "special_tokens_map.json",]
    },
    "dots_ocr": {
        "repo_id": "rednote-hilab/dots.ocr",
        "required_files": [
            "config.json", "model.safetensors", "generation_config.json", "chat_template.json", 
            "configuration_dots.py", "modeling_dots_vision.py", "modeling_dots_ocr.py",
            "preprocessor_config.json", "tokenizer.json", "tokenizer_config.json",  "special_tokens_map.json", "vocab.json"

        ]
    }
}

# --- NEW: Robust Downloader with Retries ---
def download_file_with_retries(url, destination, retries=3, timeout=90, backoff_factor=2):
    """
    Downloads a file with a progress bar, handling large files, timeouts, and retries.
    """
    for attempt in range(retries):
        try:
            print(f"Attempt {attempt + 1}/{retries}...")
            with requests.get(url, stream=True, timeout=timeout) as r:
                r.raise_for_status()
                total_size = int(r.headers.get('content-length', 0))
                with open(destination, 'wb') as f, tqdm(
                    desc=os.path.basename(destination),
                    total=total_size,
                    unit='iB',
                    unit_scale=True,
                    unit_divisor=1024,
                ) as bar:
                    for chunk in r.iter_content(chunk_size=8192):
                        size = f.write(chunk)
                        bar.update(size)
            print(f"  Successfully downloaded {os.path.basename(destination)}")
            return True # Success
        except requests.exceptions.RequestException as e:
            print(f"  Download attempt failed: {e}")
            if attempt + 1 < retries:
                wait_time = backoff_factor * (2 ** attempt)
                print(f"  Waiting {wait_time} seconds before retrying...")
                time.sleep(wait_time)
            else:
                print("  All retry attempts failed.")
                # Clean up partial file on final failure
                if os.path.exists(destination):
                    os.remove(destination)
                return False

def download_hf_model_manually(repo_id, initial_files, destination_dir):
    """
    Manually downloads all required files, handling sharded models by
    looking for an index file if the main model file is not found.
    """
    print(f"\nVerifying model '{repo_id}' in '{destination_dir}'...")
    os.makedirs(destination_dir, exist_ok=True)
    
    files_to_download = list(initial_files)

    # --- NEW: Logic to handle sharded models ---
    if "model.safetensors" in files_to_download:
        # First, check if the single file actually exists by sending a HEAD request (faster than GET)
        single_model_url = f"{HF_HUB_URL}/{repo_id}/resolve/main/model.safetensors"
        try:
            response = requests.head(single_model_url, timeout=10)
            if response.status_code == 404:
                print("'model.safetensors' not found. Checking for sharded model index...")
                index_filename = "model.safetensors.index.json"
                index_url = f"{HF_HUB_URL}/{repo_id}/resolve/main/{index_filename}"
                index_dest_path = os.path.join(destination_dir, index_filename)

                # Download the index file first
                if not os.path.exists(index_dest_path):
                    if not download_file_with_retries(index_url, index_dest_path):
                        raise Exception("Could not download the sharded model index file.")
                
                # Parse the index to find the real filenames
                with open(index_dest_path, 'r') as f:
                    index_data = json.load(f)
                
                shard_filenames = sorted(list(set(index_data["weight_map"].values())))
                print(f"Found {len(shard_filenames)} shards: {shard_filenames}")

                # Update the list of files to download
                files_to_download.remove("model.safetensors")
                files_to_download.append(index_filename)
                files_to_download.extend(shard_filenames)
        except requests.exceptions.RequestException as e:
            print(f"Could not check for single model file: {e}. Assuming sharded model.")
            # Fallback logic could be added here if needed

    # --- Download loop for all required files ---
    for filename in files_to_download:
        destination_path = os.path.join(destination_dir, filename)
        if os.path.exists(destination_path):
            print(f"'{filename}' already exists. Skipping.")
            continue
        
        url = f"{HF_HUB_URL}/{repo_id}/resolve/main/{filename}"
        print(f"Fetching '{filename}'...")
        if not download_file_with_retries(url, destination_path):
            print(f"FATAL: Failed to download a required file ('{filename}'). Aborting.")
            return False
            
    print(f"All required files for '{repo_id}' are present locally.")
    return True

if __name__ == "__main__":
    print("Starting download of required ML models for offline use.")
    print("This script will fetch models from the public Hugging Face Hub and store them locally.")
    
    success = True
    for model_key, model_info in MODELS_TO_DOWNLOAD.items():
        destination_directory = os.path.join(MODELS_DIR, model_key)
        if not download_hf_model_manually(
            model_info['repo_id'],
            model_info['required_files'],
            destination_directory
        ):
            success = False
            break
            
    if success:
        print("\nModel download process complete. The application can now run in a fully offline environment.")
    else:
        print("\nModel download process failed. Please check the errors above.")