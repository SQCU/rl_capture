# download_models.py

import os
import requests
from tqdm import tqdm
from transformers import AutoModelForVision2Seq, AutoProcessor

# --- Configuration ---
MODELS_DIR = "./models"
SIGLIP_MODEL_URL = "https://storage.googleapis.com/big_vision/siglip2/siglip2_b16_512.npz"
SIGLIP_MODEL_FILENAME = "siglip2_b16_512.npz"
DOTS_MODEL_HF_ID = "rednote-hilab/dots.ocr"

# --- Helper Function ---
def download_file(url, destination):
    """Downloads a file with a progress bar."""
    try:
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            total_size = int(r.headers.get('content-length', 0))
            with open(destination, 'wb') as f, tqdm(
                desc=destination.split('/')[-1],
                total=total_size,
                unit='iB',
                unit_scale=True,
                unit_divisor=1024,
            ) as bar:
                for chunk in r.iter_content(chunk_size=8192):
                    size = f.write(chunk)
                    bar.update(size)
        print(f"Successfully downloaded {destination}")
    except Exception as e:
        print(f"Error downloading {url}: {e}")

def download_siglip_model():
    """
    Downloads the SigLIP model weights from Google's public storage.
    Note: These are raw weights, not a Hugging Face model. Our worker
    would need code to load this .npz file into a PyTorch model definition.
    """
    print("--- Downloading SigLIP Model ---")
    siglip_dir = os.path.join(MODELS_DIR, "siglip")
    os.makedirs(siglip_dir, exist_ok=True)
    destination = os.path.join(siglip_dir, SIGLIP_MODEL_FILENAME)

    if not os.path.exists(destination):
        download_file(SIGLIP_MODEL_URL, destination)
    else:
        print(f"SigLIP model already exists at {destination}. Skipping.")

def download_dots_ocr_model():
    """
    Downloads the DOTS OCR model and processor using the Hugging Face Hub.
    This is the recommended way as it handles the complexity automatically.
    """
    print("\n--- Downloading DOTS.ocr Model ---")
    dots_dir = os.path.join(MODELS_DIR, "dots_ocr")
    os.makedirs(dots_dir, exist_ok=True)
    
    try:
        print(f"Downloading model '{DOTS_MODEL_HF_ID}' to cache dir '{dots_dir}'...")
        # The cache_dir ensures the model is saved within our project structure.
        AutoModelForVision2Seq.from_pretrained(DOTS_MODEL_HF_ID, cache_dir=dots_dir)
        AutoProcessor.from_pretrained(DOTS_MODEL_HF_ID, cache_dir=dots_dir)
        print("Successfully downloaded DOTS.ocr model and processor.")
    except Exception as e:
        print(f"Could not download model from Hugging Face Hub: {e}")
        print("Please ensure you have an internet connection and have logged in via `huggingface-cli login` if necessary.")

if __name__ == "__main__":
    print("Starting download of required ML models...")
    os.makedirs(MODELS_DIR, exist_ok=True)
    download_siglip_model()
    download_dots_ocr_model()
    print("\nModel download process complete.")