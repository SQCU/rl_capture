        

### The Offline Post-Processing Script (`generate_transcripts.py`)
# This is a completely separate program you run after the capture is done.

import pandas as pd
import torch

def generate_transcripts(capture_path: str):
    """
    'Loads saved OCR latents from a capture run and decodes them into a text transcript.'
    """
    events_df = pd.read_parquet(f"{capture_path}/events.parquet")
    ocr_events = events_df[events_df['stream_type'] == 'OCR_LATENT_KEYFRAME']

    # Load the full OCR model (encoder and decoder)
    # model, processor = load_full_ocr_model("rednote-hilab/dots.ocr")

    transcripts = []
    for index, event in ocr_events.iterrows():
        latent_path = event['payload']['latent_pointer']
        timestamp = event['timestamp']
        
        # latent_tensor = torch.load(latent_path)
        
        # --- STUB for DECODING ---
        # decoded_text = model.decoder.generate(latent_tensor.unsqueeze(0))
        decoded_text = f"Text from latent at {timestamp}" # Placeholder
        
        transcripts.append({
            "timestamp": timestamp,
            "text": decoded_text
        })
        print(f"[{timestamp:.2f}s] Decoded Text: {decoded_text}")
    
    # Save the final transcripts to a file
    # save_to_json(transcripts, f"{capture_path}/transcript.json")