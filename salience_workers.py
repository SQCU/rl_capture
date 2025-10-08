# salience_workers.py

# --- stubs_and_dependencies ---
# Define the stubs and helper classes our worker will depend on.


# These functions are called by the generic worker loop

import imagehash
import numpy as np

# A simple class to track rolling statistics without a heavy library
class RollingStats:
    def __init__(self, window_size=30):
        self.samples = []
        self.window_size = window_size
    def add(self, value):
        self.samples.append(value)
        if len(self.samples) > self.window_size:
            self.samples.pop(0)
    def mean(self):
        return np.mean(self.samples) if self.samples else 0
    def std(self):
        return np.std(self.samples) if len(self.samples) > 1 else 0

def visual_salience_analyzer(frames_chunk, timestamps_chunk):
    """Analyzes a chunk of frames for pHash spikes."""
    results = []
    # Configurable thresholds
    PHASH_SPIKE_STD_FACTOR = 5.0
    PHASH_SPIKE_ABS_THRESHOLD = 15 # Hamming distance

    stats = RollingStats(window_size=60) # Stats over ~1-2 seconds
    last_hash = None

    for i, frame_pil in enumerate(frames_chunk): # Assuming frames are PIL Images
        current_hash = imagehash.phash(frame_pil)
        
        if last_hash is not None:
            distance = current_hash - last_hash
            
            # The core detection logic
            is_spike = (distance > PHASH_SPIKE_ABS_THRESHOLD and
                        distance > stats.mean() + (stats.std() * PHASH_SPIKE_STD_FACTOR))
            
            if is_spike:
                event_timestamp = timestamps_chunk[i]
                results.append({
                    "type": "VISUAL_KEYFRAME",
                    "timestamp": event_timestamp,
                    "reason": "pHash_spike",
                    "value": distance,
                    "rolling_avg": stats.mean()
                })
            
            stats.add(distance)
        last_hash = current_hash
        
    return results # Return a list of all keyframes found in the chunk

def ocr_analyzer(frames_chunk, timestamps_chunk):
    """Analyzes a chunk for OCR changes."""
    # Placeholder for running a real OCR model and comparing latent vectors
    # A real implementation would compare latents from frame to frame.
    # For this pseudocode, we'll just pretend it finds one event.
    
    # Simulate finding text halfway through the chunk
    text_appears_at_index = len(frames_chunk) // 2
    event_timestamp = timestamps_chunk[text_appears_at_index]
    
    return [{
        "type": "OCR_KEYFRAME",
        "timestamp": event_timestamp,
        "image_quality_requirement": "LOSSLESS", # Signal to save a PNG
        "ocr_text": "Quest Started",
        "latents": [0.1, 0.5, ...] # The actual latent data
    }]


import numpy as np
import torch
import torch.nn.functional as F

class OnlineStats:
    """Implements Welford's algorithm for stable online variance calculation."""
    def __init__(self):
        self.n = 0
        self.mean = 0.0
        self.M2 = 0.0

    def update(self, new_value: float):
        self.n += 1
        delta = new_value - self.mean
        self.mean += delta / self.n
        delta2 = new_value - self.mean
        self.M2 += delta * delta2

    @property
    def variance(self) -> float:
        return self.M2 / self.n if self.n > 1 else 0.0

    @property
    def std_dev(self) -> float:
        return self.variance**0.5

# --- STUB FUNCTIONS ---
# These represent complex operations that would be implemented using libraries
# like Hugging Face's `transformers`, `PIL`, and `torchvision`.

def load_siglip_model_and_processor(model_name: str = "google/siglip-base-patch16-224"):
    """
    STUB: Loads a SigLIP model and its associated image processor.
    In reality, this uses the `transformers` library.
    """
    print(f"STUB: Loading model '{model_name}' to CUDA device...")
    # model = AutoModel.from_pretrained(model_name).to("cuda")
    # processor = AutoProcessor.from_pretrained(model_name)
    model = torch.nn.Identity() # Placeholder model
    processor = None            # Placeholder processor
    return model, processor

def preprocess_frames_batch(processor, frames_chunk: list):
    """
    STUB: Uses the model's processor to convert a list of raw image frames
    (e.g., from MSS, as numpy arrays) into a single batched tensor.
    """
    # In reality: processor(images=frames_chunk, return_tensors="pt")["pixel_values"]
    print(f"STUB: Preprocessing a batch of {len(frames_chunk)} frames.")
    # Return a correctly shaped random tensor for pseudocode execution
    return torch.randn(len(frames_chunk), 3, 224, 224)


# --- siglip_salience_worker.py ---

class SigLIPSalienceWorker:
    """
    A stateful worker that analyzes a continuous stream of frame chunks
    to detect semantically salient visual events using SigLIP latents.
    """
    def __init__(self, config: dict = None):
        print("Initializing SigLIP Salience Worker...")
        self.config = config or self.get_default_config()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model, self.processor = load_siglip_model_and_processor()
        self.model.eval() # Set model to evaluation mode

        # --- STATEFUL PROPERTIES ---
        # Online statistics for adaptive thresholding
        self.global_distance_stats = OnlineStats()
        self.global_chunk_variance_stats = OnlineStats()

        # State for maintaining continuity between chunks
        self.last_frame_latent = None # Stores the latent vector of the previous chunk's last frame

    def get_default_config(self) -> dict:
        """Provides default Z-score thresholds, making them easy to tune."""
        return {
            "z_score_global": 4.5, # How many std devs for a frame-to-frame change to be a "major global event"
            "z_score_local": 3.0,  # How many std devs for a change to be a spike *within its own chunk*
            "z_score_chunk": 2.5   # How many std devs for a chunk's variance to be considered "highly volatile"
        }

    @torch.no_grad() # Ensure no gradients are computed, saving memory and computation
    def process_chunk(self, frames_chunk: list, timestamps_chunk: list) -> list:
        """
        Processes a single chunk of frames and returns a list of detected keyframe events.
        """
        if not frames_chunk:
            return []

        # 1. BATCH INFERENCE: Get latents for the entire chunk at once for GPU efficiency.
        batch_tensor = preprocess_frames_batch(self.processor, frames_chunk).to(self.device)
        latents = self.model(batch_tensor) # In a real ViT, this would be `model.get_image_features(...)`

        # 2. CALCULATE DISTANCES: Compute frame-to-frame change, including the boundary.
        # Prepend the last frame's latent from the previous chunk to ensure continuity.
        if self.last_frame_latent is not None:
            all_latents = torch.cat([self.last_frame_latent.unsqueeze(0), latents], dim=0)
        else:
            all_latents = latents

        # Cosine distance = 1 - cosine similarity
        similarities = F.cosine_similarity(all_latents[:-1], all_latents[1:], dim=1)
        distances = (1 - similarities).cpu().numpy()

        # 3. CALCULATE LOCAL & UPDATE GLOBAL CHUNK STATS
        if len(distances) > 1:
            local_mean = np.mean(distances)
            local_variance = np.var(distances)
        else: # Handle chunks with only one valid distance
            local_mean = distances[0] if len(distances) > 0 else 0
            local_variance = 0
        
        self.global_chunk_variance_stats.update(local_variance)
        
        # 4. KEYFRAME DETECTION LOOP with adaptive Z-score logic
        detected_events = []
        is_volatile_chunk = local_variance > (self.global_chunk_variance_stats.mean +
                                              (self.global_chunk_variance_stats.std_dev * self.config['z_score_chunk']))

        for i, distance in enumerate(distances):
            global_mean = self.global_distance_stats.mean
            global_std = self.global_distance_stats.std_dev

            # --- The Core Adaptive Trigger Logic ---
            is_global_spike = global_std > 0 and (distance > global_mean + (global_std * self.config['z_score_global']))
            is_local_spike = local_variance > 0 and (distance > local_mean + (np.sqrt(local_variance) * self.config['z_score_local']))

            if is_global_spike or (is_local_spike and is_volatile_chunk):
                event = {
                    "type": "VISUAL_KEYFRAME",
                    "timestamp": timestamps_chunk[i],
                    "reason": "siglip_latent_spike",
                    "value": float(distance),
                    "details": {
                        "is_global_spike": is_global_spike,
                        "is_local_spike": is_local_spike,
                        "is_volatile_chunk": is_volatile_chunk,
                        "global_mean_at_t": global_mean,
                        "local_mean_at_t": local_mean
                    }
                }
                detected_events.append(event)
            
            # Update the global distance stats for the next frame's calculation
            self.global_distance_stats.update(distance)

        # 5. FINALIZE STATE for the next chunk
        self.last_frame_latent = latents[-1].clone() # Clone to prevent memory issues

        return detected_events

# --- stubs_and_dependencies.py ---
# (Includes OnlineStats, stubs for model loading, etc., from previous response)
import numpy as np
import torch
from scipy.spatial.distance import cdist # For efficient distance calculations

# --- STUB FUNCTIONS for OCR ---

def load_ocr_model(model_name="naver-clova-ix/parseq-base"):
    """STUB: Loads a powerful Transformer-based OCR model."""
    print(f"STUB: Loading OCR model '{model_name}' to CUDA device...")
    # model = ...
    # processor = ...
    model, processor = (torch.nn.Identity(), None) # Placeholders
    return model, processor

def detect_text_regions_batch(frames_chunk: list) -> list:
    """
    STUB: A crucial optimization. Runs a fast text *detection* model (e.g., DBNet)
    to find bounding boxes of potential text on the screen for each frame.
    Returns a list of lists of bounding boxes.
    `[[bbox1, bbox2], [bbox1], ...]` for each frame in the chunk.
    """
    print(f"STUB: Detecting text regions in {len(frames_chunk)} frames...")
    # Simulate finding one box per frame for simplicity
    return [[(100, 100, 300, 150)] for _ in frames_chunk]

def crop_and_batch_regions(frames_chunk: list, bboxes_per_frame: list):
    """
    STUB: Crops the image regions defined by the bounding boxes and prepares them
    for batch processing by the main OCR model.
    """
    print("STUB: Cropping and batching detected text regions.")
    # Return a dummy tensor and a list of metadata to track which crop is which
    dummy_tensor = torch.randn(len(frames_chunk), 3, 32, 128) # Typical OCR size
    crop_metadata = [{"frame_idx": i, "bbox": bboxes[0]} for i, bboxes in enumerate(bboxes_per_frame)]
    return dummy_tensor, crop_metadata


# --- stubs_and_dependencies.py ---
# (Includes OnlineStats, stubs for model loading, etc.)
import torch
import os

# --- STUB for DOTS.ocr Encoder ---
def get_image_encoder_latents_batch(model, processor, frames_chunk: list):
    """
    STUB: Runs ONLY THE ENCODER part of the dense OCR model on a batch of frames.
    The output is a single latent tensor representing the textual content of each frame.
    """
    print(f"STUB: Running DOTS *encoder* on {len(frames_chunk)} frames...")
    # A real implementation would call `model.encoder(pixel_values)`
    # The output shape might be (batch_size, sequence_length, hidden_dim),
    # so we'll take the mean to get a single vector per image for comparison.
    batch_size = len(frames_chunk)
    latents = torch.randn(batch_size, 256, 768) # (batch, seq_len, hidden_dim)
    # Aggregate to a single vector per frame for a simple "whole screen" text hash
    aggregated_latents = latents.mean(dim=1) # Shape: (batch_size, hidden_dim)
    return aggregated_latents


# --- ocr_latent_worker.py ---

class OCRLatentWorker:
    """
    A stateful worker that uniformly samples frames to detect significant changes
    in the OCR model's latent space, saving the raw latents for offline decoding.
    """
    def __init__(self, config: dict, output_path: str = "./capture_run/latents"):
        self.config = config or {"change_threshold_z_score": 3.5}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model, self.processor = ("dots_model_stub", "dots_processor_stub")
        self.output_path = output_path
        os.makedirs(self.output_path, exist_ok=True)

        # --- STATEFUL PROPERTIES ---
        self.distance_stats = OnlineStats()
        self.last_latent_hash = None # The last latent that was *saved*
        self.frame_counter = 0

    @torch.no_grad()
    def process_chunk(self, frames_chunk: list, timestamps_chunk: list) -> list:
        """Processes a chunk and returns events pointing to saved latent files."""
        events = []

        # We can analyze all frames in the chunk since it's a fast operation
        latents_batch = get_image_encoder_latents_batch(self.model, self.processor, frames_chunk)

        for i, current_latent in enumerate(latents_batch):
            if self.last_latent_hash is None:
                # First frame ever, always save it as the baseline.
                self.last_latent_hash = current_latent.clone()
                events.append(self._create_latent_event(current_latent, timestamps_chunk[i]))
                continue

            # Compare the current frame's latent to the last *saved* latent
            distance = torch.linalg.norm(self.last_latent_hash - current_latent).item()
            
            # Update stats with every single frame-to-frame comparison for a robust baseline
            self.distance_stats.update(distance)
            
            mean = self.distance_stats.mean
            std_dev = self.distance_stats.std_dev
            
            # Adaptive threshold check
            is_significant_change = std_dev > 0 and (distance > mean + (std_dev * self.config['change_threshold_z_score']))
            
            if is_significant_change:
                # The change is significant! Save this new latent as the current ground truth.
                events.append(self._create_latent_event(current_latent, timestamps_chunk[i]))
                self.last_latent_hash = current_latent.clone()
        
        return events

    def _create_latent_event(self, latent_tensor: torch.Tensor, timestamp: float) -> dict:
        """Saves a latent tensor to disk and creates the corresponding event object."""
        
        # Save the raw latent tensor to a binary file
        latent_filename = f"ocr_latent_{timestamp:.4f}.pt"
        latent_filepath = os.path.join(self.output_path, latent_filename)
        # torch.save(latent_tensor, latent_filepath) # The real save operation

        print(f"STUB: Saving new OCR latent to {latent_filepath}")

        # The event payload is now just a pointer to this file.
        event = {
            "type": "OCR_LATENT_KEYFRAME",
            "timestamp": timestamp, # This is an instantaneous event
            "payload": {
                "latent_pointer": latent_filepath,
                "model_name": "rednote-hilab/dots.ocr" # For future-proofing
            }
        }
        return event
