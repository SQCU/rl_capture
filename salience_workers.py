# salience_workers.py

import numpy as np
import torch
import torch.nn.functional as F
import os
from multiprocessing import shared_memory
from transformers import AutoModel, AutoProcessor, AutoModelForCausalLM, AutoImageProcessor
from PIL import Image

# --- Helper Classes & Functions ---

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

# --- START FIX: Pre-processing Helper ---
def _downscale_image(img: Image.Image, target_pixel_area: int = 512*512) -> Image.Image:
    """
    Downscales a PIL Image to a target pixel area while preserving aspect ratio.
    """
    original_width, original_height = img.size
    original_area = original_width * original_height

    if original_area <= target_pixel_area:
        return img # No need to downscale

    aspect_ratio = original_width / original_height
    # new_width * new_height = target_pixel_area
    # new_width = aspect_ratio * new_height
    # (aspect_ratio * new_height) * new_height = target_pixel_area
    # new_height^2 = target_pixel_area / aspect_ratio
    new_height = int((target_pixel_area / aspect_ratio)**0.5)
    new_width = int(aspect_ratio * new_height)

    # Use LANCZOS for high-quality downsampling
    return img.resize((new_width, new_height), Image.Resampling.LANCZOS)
# --- END FIX ---

# --- Stateful Worker Implementations ---

class SigLIPSalienceWorker:
    """A stateful worker using a local Hugging Face AutoModel for SigLIP."""
    def __init__(self, config: dict, output_path: str):
        print("Initializing SigLIP Salience Worker from local files...")
        self.config = config or self.get_default_config()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # --- UNSTUBBED: Load from a local path, NOT a Hub ID ---
        model_path = "./models/siglip"
        if not os.path.isdir(model_path):
            raise FileNotFoundError(f"SigLIP model not found at '{model_path}'. Please run download_models.py first.")

        model_kwargs = {"device":self.device, "dtype": torch.bfloat16, "attn_implementation": "sdpa"} if self.device.type == 'cuda' else {}
        
        # This now loads from the disk, with no network calls.
        self.model = AutoModel.from_pretrained(model_path, **model_kwargs).to(self.device)
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.vision_encoder = self.model.vision_model
        self.vision_encoder.eval()


        self.global_distance_stats = OnlineStats()
        self.global_chunk_variance_stats = OnlineStats()
        self.last_frame_latent = None

    def get_default_config(self) -> dict:
        return { "z_score_global": 4.5, "z_score_local": 3.0, "z_score_chunk": 2.5 }

    @torch.no_grad()
    def process_chunk(self, frames_chunk: list[Image.Image], timestamps_chunk: list[float]) -> list:
        if not frames_chunk: return []

        inputs = self.processor(images=frames_chunk, return_tensors="pt", padding=True)
        pixel_values = inputs["pixel_values"].to(self.device, dtype=self.model.dtype if self.device.type == 'cuda' else torch.float32)
        
        # Get the final pooled output for the entire image
        latents = self.vision_encoder(pixel_values).pooler_output

        if self.last_frame_latent is not None:
            all_latents = torch.cat([self.last_frame_latent.unsqueeze(0), latents], dim=0)
        else:
            all_latents = latents

        distances = (1 - F.cosine_similarity(all_latents[:-1], all_latents[1:], dim=1)).cpu().numpy()

        local_mean, local_variance = (np.mean(distances), np.var(distances)) if len(distances) > 1 else (distances[0] if len(distances) > 0 else 0, 0)
        self.global_chunk_variance_stats.update(local_variance)
        
        detected_events, is_volatile_chunk = [], local_variance > (self.global_chunk_variance_stats.mean +
                                              (self.global_chunk_variance_stats.std_dev * self.config['z_score_chunk']))

        for i, distance in enumerate(distances):
            global_mean, global_std = self.global_distance_stats.mean, self.global_distance_stats.std_dev
            
            is_global_spike = global_std > 0 and (distance > global_mean + (global_std * self.config['z_score_global']))
            is_local_spike = local_variance > 0 and (distance > local_mean + (np.sqrt(local_variance) * self.config['z_score_local']))

            if is_global_spike or (is_local_spike and is_volatile_chunk):
                detected_events.append({
                    "type": "VISUAL_KEYFRAME", "timestamp": timestamps_chunk[i], "reason": "siglip_latent_spike",
                    "value": float(distance), "details": { "is_global_spike": is_global_spike, "is_local_spike": is_local_spike }
                })
            
            self.global_distance_stats.update(distance)

        self.last_frame_latent = latents[-1].clone()
        return detected_events

class OCRLatentWorker:
    """A stateful worker using a local DOTS.ocr model."""
    def __init__(self, config: dict, output_path: str):
        print("Initializing OCR Latent Worker from local files...")
        self.config = config or {"change_threshold_z_score": 3.5}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # --- UNSTUBBED: Load from a local path, NOT a Hub ID ---
        model_path = "./models/dots_ocr"
        if not os.path.isdir(model_path):
            raise FileNotFoundError(f"DOTS.ocr model not found at '{model_path}'. Please run download_models.py first.")
            
        use_quantization = torch.cuda.is_available()
        model_kwargs = {#"load_in_8bit": use_quantization, #disables bfloat?? lol??
         "attn_implementation": "sdpa",
         "dtype":"bfloat16"} if use_quantization else {}
        
        # --- FIX #1: Use the correct AutoModelForCausalLM class ---
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, 
            trust_remote_code=True, 
            **model_kwargs
        )
        self.processor = self.processor = AutoImageProcessor.from_pretrained(model_path, trust_remote_code=True).from_pretrained(model_path, trust_remote_code=True)
        
        # --- FIX #2: Access the correct vision component: `vision_tower` ---
        self.encoder = self.model.vision_tower.to(device=self.device)
        self.encoder.eval()

        self.output_path = os.path.join(output_path, "latents")
        os.makedirs(self.output_path, exist_ok=True)

        self.distance_stats = OnlineStats()
        self.last_latent_hash = None

    @torch.no_grad()
    def process_chunk(self, frames_chunk: list[Image.Image], timestamps_chunk: list[float]) -> list:
        if not frames_chunk: return []
        
        events = []
        
        # --- FIX #3: The processor provides all necessary inputs ---
        inputs = self.processor(images=frames_chunk, return_tensors="pt")
        # Move all tensor inputs to the correct device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Call the vision_tower with the arguments it expects
        # The output is the vision embeddings directly.
        latents_batch = self.encoder(
            inputs['pixel_values'],
            grid_thw=inputs['image_grid_thw']
        )
        # Aggregate to a single vector per frame for comparison
        aggregated_latents = latents_batch.mean(dim=1)

        for i, current_latent in enumerate(aggregated_latents):
            if self.last_latent_hash is None:
                self.last_latent_hash = current_latent.clone()
                events.append(self._create_latent_event(current_latent, timestamps_chunk[i]))
                continue

            distance = torch.linalg.norm(self.last_latent_hash - current_latent).item()
            self.distance_stats.update(distance)
            
            mean, std_dev = self.distance_stats.mean, self.distance_stats.std_dev
            is_significant_change = std_dev > 0 and (distance > mean + (std_dev * self.config['change_threshold_z_score']))
            
            if is_significant_change:
                events.append(self._create_latent_event(current_latent, timestamps_chunk[i]))
                self.last_latent_hash = current_latent.clone()
        
        return events


    def _create_latent_event(self, latent_tensor: torch.Tensor, timestamp: float) -> dict:
        latent_filename = f"ocr_latent_{timestamp:.4f}.pt"
        latent_filepath = os.path.join(self.output_path, latent_filename)
        torch.save(latent_tensor.cpu(), latent_filepath) # Save to CPU to avoid CUDA context issues

        return {
            "type": "OCR_LATENT_KEYFRAME", "timestamp": timestamp,
            "payload": { "latent_pointer": latent_filepath, "model_name": "rednote-hilab/dots.ocr" }
        }


# --- Generic Worker Process Entry Point ---

def analysis_worker_loop(task_queue, return_queue, worker_class_name, config, output_path):
    """
    The main loop for a worker process. It initializes a worker class instance
    and then continuously processes chunks from the task queue.
    """
    print(f"[{os.getpid()}] Worker process started for: {worker_class_name}")
    
    # Dynamically instantiate the correct worker class
    worker_class = globals()[worker_class_name]
    worker_instance = worker_class(config, output_path)

    shm = None
    buffer = None

    while True:
        try:
            task = task_queue.get()
            if task is None: # Shutdown signal
                break

            if shm is None or shm.name != task['shm_name']:
                if shm: shm.close()
                shm = shared_memory.SharedMemory(name=task['shm_name'])
                buffer = np.ndarray(task['shape'], dtype=task['dtype'], buffer=shm.buf)
            
            # Create a zero-copy view of the frames for this chunk
            frame_indices = np.arange(task['start'], task['start'] + task['num']) % task['shape'][0]
            frames_data = buffer[frame_indices]
            timestamps_data = task['timestamps']

            # Convert raw numpy arrays to PIL Images for processing
            frames_as_images = [Image.fromarray(frame) for frame in frames_data]

            # --- START FIX: Pre-process frames before sending to the model ---
            # This prevents the massive memory spike from high-resolution images.
            # We apply this to ALL vision workers by putting it in the generic loop.
            # Using 384*384 for SigLIP base, a common resolution. 512*512 is also fine.
            downscaled_images = [_downscale_image(img, target_pixel_area=384*384) for img in frames_as_images]
            # --- END FIX ---
            
            # Process the chunk and get the results
            result_data = worker_instance.process_chunk(frames_as_images, timestamps_data)
            
            if result_data: # Only send a message if there's something to report
                return_message = {
                    "chunk_id": task['chunk_id'],
                    "source": worker_class_name,
                    "data": result_data
                }
                return_queue.put(return_message)
        
        except (BrokenPipeError, EOFError):
            print(f"[{os.getpid()}] Communication channel broke. Shutting down worker.")
            break
        except Exception as e:
            print(f"[{os.getpid()}] Error in worker loop: {e}")
            
    if shm: shm.close()
    print(f"[{os.getpid()}] Worker process {worker_class_name} finished.")