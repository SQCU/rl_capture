# salience_workers.py

import numpy as np
import torch
import torch.nn.functional as F
import os
from multiprocessing import shared_memory, Event
from transformers import AutoModel, AutoProcessor, AutoModelForCausalLM, AutoImageProcessor
from PIL import Image
from collections import deque

# --- Helper Classes & Functions (Unchanged) ---

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

def _downscale_image(img: Image.Image, target_pixel_area: int = 512*512) -> Image.Image:
    """
    Downscales a PIL Image to a target pixel area while preserving aspect ratio.
    """
    original_width, original_height = img.size
    original_area = original_width * original_height

    if original_area <= target_pixel_area:
        return img

    aspect_ratio = original_width / original_height
    new_height = int((target_pixel_area / aspect_ratio)**0.5)
    new_width = int(aspect_ratio * new_height)

    return img.resize((new_width, new_height), Image.Resampling.LANCZOS)

# --- Base Worker with Hierarchical Logic ---

class BaseSalienceWorker:
    """
    --- NEW ---
    A base class that contains the shared hierarchical search logic.
    """
    def __init__(self, config, output_path):
        self.config = self.get_default_config()
        if config: self.config.update(config)
        
        self.output_path = output_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.distance_stats = OnlineStats()
        self.latent_cache = {} # Cache latents within a single chunk processing call
        self.model = None
        self.processor = None

    def get_default_config(self) -> dict:
        # --- NEW: Configuration for the search ---
        return {
            "branching_factor": 8, # How many sentinel frames to check per chunk
            "z_score_threshold": 3.0, # Z-score to trigger exhaustive search of a sub-span
            "min_exhaustive_size": 16, # Chunks smaller than this are always processed fully
        }

    @torch.no_grad()
    def _get_latents_for_indices(self, frames_chunk: list[Image.Image], indices: list[int]) -> torch.Tensor:
        """Helper to compute or retrieve latents from cache for specific indices."""
        indices_to_compute = [i for i in indices if i not in self.latent_cache]
        
        if indices_to_compute:
            images_to_process = [frames_chunk[i] for i in indices_to_compute]
            
            # --- MODIFIED: Use a unified processing method ---
            new_latents = self._inference(images_to_process)

            for i, latent in zip(indices_to_compute, new_latents):
                self.latent_cache[i] = latent.to('cpu', non_blocking=True) # Cache on CPU
        
        # Retrieve all required latents (now on CPU) and stack them for the current device
        return torch.stack([self.latent_cache[i] for i in indices]).to(self.device)

    def _inference(self, images: list[Image.Image]) -> torch.Tensor:
        """This method must be implemented by subclasses."""
        raise NotImplementedError

    def _create_keyframe_event(self, timestamp: float, distance: float, reason: str) -> dict:
        """This method must be implemented by subclasses."""
        raise NotImplementedError

    def process_chunk(self, frames_chunk: list[Image.Image], timestamps_chunk: list[float], shutdown_event: Event) -> list:
        """
        --- NEW: Main entry point that kicks off the hierarchical search. ---
        """
        self.latent_cache = {} # Clear cache for each new chunk
        if not frames_chunk: return []
        
        # The search queue holds tuples of (start_index, end_index) to investigate
        search_queue = deque([(0, len(frames_chunk) - 1)])
        final_events = []

        while search_queue:
            if shutdown_event.is_set():
                print(f"[{os.getpid()}] Shutdown detected, aborting chunk processing.")
                return [] # Abort gracefully

            start_idx, end_idx = search_queue.popleft()
            num_frames_in_span = end_idx - start_idx + 1

            if num_frames_in_span <= self.config['min_exhaustive_size']:
                # --- Base Case: Exhaustive Search ---
                # This span is small enough, process every frame within it.
                span_indices = list(range(start_idx, end_idx + 1))
                if len(span_indices) < 2: continue
                
                latents = self._get_latents_for_indices(frames_chunk, span_indices)
                distances = (1 - F.cosine_similarity(latents[:-1], latents[1:], dim=1)).cpu().numpy()

                for i, distance in enumerate(distances):
                    mean, std_dev = self.distance_stats.mean, self.distance_stats.std_dev
                    is_spike = std_dev > 0 and (distance - mean) > (std_dev * self.config['z_score_threshold'])

                    if is_spike:
                        # The actual event is at the *end* of the pair, so i+1
                        event_index = start_idx + i + 1
                        event = self._create_keyframe_event(
                            timestamps_chunk[event_index],
                            float(distance),
                            "exhaustive_search_spike"
                        )
                        final_events.append(event)
                    
                    self.distance_stats.update(distance)
                continue

            # --- Recursive Step: Branching Search ---
            # 1. Select sentinel frames
            sentinel_indices = np.linspace(start_idx, end_idx, self.config['branching_factor'], dtype=int).tolist()
            sentinel_latents = self._get_latents_for_indices(frames_chunk, sentinel_indices)

            # 2. Calculate distances between sentinels
            sentinel_distances = (1 - F.cosine_similarity(sentinel_latents[:-1], sentinel_latents[1:], dim=1)).cpu().numpy()
            
            # 3. Analyze sub-spans
            found_hotspot = False
            max_dist = -1
            best_span = None

            for i, dist in enumerate(sentinel_distances):
                mean, std_dev = self.distance_stats.mean, self.distance_stats.std_dev
                is_significant = std_dev > 0 and (dist - mean) > (std_dev * self.config['z_score_threshold'])
                
                sub_span_start = sentinel_indices[i]
                sub_span_end = sentinel_indices[i+1]
                
                if is_significant:
                    # This sub-span is interesting, queue it for a deeper look
                    search_queue.append((sub_span_start, sub_span_end))
                    found_hotspot = True
                
                if dist > max_dist:
                    max_dist = dist
                    best_span = (sub_span_start, sub_span_end)

            # 4. If no span was above the threshold, queue only the most different one
            if not found_hotspot and best_span is not None:
                search_queue.append(best_span)

        return final_events

# --- Stateful Worker Implementations ---

class SigLIPSalienceWorker(BaseSalienceWorker):
    """A stateful worker using SigLIP, now with hierarchical search."""
    def __init__(self, config: dict, output_path: str):
        super().__init__(config, output_path)
        print("Initializing SigLIP Salience Worker...")
        model_path = "./models/siglip"
        if not os.path.isdir(model_path):
            raise FileNotFoundError(f"SigLIP model not found at '{model_path}'.")

        model_kwargs = {"dtype": torch.bfloat16, "attn_implementation": "sdpa"} if self.device.type == 'cuda' else {}
        self.model = AutoModel.from_pretrained(model_path, **model_kwargs).to(self.device)
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.vision_encoder = self.model.vision_model
        self.vision_encoder.eval()

    def _inference(self, images: list[Image.Image]) -> torch.Tensor:
        inputs = self.processor(images=images, return_tensors="pt", padding=True)
        pixel_values = inputs["pixel_values"].to(self.device, dtype=self.model.dtype)
        return self.vision_encoder(pixel_values).pooler_output

    def _create_keyframe_event(self, timestamp: float, distance: float, reason: str) -> dict:
        return {
            "type": "VISUAL_KEYFRAME", "timestamp": timestamp,
            "reason": reason, "value": distance
        }


class OCRLatentWorker(BaseSalienceWorker):
    """A stateful worker using DOTS.ocr, now with hierarchical search."""
    def __init__(self, config: dict, output_path: str):
        super().__init__(config, output_path)
        print("Initializing OCR Latent Worker...")
        model_path = "./models/dots_ocr"
        if not os.path.isdir(model_path):
            raise FileNotFoundError(f"DOTS.ocr model not found at '{model_path}'.")

        model_kwargs = {"attn_implementation": "sdpa", "dtype": torch.bfloat16} if self.device.type == 'cuda' else {}
        self.model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, **model_kwargs)
        self.processor = AutoImageProcessor.from_pretrained(model_path, trust_remote_code=True)
        self.encoder = self.model.vision_tower.to(device=self.device)
        self.encoder.eval()

        latents_path = os.path.join(output_path, "latents")
        os.makedirs(latents_path, exist_ok=True)
        self.latents_path = latents_path

    def _inference(self, images: list[Image.Image]) -> torch.Tensor:
        inputs = self.processor(images=images, return_tensors="pt")
        pixel_values = inputs['pixel_values'].to(self.device, dtype=self.model.dtype)
        grid_thw = inputs['image_grid_thw'].to(self.device)
        # Flatten the spatial dimensions to get one vector per frame
        return self.encoder(pixel_values, grid_thw=grid_thw).flatten(start_dim=1)

    def _create_keyframe_event(self, timestamp: float, distance: float, reason: str) -> dict:
        # For OCR, the event itself doesn't need to save the latent,
        # but you might want to trigger a separate, more detailed OCR process here.
        # For now, we'll just log the keyframe.
        return {
            "type": "OCR_LATENT_KEYFRAME", "timestamp": timestamp,
            "reason": reason, "value": distance,
            "payload": {"model_name": "rednote-hilab/dots.ocr"}
        }

# --- Generic Worker Process Entry Point ---

def analysis_worker_loop(task_queue, return_queue, worker_class_name, config, output_path, shutdown_event): # --- MODIFIED ---
    """
    The main loop for a worker process. Now checks a shutdown event.
    """
    print(f"[{os.getpid()}] Worker process started for: {worker_class_name}")
    
    worker_class = globals()[worker_class_name]
    worker_instance = worker_class(config, output_path)

    shm = None
    buffer = None

    while True:
        try:
            # --- MODIFIED: Non-blocking check for shutdown before getting a task ---
            if shutdown_event.is_set():
                print(f"[{os.getpid()}] Shutdown signal received, exiting loop.")
                break

            task = task_queue.get(timeout=0.1) # Use timeout to remain responsive
            if task is None:
                break

            if shm is None or shm.name != task['shm_name']:
                if shm: shm.close()
                shm = shared_memory.SharedMemory(name=task['shm_name'])
                buffer = np.ndarray(task['shape'], dtype=task['dtype'], buffer=shm.buf)
            
            frame_indices = np.arange(task['start'], task['start'] + task['num']) % task['shape'][0]
            frames_data = buffer[frame_indices]
            timestamps_data = task['timestamps']

            frames_as_images = [Image.fromarray(frame) for frame in frames_data]
            downscaled_images = [_downscale_image(img, target_pixel_area=384*384) for img in frames_as_images]
            
            # --- MODIFIED: Call the new hierarchical processor ---
            result_data = worker_instance.process_chunk(downscaled_images, timestamps_data, shutdown_event)
            
            if result_data:
                return_message = {
                    "chunk_id": task['chunk_id'],
                    "source": worker_class_name,
                    "data": result_data
                }
                return_queue.put(return_message)
        
        except queue.Empty: # Comes from task_queue.get(timeout=0.1)
            continue # This is normal, just loop again and check for shutdown
        except (BrokenPipeError, EOFError):
            print(f"[{os.getpid()}] Communication channel broke. Shutting down worker.")
            break
        except Exception as e:
            print(f"[{os.getpid()}] Error in worker loop: {e}")
            import traceback
            traceback.print_exc() # Print full traceback for debugging
            
    if shm: shm.close()
    print(f"[{os.getpid()}] Worker process {worker_class_name} finished.")