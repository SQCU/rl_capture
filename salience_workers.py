# salience_workers.py

import numpy as np
import torch
import torch.nn.functional as F
import os
import time  # --- NEW ---
from tqdm import tqdm # --- NEW ---
from multiprocessing import shared_memory, Event
from transformers import AutoModel, AutoProcessor, AutoModelForCausalLM, AutoImageProcessor
from PIL import Image
from collections import deque
import queue # --- NEW ---


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
        # --- NEW: Add a worker identifier for clear logging ---
        self.worker_id = f"{self.__class__.__name__}_{os.getpid()}"

    def get_default_config(self) -> dict:
        # --- NEW: Configuration for the search ---
        return {
            "branching_factor": 8, # How many sentinel frames to check per chunk
            "z_score_threshold": 3.0, # Z-score to trigger exhaustive search of a sub-span
            "min_exhaustive_size": 16, # Chunks smaller than this are always processed fully
            "max_batch_size": 16, # A safe default for most modern GPUs
            "target_resolution": 224, # The side length
        }

    @torch.no_grad()
    def _get_latents_for_indices(self, frames_chunk: list[Image.Image], indices: list[int]) -> torch.Tensor:
        """
        --- MODIFIED: This function now slices work into hardware-safe batches. ---
        Helper to compute or retrieve latents from cache for specific indices.
        """
        indices_to_compute = [i for i in indices if i not in self.latent_cache]
        
        if indices_to_compute:
            # --- NEW: Batching loop to prevent VRAM overflow ---
            # This is the core of the fix. We iterate through the work in chunks.
            all_newly_computed_latents = {}
            
            for i in range(0, len(indices_to_compute), self.config['max_batch_size']):
                # 1. Slice the work into a hardware-safe mini-batch
                batch_indices_to_compute = indices_to_compute[i : i + self.config['max_batch_size']]
                
                images_to_process = [frames_chunk[idx] for idx in batch_indices_to_compute]
                
                # 2. Perform inference on just this small batch
                batch_size = len(images_to_process)
                start_time = time.time()
                
                new_latents_batch = self._inference(images_to_process)
                
                if self.device.type == 'cuda': torch.cuda.synchronize()
                end_time = time.time()
                elapsed = end_time - start_time
                it_per_sec = batch_size / elapsed if elapsed > 0 else float('inf')

                # The diagnostic print now fires for each mini-batch, giving more granular feedback
                print(f"[{self.worker_id}] GPU INFERENCE | Batch: {batch_size:<3d} | Time: {elapsed:.3f}s | it/s: {it_per_sec:<8.2f}")

                # 3. Collect the results
                for idx, latent in zip(batch_indices_to_compute, new_latents_batch):
                    all_newly_computed_latents[idx] = latent.to('cpu', non_blocking=True)
            
            # 4. Update the main cache with all the results from the loops
            self.latent_cache.update(all_newly_computed_latents)

        # The final step is unchanged: retrieve all required latents from the now-populated cache
        return torch.stack([self.latent_cache[i] for i in indices]).to(self.device)

    def _inference(self, images: list[Image.Image]) -> torch.Tensor:
        """This method must be implemented by subclasses."""
        raise NotImplementedError

    def _create_keyframe_event(self, timestamp: float, distance: float, reason: str) -> dict:
        """This method must be implemented by subclasses."""
        raise NotImplementedError

    def process_chunk(self, frames_chunk: list[Image.Image], timestamps_chunk: list[float], shutdown_event: Event) -> list:
        """
        --- MODIFIED: Main entry point with TQDM progress bars. ---
        """
        self.latent_cache = {}
        if not frames_chunk: return []
        
        # --- MODIFIED: Queue now stores (start, end, depth) for better logging ---
        search_queue = deque([(0, len(frames_chunk) - 1, 1)])
        final_events = []
        
        # --- NEW: Outer progress bar for the entire chunk search ---
        with tqdm(total=len(frames_chunk), desc=f"[{self.worker_id}] Search", unit="frame", position=0) as search_pbar:
            while search_queue:
                if shutdown_event.is_set():
                    print(f"[{self.worker_id}] Shutdown detected, aborting chunk processing.")
                    return []

                start_idx, end_idx, depth = search_queue.popleft()
                num_frames_in_span = end_idx - start_idx + 1

                # Update the main progress bar's description
                search_pbar.set_postfix_str(f"Queue: {len(search_queue)}, Depth: {depth}")

                if num_frames_in_span <= self.config['min_exhaustive_size']:
                    # --- Base Case: Exhaustive Search ---
                    span_indices = list(range(start_idx, end_idx + 1))
                    if len(span_indices) < 2:
                        search_pbar.update(num_frames_in_span) # Mark these frames as processed
                        continue
                    
                    latents = self._get_latents_for_indices(frames_chunk, span_indices)
                    distances = (1 - F.cosine_similarity(latents[:-1], latents[1:], dim=1)).cpu().float().numpy()
                    
                    # --- NEW: Inner progress bar for the exhaustive scan ---
                    kernel_desc = f"[{self.worker_id}] Kernel (Depth {depth})"
                    for i, distance in enumerate(tqdm(distances, desc=kernel_desc, unit="comp", leave=False, position=1)):
                        mean, std_dev = self.distance_stats.mean, self.distance_stats.std_dev
                        is_spike = std_dev > 0 and (distance - mean) > (std_dev * self.config['z_score_threshold'])

                        if is_spike:
                            event_index = start_idx + i + 1
                            event = self._create_keyframe_event(timestamps_chunk[event_index], float(distance), "exhaustive_search_spike")
                            final_events.append(event)
                        
                        self.distance_stats.update(distance)
                    
                    search_pbar.update(num_frames_in_span) # Mark the whole span as processed
                    continue

                # In process_chunk, recursive step:
                # --- Single, Efficient Fetch ---
                sentinel_latents = self._get_latents_for_indices(frames_chunk, sentinel_indices) # This returns a tensor already on the correct device.

                # Now use it directly
                sentinel_distances = (1 - F.cosine_similarity(sentinel_latents[:-1], sentinel_latents[1:], dim=1)).cpu().float().numpy()
                
                found_hotspot = False
                max_dist = -1
                best_span = None

                for i, dist in enumerate(sentinel_distances):
                    mean, std_dev = self.distance_stats.mean, self.distance_stats.std_dev
                    is_significant = std_dev > 0 and (dist - mean) > (std_dev * self.config['z_score_threshold'])
                    
                    sub_span_start = sentinel_indices[i]
                    sub_span_end = sentinel_indices[i+1]
                    
                    if is_significant:
                        search_queue.append((sub_span_start, sub_span_end, depth + 1))
                        found_hotspot = True
                    
                    if dist > max_dist:
                        max_dist = dist
                        best_span = (sub_span_start, sub_span_end)

                if not found_hotspot and best_span is not None:
                    search_queue.append((*best_span, depth + 1))
                
                # Mark the space between sentinels as "processed" at this depth
                # This updates the progress bar by 8 (the number of sentinels), 
                # not by the number of frames that have been "cleared" from the search. 
                # This will cause the progress bar to behave erratically and likely not finish at exactly 100%. 
                #search_pbar.update(self.config['branching_factor'])

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
            if shutdown_event.is_set():
                break
            # --- MODIFIED: Add timeout to keep the loop responsive ---
            task = task_queue.get(timeout=0.1)
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
            #delocalize config
            res = worker_instance.config['target_resolution']
            downscaled_images = [_downscale_image(img, target_pixel_area=res*res) for img in frames_as_images]
            
            # --- MODIFIED: Call the new hierarchical processor ---
            result_data = worker_instance.process_chunk(downscaled_images, timestamps_data, shutdown_event)
            
            if result_data:
                return_message = {
                    "shm_name": task['shm_name'],     # Pass this through
                    "shape": task['shape'],           # Pass this through
                    "dtype": task['dtype'],           # Pass this through
                    "timestamps": task['timestamps'], # Pass this through
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