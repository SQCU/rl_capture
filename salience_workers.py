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

# --- NEW: Import our new tracker classes ---
from salience_trackers import NaiveZScoreTracker, PCAMahanalobisTracker, TRACKER_STRATEGIES

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

# =====================================================================================
#  NEW ARCHITECTURE: Salience Kernels (The "Experts")
# =====================================================================================

class SalienceKernel:
    """
    Abstract Base Class for a self-contained "expert" on one type of salience.
    It bundles a model with its own private statistical tracker.
    """
    def __init__(self, config: dict):
        self.kernel_name = "BaseKernel"
        self.device = config['device']
        self.config = config
        
        # Each kernel gets its own, independent statistical tracker.
        tracker_class = TRACKER_STRATEGIES[config['salience_strategy']]
        self.tracker = tracker_class(config)
        print(f"[{self.kernel_name}_{os.getpid()}] Initialized with tracker: {config['salience_strategy']}")

    @torch.no_grad()
    def get_latents_for_images(self, images: list[Image.Image]) -> torch.Tensor:
        """Performs inference on a batch of images and returns latents."""
        raise NotImplementedError

    def create_keyframe_event(self, timestamp: float, score: float, reason: str) -> dict:
        """Creates a kernel-specific keyframe event dictionary."""
        return {
            "type": f"{self.kernel_name}_KEYFRAME",
            "timestamp": timestamp, "value": score, "reason": reason
        }

class SigLIPKernel(SalienceKernel):
    """The expert on whole-image visual semantics."""
    def __init__(self, config: dict):
        super().__init__(config)
        self.kernel_name = "VISUAL_SEMANTIC"
        
        model_path = "./models/siglip"
        if not os.path.isdir(model_path): raise FileNotFoundError(f"SigLIP model not found at '{model_path}'.")
        model_kwargs = {"dtype": torch.bfloat16, "attn_implementation": "sdpa"} if self.device.type == 'cuda' else {}
        self.model = AutoModel.from_pretrained(model_path, **model_kwargs).to(self.device)
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.vision_encoder = self.model.vision_model.eval()

    @torch.no_grad()
    def get_latents_for_images(self, images: list[Image.Image]) -> torch.Tensor:
        inputs = self.processor(images=images, return_tensors="pt", padding=True)
        pixel_values = inputs["pixel_values"].to(self.device, dtype=self.model.dtype)
        return self.vision_encoder(pixel_values).pooler_output

class OCRLatentKernel(SalienceKernel):
    """The expert on typographic change and drift."""
    def __init__(self, config: dict):
        super().__init__(config)
        self.kernel_name = "OCR_LATENT"
        
        model_path = "./models/dots_ocr"
        if not os.path.isdir(model_path): raise FileNotFoundError(f"DOTS.ocr model not found at '{model_path}'.")
        model_kwargs = {"attn_implementation": "sdpa", "dtype": torch.bfloat16} if self.device.type == 'cuda' else {}
        self.model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, **model_kwargs).to(self.device)
        self.processor = AutoImageProcessor.from_pretrained(model_path, trust_remote_code=True)
        self.encoder = self.model.vision_tower.to(device=self.device).eval()

    @torch.no_grad()
    def get_latents_for_images(self, images: list[Image.Image]) -> torch.Tensor:
        inputs = self.processor(images=images, return_tensors="pt")
        pixel_values = inputs['pixel_values'].to(self.device, dtype=self.model.dtype)
        grid_thw = inputs['image_grid_thw'].to(self.device)
        return self.encoder(pixel_values, grid_thw=grid_thw).flatten(start_dim=1)

# =====================================================================================
#  NEW ARCHITECTURE: The Search Manager (The "Brain")
# =====================================================================================

class HierarchicalSearchManager:
    """
    Manages the recursive search process, polls kernels for novelty,
    and logs the entire search timeline. It is model-agnostic.
    """
    def __init__(self, kernels: list[SalienceKernel], config: dict):
        self.kernels = kernels
        self.config = config
        self.worker_id = f"SearchManager_{os.getpid()}"
        self.latent_cache = {}

        # --- NEW: Restore the robust default configuration pattern ---
        # 1. Start with a baseline of hard-coded, sensible defaults.
        self.config = self._get_default_search_config()
        # 2. Merge the externally provided config on top of the defaults.
        if config:
            self.config.update(config)

    def _get_default_search_config(self) -> dict:
        """
        Provides a centralized, fallback configuration for the search process.
        """
        print(f"[{self.worker_id}] Loading default search configuration.")
        return {
            "branching_factor": 8,       # How many sentinel frames to check per chunk
            "min_exhaustive_size": 16,   # Chunks smaller than this are always processed fully
            "max_batch_size": 16,        # A safe default for most modern GPUs
        }

    @torch.no_grad()
    def _get_latents_for_indices(self, frames_chunk: list, indices: list[int]) -> dict[str, torch.Tensor]:
        indices_to_compute = [i for i in indices if any((k.kernel_name, i) not in self.latent_cache for k in self.kernels)]
        
        if indices_to_compute:
            for i in range(0, len(indices_to_compute), self.config.get('max_batch_size', 16)):
                batch_indices = indices_to_compute[i : i + self.config.get('max_batch_size', 16)]
                images_to_process = [frames_chunk[idx] for idx in batch_indices]
                
                for kernel in self.kernels:
                    latents_batch = kernel.get_latents_for_images(images_to_process)
                    for idx, latent in zip(batch_indices, latents_batch):
                        self.latent_cache[(kernel.kernel_name, idx)] = latent.to('cpu', non_blocking=True)

        results = {k.kernel_name: [] for k in self.kernels}
        for i in indices:
            for k in self.kernels:
                results[k.kernel_name].append(self.latent_cache[(k.kernel_name, i)])
        
        for name, latents in results.items():
            results[name] = torch.stack(latents).to(self.kernels[0].device)
        return results

    def _exhaustive_scan(self, frames_chunk: list, timestamps_chunk: list, start_idx: int, end_idx: int, search_log: list) -> list:
        """
        Performs a fine-grained scan, creating keyframes for every novel transition.
        """
        span_indices = list(range(start_idx, end_idx + 1))
        if len(span_indices) < 2: return []
        
        all_latents_by_kernel = self._get_latents_for_indices(frames_chunk, span_indices)
        keyframes_found = [] # Local list to hold results

        search_log.append({'type': 'exhaustive_scan', 'span': (timestamps_chunk[start_idx], timestamps_chunk[end_idx]), 'num_frames': len(span_indices)})

        for kernel in self.kernels:
            latents_tensor = all_latents_by_kernel[kernel.kernel_name]
            kernel.tracker.update(latents_tensor) # Update the tracker first

            scores = kernel.tracker.get_novelty_scores(latents_tensor)
            is_novel_mask = kernel.tracker.is_novel(scores)
            
            novel_indices = torch.where(is_novel_mask)[0]
            cpu_scores = scores.detach().cpu().float().numpy()

            for i in novel_indices:
                # The event corresponds to the transition *ending* at this frame
                event_index = start_idx + i.item() + 1
                event = kernel.create_keyframe_event(timestamps_chunk[event_index], float(cpu_scores[i]), "exhaustive_scan_spike")
                keyframes_found.append(event) # Append to our local list
            
        return keyframes_found # Return the list


    def _recursive_search_step(self, frames_chunk: list, timestamps_chunk: list, search_queue: deque, start_idx: int, end_idx: int, depth: int, search_log: list) -> list:
        """
        Performs a coarse-grained scan. It BOTH creates keyframes for novel coarse
        transitions AND schedules deeper searches.
        """
        sentinel_indices = np.linspace(start_idx, end_idx, self.config['branching_factor'], dtype=int)
        all_latents_by_kernel = self._get_latents_for_indices(frames_chunk, sentinel_indices.tolist())

        for kernel in self.kernels:
            kernel.tracker.update(all_latents_by_kernel[kernel.kernel_name])
        
        keyframes_found = [] # Local list to hold results
        found_hotspot = False
        max_score_info = {'score': -1.0, 'span': None, 'reason': None}

        for i in range(len(sentinel_indices) - 1):
            sub_span_start, sub_span_end = int(sentinel_indices[i]), int(sentinel_indices[i+1])
            
            for kernel in self.kernels:
                latents_tensor_slice = all_latents_by_kernel[kernel.kernel_name][[i, i+1]]
                score_tensor = kernel.tracker.get_novelty_scores(latents_tensor_slice)[0]
                is_novel_flag = kernel.tracker.is_novel(score_tensor.unsqueeze(0))[0]

                if is_novel_flag.item():
                    # --- FIX: A surprising coarse transition IS a keyframe ---
                    event_index = sub_span_end
                    event = kernel.create_keyframe_event(timestamps_chunk[event_index], score_tensor.item(), "recursive_search_spike")
                    keyframes_found.append(event)

                    # Also schedule this span for a deeper look
                    search_queue.append((sub_span_start, sub_span_end, depth + 1))
                    search_log.append({
                        'type': 'recursive_descent', 
                        'span_indices': (sub_span_start, sub_span_end), 
                        'reason': f'{kernel.kernel_name}_spike', 
                        'score': score_tensor.item()
                    })
                    found_hotspot = True
                    break # One kernel is enough to flag the span
            
            # --- MODIFIED: Also use .item() for the max score comparison ---
            # Get the scalar value of the last computed score
            current_score_value = score_tensor.item()
            if current_score_value > max_score_info['score']:
                max_score_info.update({
                    'score': current_score_value, 
                    'span': (sub_span_start, sub_span_end), 
                    'reason': f'{kernel.kernel_name}_max_score'
                })

        # If no statistically significant spike was found, descend into the most different sub-span.
        if not found_hotspot and max_score_info['span'] is not None:
            search_queue.append((*max_score_info['span'], depth + 1))
            search_log.append({'type': 'recursive_descent_optimistic', **max_score_info})
        return keyframes_found # Return the list

    def process_chunk(self, frames_chunk: list, timestamps_chunk: list, shutdown_event: Event) -> dict:
        self.latent_cache = {}
        if not frames_chunk: return {'keyframes': [], 'search_log': []}

        # --- NEW: Perform an initial, full-chunk update for the trackers ---
        # This is the most important fix. It "pre-warms" the model on the entire
        # chunk's coarse representation before any scoring begins.
        initial_sentinel_indices = np.linspace(0, len(frames_chunk) - 1, self.config['branching_factor'], dtype=int).tolist()
        initial_latents = self._get_latents_for_indices(frames_chunk, initial_sentinel_indices)
        for kernel in self.kernels:
            kernel.tracker.update(initial_latents[kernel.kernel_name])

        search_queue = deque([(0, len(frames_chunk) - 1, 1)])
        final_events, search_log = [], []

        with tqdm(total=len(frames_chunk), desc=f"[{self.worker_id}] Search", unit="frame", position=0) as pbar:
            while search_queue:
                if shutdown_event.is_set(): break
                start_idx, end_idx, depth = search_queue.popleft()
                
                keyframes_from_step = [] # Temp list to hold results from this step
                
                if (end_idx - start_idx + 1) <= self.config['min_exhaustive_size']:
                    # Call the exhaustive scan and capture its results
                    keyframes_from_step = self._exhaustive_scan(frames_chunk, timestamps_chunk, start_idx, end_idx, search_log)
                    pbar.update(end_idx - start_idx + 1)
                else:
                    # Call the recursive scan and capture its results
                    keyframes_from_step = self._recursive_search_step(frames_chunk, timestamps_chunk, search_queue, start_idx, end_idx, depth, search_log)
                    pbar.update(self.config['branching_factor'])

                # --- THE CRITICAL FIX ---
                # Add the keyframes found in this step to the final list.
                if keyframes_from_step:
                    final_events.extend(keyframes_from_step)

        pbar.update(pbar.total - pbar.n)
        return {'keyframes': final_events, 'search_log': search_log}


# =====================================================================================
#  NEW: The Kernel Factory (Registry)
# =====================================================================================


# This dictionary maps the string names used in the config to the actual kernel classes.
# This is the single point of truth for what kernels are available.

KERNEL_REGISTRY = {
    "siglip": SigLIPKernel,
    "ocr": OCRLatentKernel,
}


# =====================================================================================
#  The Main Worker Loop (Now much simpler)
# =====================================================================================

def analysis_worker_loop(task_queue, return_queue, config, output_path, shutdown_event):
    worker_pid = os.getpid()
    print(f"[{worker_pid}] Analysis worker started.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config['device']=device

    # --- Instantiate all enabled kernels based on config ---
    kernels_to_load = config.get("salience_kernels", ["siglip"])
    kernels = []
    print(f"[{worker_pid}] Loading kernels: {kernels_to_load}")
    if "siglip" in kernels_to_load:
        kernels.append(SigLIPKernel(config))
    if "ocr" in kernels_to_load:
        kernels.append(OCRLatentKernel(config))
    
    search_manager = HierarchicalSearchManager(kernels, config)

    while True:
        try:
            if shutdown_event.is_set(): break
            task = task_queue.get(timeout=0.1)
            if task is None: break

            shm = shared_memory.SharedMemory(name=task['shm_name'])
            buffer = np.ndarray(task['shape'], dtype=task['dtype'], buffer=shm.buf)
            
            frames_as_images = [Image.fromarray(frame) for frame in buffer]
            downscaled_images = [_downscale_image(img) for img in frames_as_images]
            
            result_data = search_manager.process_chunk(downscaled_images, task['timestamps'], shutdown_event)
            
            return_message = {
                "shm_name": task['shm_name'], "shape": task['shape'],
                "dtype": task['dtype'], "timestamps": task['timestamps'],
                "source": "HierarchicalSearchManager", "data": result_data
            }
            return_queue.put(return_message)
            shm.close()

        except queue.Empty:
            continue
        except Exception as e:
            print(f"[{worker_pid}] CRITICAL ERROR in worker loop: {e}")
            import traceback; traceback.print_exc()
            
    print(f"[{worker_pid}] Analysis worker finished.")

