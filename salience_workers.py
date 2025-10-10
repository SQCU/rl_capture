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
    def __init__(self, config: dict, device):
        self.kernel_name = "BaseKernel"
        self.device = device
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
    def __init__(self, config: dict, device):
        super().__init__(config, device)
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
    def __init__(self, config: dict, device):
        super().__init__(config, device)
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
        span_indices = list(range(start_idx, end_idx + 1))
        if len(span_indices) < 2: return []
        
        all_latents = self._get_latents_for_indices(frames_chunk, span_indices)
        all_keyframes = []

        search_log.append({'type': 'exhaustive_scan', 'span': (timestamps_chunk[start_idx], timestamps_chunk[end_idx]), 'num_frames': len(span_indices)})

        for kernel in self.kernels:
            latents_np = all_latents[kernel.kernel_name].cpu().numpy()
            scores = kernel.tracker.get_novelty_scores(latents_np)
            is_novel_mask = kernel.tracker.is_novel(scores)
            
            novel_indices = np.where(is_novel_mask)[0]
            for i in novel_indices:
                event_index = start_idx + i
                event = kernel.create_keyframe_event(timestamps_chunk[event_index], float(scores[i]), "exhaustive_scan_spike")
                all_keyframes.append(event)
            
            kernel.tracker.update(latents_np)
            
        return all_keyframes

    def _recursive_search_step(self, frames_chunk: list, search_queue: deque, start_idx: int, end_idx: int, depth: int, search_log: list):
        sentinel_indices = np.linspace(start_idx, end_idx, self.config.get('branching_factor', 8), dtype=int)
        all_latents = self._get_latents_for_indices(frames_chunk, sentinel_indices.tolist())

        found_hotspot = False
        max_score_info = {'score': -1, 'span': None, 'reason': None}

        for i in range(len(sentinel_indices) - 1):
            sub_span_start, sub_span_end = int(sentinel_indices[i]), int(sentinel_indices[i+1])
            is_interesting_span = False

            for kernel in self.kernels:
                latents_np = all_latents[kernel.kernel_name][[i, i+1]].cpu().numpy()
                score = kernel.tracker.get_novelty_scores(latents_np)[0]
                
                if kernel.tracker.is_novel(np.array([score]))[0]:
                    is_interesting_span = True
                    search_log.append({'type': 'recursive_descent', 'span_indices': (sub_span_start, sub_span_end), 'reason': f'{kernel.kernel_name}_spike', 'score': float(score)})
                    break # One kernel is enough to flag the span
            
            if is_interesting_span:
                search_queue.append((sub_span_start, sub_span_end, depth + 1))
                found_hotspot = True
            
            # Keep track of the most different span, regardless of statistical novelty
            if score > max_score_info['score']:
                max_score_info.update({'score': score, 'span': (sub_span_start, sub_span_end), 'reason': f'{kernel.kernel_name}_max_score'})

        if not found_hotspot and max_score_info['span'] is not None:
            search_queue.append((*max_score_info['span'], depth + 1))
            search_log.append({'type': 'recursive_descent_optimistic', 'span_indices': max_score_info['span'], **max_score_info})

    def process_chunk(self, frames_chunk: list, timestamps_chunk: list, shutdown_event: Event) -> dict:
        self.latent_cache = {}
        if not frames_chunk: return {'keyframes': [], 'search_log': []}

        search_queue = deque([(0, len(frames_chunk) - 1, 1)])
        final_events, search_log = [], []

        with tqdm(total=len(frames_chunk), desc=f"[{self.worker_id}] Search", unit="frame", position=0) as pbar:
            while search_queue:
                if shutdown_event.is_set(): break
                start_idx, end_idx, depth = search_queue.popleft()
                
                if (end_idx - start_idx + 1) <= self.config.get('min_exhaustive_size', 16):
                    keyframes = self._exhaustive_scan(frames_chunk, timestamps_chunk, start_idx, end_idx, search_log)
                    final_events.extend(keyframes)
                    pbar.update(end_idx - start_idx + 1)
                else:
                    self._recursive_search_step(frames_chunk, search_queue, start_idx, end_idx, depth, search_log)
        
        return {'keyframes': final_events, 'search_log': search_log}

# =====================================================================================
#  The Main Worker Loop (Now much simpler)
# =====================================================================================

def analysis_worker_loop(task_queue, return_queue, config, output_path, shutdown_event):
    worker_pid = os.getpid()
    print(f"[{worker_pid}] Analysis worker started.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Instantiate all enabled kernels based on config ---
    kernels_to_load = config.get("salience_kernels", ["siglip"])
    kernels = []
    print(f"[{worker_pid}] Loading kernels: {kernels_to_load}")
    if "siglip" in kernels_to_load:
        kernels.append(SigLIPKernel(config, device))
    if "ocr" in kernels_to_load:
        kernels.append(OCRLatentKernel(config, device))
    
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