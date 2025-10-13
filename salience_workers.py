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
import gc
from typing import Callable

# --- NEW: Import our new tracker classes ---
from salience_trackers import NaiveZScoreTracker, PCAMahanalobisTracker, TRACKER_STRATEGIES

class LazyFrameAccessor:
    """
        Wraps the SHM buffer and preprocesses frames only when accessed.
        Args:
            shm_name (str): The name of the shared memory block.
            shape (tuple): The shape of the numpy array in SHM.
            dtype (np.dtype): The data type of the numpy array.
            preprocess_fn (Callable): A function that takes a raw numpy frame 
                                      and returns a processed PIL Image.
    """
    def __init__(self, shm_name: str, shape: tuple, preprocess_fn: Callable[[np.ndarray], Image.Image], dtype: np.dtype):
        self.shm = shared_memory.SharedMemory(name=shm_name)
        self.buffer = np.ndarray(shape, dtype=dtype, buffer=self.shm.buf)
        self.preprocess_fn = preprocess_fn
        # Cache preprocessed images to avoid redundant work within a single chunk
        self.cache = {}
        self.num_frames = shape[0]

    # --- NEW METHOD ---
    def __len__(self) -> int:
        """Returns the total number of frames in the buffer."""
        return self.num_frames

    def __getitem__(self, index: int) -> Image.Image:
        """Retrieves and preprocesses a single frame, using a cache."""
        if index in self.cache:
            return self.cache[index]
        
        # Get the raw frame from the shared buffer
        raw_frame = self.buffer[index]
        
        # Preprocess on-demand using the provided function
        processed_frame = self.preprocess_fn(raw_frame)
        
        # Cache the result for future access
        self.cache[index] = processed_frame
        return processed_frame

    def get_batch(self, indices: list[int]) -> list[Image.Image]:
        return [self[i] for i in indices]

    def close(self):
        self.shm.close()

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
       #print(f"[{self.kernel_name}_{os.getpid()}] Initialized with tracker: {config['salience_strategy']}")

    @torch.no_grad()
    def get_latents_for_images(self, images: list[Image.Image]) -> torch.Tensor:
        """Performs inference on a batch of images and returns latents."""
        raise NotImplementedError

    def create_keyframe_event(self, timestamp: float, score: float, reason: str, z_score: float) -> dict:
        """Creates a kernel-specific keyframe event dictionary."""
        return {
            "type": f"{self.kernel_name}_KEYFRAME",
            "timestamp": timestamp, "value": score,
            "z_score": z_score,    # This is the new, crucial metadata
            "reason": reason
        }

class SigLIPKernel(SalienceKernel):
    """The expert on whole-image visual semantics."""
    def __init__(self, config: dict):
        super().__init__(config)
        self.kernel_name = "SIGLIP"
        
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
        self.kernel_name = "OCR"
        
        model_path = "./models/dots_ocr"
        model_kwargs = {"attn_implementation": "sdpa", "dtype": torch.bfloat16} if self.device.type == 'cuda' else {}
        
        # --- THE "LOAD AND PLUCK" STRATEGY ---

        # 1. Load the full, memory-intensive Causal LM into VRAM temporarily.
        #    This is the step that allocates the ~36GB.
       #print(f"[{self.kernel_name}] Temporarily loading full Causal LM to extract vision tower...")
        full_model = AutoModelForCausalLM.from_pretrained(
            model_path, 
            trust_remote_code=True, 
            **model_kwargs
        ).to(self.device)

        # 2. Immediately extract the vision tower module. This is just a reference.
        #    The actual weights are still part of the `full_model` graph in VRAM.
       #print(f"[{self.kernel_name}] Plucking the vision tower...")
        self.encoder = full_model.vision_tower

        # 3. CRITICAL: Delete the reference to the full model.
       #print(f"[{self.kernel_name}] Deleting reference to the full model...")
        del full_model

        # 4. CRITICAL: Force Python's garbage collector and PyTorch's cache to run.
        #    This tells PyTorch to release the gigabytes of VRAM that were part of the
        #    language model, as they are no longer referenced by any object.
       #print(f"[{self.kernel_name}] Clearing garbage and CUDA cache...")
        gc.collect()
        torch.cuda.empty_cache()

        # By this point, only the vision_tower's weights should remain in VRAM.
       #print(f"[{self.kernel_name}] Vision tower extracted. VRAM should now be freed.")
        self.encoder.eval() # Ensure it's in eval mode

        # The processor is loaded as usual
        self.processor = AutoImageProcessor.from_pretrained(model_path, trust_remote_code=True)

    @torch.no_grad()
    def get_latents_for_images(self, images: list[Image.Image]) -> torch.Tensor:
        # 1. Get inputs from the processor
        inputs = self.processor(images=images, return_tensors="pt")
        pixel_values = inputs['pixel_values'].to(self.device, dtype=self.encoder.dtype)
        grid_thw = inputs['image_grid_thw'].to(self.device)

        # 2. Get the final embeddings from the vision tower.
        # Shape is [total_merged_tokens_in_batch, hidden_size]
        final_embeddings = self.encoder(hidden_states=pixel_values, grid_thw=grid_thw)
        
        # --- THE FIX: Structured Pooling ---
        # 3. We need to average the tokens that belong to each image separately.
        
        # Get the spatial merge size from the model's config
        spatial_merge_size = self.encoder.config.spatial_merge_size
        
        # Calculate how many merged tokens each image in the batch produces.
        # grid_thw[:, 1] is height, grid_thw[:, 2] is width.
        merged_h = grid_thw[:, 1] // spatial_merge_size
        merged_w = grid_thw[:, 2] // spatial_merge_size
        tokens_per_image = (merged_h * merged_w).tolist()

        # 4. Split the embedding tensor into chunks, one for each image.
        image_embedding_chunks = torch.split(final_embeddings, tokens_per_image, dim=0)
        
        # 5. Average the tokens within each chunk to get a single vector per image.
        #    We average across dim=0 of each chunk.
        averaged_chunks = [chunk.mean(dim=0) for chunk in image_embedding_chunks]
        
        # 6. Stack the averaged vectors back into a single 2D tensor.
        #    The final shape is [batch_size, hidden_size], which is what PCA expects.
        latent_vectors = torch.stack(averaged_chunks, dim=0)
        
        return latent_vectors

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

        # This will be reset for each chunk and used for deferred learning.
        self.latents_from_search = {k.kernel_name: [] for k in self.kernels}
        self.latent_cache = {} # Caches latents within a single chunk's search

        self.siglip_stream = torch.cuda.Stream()
        self.ocr_stream = torch.cuda.Stream()
        # A map to make it easy to look up
        self.kernel_streams = {"SIGLIP": self.siglip_stream, "OCR": self.ocr_stream}

    def _get_default_search_config(self) -> dict:
        """
        Provides a centralized, fallback configuration for the search process.
        """
       #print(f"[{self.worker_id}] Loading default search configuration.")
        return {
            "branching_factor": 8,       # How many sentinel frames to check per chunk
            "min_exhaustive_size": 16,   # Chunks smaller than this are always processed fully
            "max_batch_size": 16,        # A safe default for most modern GPUs
        }

    @torch.no_grad()
    def _get_latents_for_indices(self, frame_accessor: LazyFrameAccessor, indices: list[int]) -> dict[str, torch.Tensor]:
        # This method needs a significant rewrite to handle per-kernel batching.

        # 1. Figure out what needs computing for each kernel
        work_items_by_kernel = {k.kernel_name: [] for k in self.kernels}
        for i in indices:
            for k in self.kernels:
                if (k.kernel_name, i) not in self.latent_cache:
                    work_items_by_kernel[k.kernel_name].append(i)

        # 2. Process each kernel's work items with its own batch size
        for kernel in self.kernels:
            kernel_name = kernel.kernel_name
            stream = self.kernel_streams.get(kernel_name) # Get the specific stream
            indices_to_compute = list(dict.fromkeys(work_items_by_kernel[kernel_name]))
            if not indices_to_compute:
                continue

            # --- THE CRITICAL CHANGE ---
            # Get the batch size specific to this kernel, with a fallback to a global default.
            kernel_conf = self.config.get("kernel_configs", {}).get(kernel.kernel_name.lower(), {})
            batch_size = kernel_conf.get('max_batch_size', 1) # Default to 1 for safety
            #print(f"[{self.worker_id}] Processing for {kernel_name} with batch size {batch_size}")

            for i in range(0, len(indices_to_compute), batch_size):
                # Tell PyTorch to run this block of work on the kernel's dedicated stream
                with torch.cuda.stream(stream):
                    batch_indices = indices_to_compute[i : i + batch_size]
                    images_to_process = frame_accessor.get_batch(batch_indices)
                    
                    latents_batch = kernel.get_latents_for_images(images_to_process)
                    for idx, latent in zip(batch_indices, latents_batch):
                        # The cache put happens within the stream context
                        self.latent_cache[(kernel_name, idx)] = latent.to('cpu', non_blocking=True)

        # After launching all work, you need to synchronize
        torch.cuda.synchronize() # Wait for all streams to finish before assembling results

        # 3. Assemble results (this part remains the same)
        results = {k.kernel_name: [] for k in self.kernels}
        for i in indices:
            for k in self.kernels:
                results[k.kernel_name].append(self.latent_cache[(k.kernel_name, i)])
        
        for name, latents in results.items():
            results[name] = torch.stack(latents).to(self.kernels[0].device)
        return results

    def _recursive_search_step(self, frames_chunk: list, timestamps_chunk: list, search_queue: deque, start_idx: int, end_idx: int, depth: int, is_hot: bool, search_log: list) -> tuple[list, int]:
        """
        Performs a single, sparse probe of a span and queues further work based on adaptive parameters.
        Returns a tuple of (keyframes_found_in_this_step, frames_processed_in_this_step).
        """
        # --- 1. Determine search parameters based on "hotness" (This part is fine) ---
        if is_hot:
            k_to_use = self.config.get("top_k", 2)
            max_d_to_use = self.config.get("max_d", 3)
        else: # "Cold" or "Lazy" search
            k_to_use = self.config.get("top_k_lazy", 1)
            max_d_to_use = self.config.get("max_d_lazy", 2)

        # --- 2. Termination checks for this branch (This part is fine) ---
        if depth >= max_d_to_use:
            search_log.append({'type': 'terminate_branch', 'reason': 'max_depth_reached', 'depth': depth})
            return [], 0

        span_size = end_idx - start_idx + 1
        branching_factor = self.config['branching_factor']
        if span_size <= branching_factor:
            search_log.append({'type': 'terminate_branch', 'reason': 'span_too_small', 'size': span_size})
            return [], 0
        
        # --- 3. Perform the sparse probe ---
        sentinel_indices = np.linspace(start_idx, end_idx, branching_factor, dtype=int).tolist()
        all_latents_by_kernel = self._get_latents_for_indices(frames_chunk, sentinel_indices)

        # Add all computed latents to our cache for the final batch update
        for kernel_name, latents_tensor in all_latents_by_kernel.items():
            self.latents_from_search[kernel_name].append(latents_tensor)
        
        frames_processed_this_step = len(sentinel_indices)
        keyframes_found = []
        sub_spans_to_consider = []

        # --- 4. Evaluate sub-spans and prepare for ranking ---
        for i in range(len(sentinel_indices) - 1):
            sub_span_start, sub_span_end = int(sentinel_indices[i]), int(sentinel_indices[i+1])
            is_sub_span_hot = False
            highest_score_in_sub_span = -1.0

            for kernel in self.kernels:
                latents_slice = all_latents_by_kernel[kernel.kernel_name][[i, i+1]]
                scores_tensor = kernel.tracker.get_novelty_scores(latents_slice)
                is_novel_mask, z_scores_tensor = kernel.tracker.is_novel(scores_tensor)

                if is_novel_mask.any().item():
                    is_sub_span_hot = True
                    max_score_val = scores_tensor.max().item()
                    event_idx = sub_span_start if scores_tensor.argmax().item() == 0 else sub_span_end
                    # --- ADD: Extract the corresponding Z-score ---
                    z_score_val = z_scores_tensor[event_idx_in_batch].item()
                    event = kernel.create_keyframe_event(timestamps_chunk[event_idx],
                        max_score_val,
                        "sparse_search_spike", 
                        z_score=z_score_val)
                    keyframes_found.append(event)
                
                highest_score_in_sub_span = max(highest_score_in_sub_span, scores_tensor.max().item())

            sub_spans_to_consider.append({
                'score': highest_score_in_sub_span,
                'start': sub_span_start,
                'end': sub_span_end,
                'is_hot': is_sub_span_hot # Carry the "hot" status forward
                
            })
        
            sub_spans_to_consider.sort(key=lambda x: x['score'], reverse=True)
        
        any_hot_spans_found = any(s['is_hot'] for s in sub_spans_to_consider)
        
        if any_hot_spans_found:
            # --- Standard Path: At least one sub-span is interesting ---
            # Explore the top `k_to_use` spans.
            for i in range(min(k_to_use, len(sub_spans_to_consider))):
                span_to_queue = sub_spans_to_consider[i]
                search_queue.append((span_to_queue['start'], span_to_queue['end'], depth + 1, span_to_queue['is_hot']))
                search_log.append({'type': 'queue_sub_span', 'depth': depth + 1, 'is_hot': span_to_queue['is_hot'], 'score': span_to_queue['score']})
        else:
            # --- "Pity Probe" Path: Nothing was hot, so we perform a LAZY search ---
            # We use the _lazy parameters to ensure we gather some data without committing to a deep search.
            lazy_k = self.config.get("top_k_lazy", 1)
            lazy_max_d = self.config.get("max_d_lazy", 1)
            
            # We check the depth against the lazy depth limit.
            if depth < lazy_max_d:
                for i in range(min(lazy_k, len(sub_spans_to_consider))):
                    span_to_queue = sub_spans_to_consider[i]
                    # We queue the highest-scoring cold span and explicitly mark its children as `is_hot=False`.
                    search_queue.append((span_to_queue['start'], span_to_queue['end'], depth + 1, False))
                    search_log.append({'type': 'queue_lazy_probe', 'depth': depth + 1, 'score': span_to_queue['score']})

        return keyframes_found, frames_processed_this_step

    def process_chunk(self, frames_chunk: list, timestamps_chunk: list, shutdown_event: Event) -> dict:
        # Reset state for the new chunk
        self.latent_cache.clear()
        self.latents_from_search = {k.kernel_name: [] for k in self.kernels}
        if len(frames_chunk) == 0: return {'keyframes': [], 'search_log': []}

        # --- 1. Setup search budget and initial state ---
        processing_budget = int(len(frames_chunk) * self.config.get("max_p", 0.33))
        frames_processed = 0
        final_events, search_log = [], []

        # The queue tracks: (start, end, depth, is_hot). The root is always considered "hot".
        search_queue = deque([(0, len(frames_chunk) - 1, 0, True)])
        
        pbar_desc = f"[{self.worker_id}] Sparse Search (Budget: {processing_budget} frames)"
        with tqdm(total=processing_budget, desc=pbar_desc, unit="frame") as pbar:
            while search_queue:
                if shutdown_event.is_set() or frames_processed >= processing_budget:
                    if frames_processed >= processing_budget:
                        search_log.append({'type': 'terminate_search', 'reason': 'budget_exceeded'})
                    break
                
                start_idx, end_idx, depth, is_hot = search_queue.popleft()

                # --- 2. Execute one step of the recursive search ---
                keyframes_from_step, processed_this_step = self._recursive_search_step(
                    frames_chunk, timestamps_chunk, search_queue,
                    start_idx, end_idx, depth, is_hot, search_log
                )
                
                if keyframes_from_step:
                    final_events.extend(keyframes_from_step)
                
                frames_processed += processed_this_step
                pbar.update(processed_this_step)

        # --- 3. Deferred Batch Learning: Train the model on everything we saw ---
        search_log.append({'type': 'deferred_learning', 'total_latents': sum(len(l) for l in self.latents_from_search.values())})
        for kernel in self.kernels:
            kernel_name = kernel.kernel_name
            if self.latents_from_search[kernel_name]:
                all_latents_tensor = torch.cat(self.latents_from_search[kernel_name], dim=0)
                # The update call now trains the PCA model AND the score statistics
                kernel.tracker.update(all_latents_tensor)
        
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
   #print(f"[{worker_pid}] Analysis worker started.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config['device']=device

    # --- Instantiate all enabled kernels based on config ---
    kernels_to_load = config.get("salience_kernels", ["siglip"])
    kernels = []
   #print(f"[{worker_pid}] Loading kernels: {kernels_to_load}")
    if "siglip" in kernels_to_load:
        kernels.append(SigLIPKernel(config))
    if "ocr" in kernels_to_load:
        kernels.append(OCRLatentKernel(config))
    
    search_manager = HierarchicalSearchManager(kernels, config)
    preprocessor = lambda frame: _downscale_image(Image.fromarray(frame))

    while True:
        try:
            if shutdown_event.is_set(): break
            task = task_queue.get(timeout=0.1)
            if task is None: break

            #shm = shared_memory.SharedMemory(name=task['shm_name'])
            #buffer = np.ndarray(task['shape'], dtype=task['dtype'], buffer=shm.buf)
            # doubled copy buffers? for no reason?
            #frames_as_images = [Image.fromarray(frame) for frame in buffer]
            #downscaled_images = [_downscale_image(img) for img in frames_as_images]

            # --- LAZY INITIALIZATION ---
            # Instantiate the accessor, injecting our defined preprocessor.
            frame_accessor = LazyFrameAccessor(shm_name=task['shm_name'],
                shape=task['shape'],
                dtype=task['dtype'],
                preprocess_fn=preprocessor)
            
            # The search manager now operates on the accessor, triggering
            # lazy preprocessing for only the frames it needs.
            result_data = search_manager.process_chunk(frame_accessor, task['timestamps'], shutdown_event)
            
            return_message = {
                "shm_name": task['shm_name'], "shape": task['shape'],
                "dtype": task['dtype'], "timestamps": task['timestamps'],
                "source": "HierarchicalSearchManager", "data": result_data
            }
            print(f"repr:{repr(result_data)}")
            return_queue.put(return_message)
            #shm.close()
            frame_accessor.close()

        except queue.Empty:
            continue
        except Exception as e:
            print(f"[{worker_pid}] CRITICAL ERROR in worker loop: {e}")
            import traceback; traceback.print_exc()
            if 'task' in locals() and task:
                failure_message = {
                    "shm_name": task['shm_name'],
                    "shape": task.get('shape'), # Include what we can
                    "dtype": task.get('dtype'),
                    "timestamps": task.get('timestamps'),
                    "source": "HierarchicalSearchManager",
                    "data": {"error": str(e)} # Signal that this was a failure
                }
                return_queue.put(failure_message)
                
                # Ensure we don't hold a dangling reference to the failed SHM
                if 'shm' in locals() and shm:
                    shm.close()
            
   #print(f"[{worker_pid}] Analysis worker finished.")

