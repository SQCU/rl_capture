# salience_workers.py

import numpy as np
import torch
import torch.nn.functional as F
from torch.backends.cuda import sdp_kernel, SDPBackend
import os
import time  # --- NEW ---
from tqdm import tqdm # --- NEW ---
from multiprocessing import shared_memory, Event, Queue
from transformers import AutoModel, AutoProcessor, AutoModelForCausalLM, AutoImageProcessor
from PIL import Image
from collections import deque
import queue # --- NEW ---
import gc
from typing import Callable

# --- NEW: Import our new tracker classes ---
from hyperparameter_schedulers import ConfigScheduler
from salience_trackers import NaiveZScoreTracker, PCAMahanalobisTracker, TRACKER_STRATEGIES

def log_attention_backend_status(worker_pid):
    """Checks and logs the status of the selected PyTorch attention backend."""
    if not torch.cuda.is_available():
        print(f"[{worker_pid}] CUDA not available. Using default CPU attention backend.")
        return

    # Check which backend was successfully enabled by the sdp_kernel context
    if torch.backends.cuda.flash_sdp_enabled():
        print(f"[{worker_pid}] ✅ PyTorch attention backend set to FLASH ATTENTION (Fastest)")
    elif torch.backends.cuda.mem_efficient_sdp_enabled():
        print(f"[{worker_pid}] ✅ PyTorch attention backend set to MEMORY-EFFICIENT (xFormers/Native)")
    else:
        print(f"[{worker_pid}] ⚠️ PyTorch attention backend fell back to MATH (Eager/Slowest)")


# =============================================================================
#  GPU-ACCELERATED FRAME PREPROCESSING
# =============================================================================

class GPUFramePreprocessor:
    """
    GPU-accelerated frame preprocessing. Avoids PIL entirely.

    Flow: numpy (HWC uint8) -> pinned tensor -> GPU -> resize -> normalize -> model

    This is MUCH faster than the CPU path:
      CPU: numpy -> PIL.Image -> PIL.resize(LANCZOS) -> processor -> tensor -> .to(GPU)
      GPU: numpy -> pinned tensor -> GPU -> F.interpolate -> normalize (all on GPU)
    """

    # ImageNet normalization (used by SigLIP and most vision models)
    IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406])
    IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225])

    def __init__(self, device: torch.device, target_size: tuple[int, int] = (384, 384),
                 dtype: torch.dtype = torch.bfloat16):
        """
        Args:
            device: Target GPU device
            target_size: (H, W) output size for model input
            dtype: Model dtype (bfloat16 for most modern models)
        """
        self.device = device
        self.target_size = target_size
        self.dtype = dtype

        # Move normalization constants to GPU
        self.mean = self.IMAGENET_MEAN.view(1, 3, 1, 1).to(device, dtype=torch.float32)
        self.std = self.IMAGENET_STD.view(1, 3, 1, 1).to(device, dtype=torch.float32)

        # Pinned memory buffer for async CPU->GPU transfer (reused across batches)
        self._pinned_buffer = None
        self._pinned_buffer_size = 0

    def _ensure_pinned_buffer(self, shape: tuple) -> torch.Tensor:
        """Get or create a pinned memory buffer of sufficient size."""
        needed_size = int(np.prod(shape))
        if self._pinned_buffer is None or self._pinned_buffer_size < needed_size:
            # Allocate new pinned buffer (this is expensive, so we reuse)
            self._pinned_buffer = torch.empty(
                needed_size, dtype=torch.uint8, pin_memory=True
            )
            self._pinned_buffer_size = needed_size
        return self._pinned_buffer[:needed_size].view(shape)

    @torch.no_grad()
    def preprocess_batch(self, frames: np.ndarray) -> torch.Tensor:
        """
        Preprocess a batch of frames entirely on GPU.

        Args:
            frames: numpy array of shape (N, H, W, C) uint8 RGB

        Returns:
            Tensor of shape (N, C, target_H, target_W) normalized and ready for model
        """
        if frames.ndim == 3:
            frames = frames[np.newaxis, ...]  # Add batch dim

        N, H, W, C = frames.shape

        # Step 1: Copy to pinned memory (enables async transfer)
        pinned = self._ensure_pinned_buffer((N, H, W, C))
        pinned.copy_(torch.from_numpy(frames))

        # Step 2: Async transfer to GPU + convert to float [0, 1]
        gpu_tensor = pinned.to(self.device, non_blocking=True).float() / 255.0

        # Step 3: Reorder HWC -> CHW (vision model format)
        gpu_tensor = gpu_tensor.permute(0, 3, 1, 2)  # (N, C, H, W)

        # Step 4: Resize using GPU-accelerated bilinear interpolation
        # (bicubic is closer to LANCZOS but bilinear is faster and good enough)
        if (H, W) != self.target_size:
            gpu_tensor = F.interpolate(
                gpu_tensor,
                size=self.target_size,
                mode='bilinear',
                align_corners=False,
                antialias=True  # Important for downscaling quality
            )

        # Step 5: Normalize with ImageNet stats
        gpu_tensor = (gpu_tensor - self.mean) / self.std

        # Step 6: Convert to model dtype
        return gpu_tensor.to(self.dtype)

    def preprocess_single(self, frame: np.ndarray) -> torch.Tensor:
        """Convenience method for single frame."""
        return self.preprocess_batch(frame[np.newaxis, ...])[0]


# For models that need different normalization (e.g., CLIP uses different values)
class SigLIPPreprocessor(GPUFramePreprocessor):
    """SigLIP-specific preprocessor with correct normalization."""
    # SigLIP uses these values (from the processor config)
    SIGLIP_MEAN = torch.tensor([0.5, 0.5, 0.5])
    SIGLIP_STD = torch.tensor([0.5, 0.5, 0.5])

    def __init__(self, device: torch.device, target_size: tuple[int, int], dtype: torch.dtype = torch.bfloat16):
        """
        Args:
            device: GPU device
            target_size: (H, W) from processor config - MUST match model's expected size
            dtype: Model dtype
        """
        super().__init__(device, target_size=target_size, dtype=dtype)
        # Override with SigLIP normalization
        self.mean = self.SIGLIP_MEAN.view(1, 3, 1, 1).to(device, dtype=torch.float32)
        self.std = self.SIGLIP_STD.view(1, 3, 1, 1).to(device, dtype=torch.float32)

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
        """Get batch of preprocessed PIL images (legacy CPU path)."""
        return [self[i] for i in indices]

    def get_raw_batch(self, indices: list[int]) -> np.ndarray:
        """
        Get batch of raw numpy frames (for GPU preprocessing path).

        Returns:
            numpy array of shape (N, H, W, C) uint8
        """
        return self.buffer[indices]  # Direct numpy slicing - very fast

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

        # GPU-accelerated preprocessing (skips PIL entirely)
        if self.device.type == 'cuda':
            # Get correct image size from processor config
            try:
                img_size = self.processor.image_processor.size
                if isinstance(img_size, dict):
                    # Handle {"height": H, "width": W} format
                    target_size = (img_size.get('height', 384), img_size.get('width', 384))
                elif isinstance(img_size, int):
                    target_size = (img_size, img_size)
                else:
                    target_size = (384, 384)  # fallback
                print(f"[SigLIPKernel] GPU preprocessor target size: {target_size}")
            except Exception as e:
                print(f"[SigLIPKernel] Could not detect image size, using 384x384: {e}")
                target_size = (384, 384)

            self.gpu_preprocessor = SigLIPPreprocessor(self.device, target_size=target_size, dtype=self.model.dtype)
        else:
            self.gpu_preprocessor = None

    @torch.no_grad()
    def get_latents_for_frames(self, frames: np.ndarray) -> torch.Tensor:
        """
        GPU-accelerated path: numpy frames -> GPU preprocess -> model.

        Args:
            frames: numpy array of shape (N, H, W, C) uint8 RGB

        Returns:
            Latent tensor of shape (N, latent_dim)
        """
        if self.gpu_preprocessor is not None:
            # Fast GPU path: no PIL, no CPU preprocessing
            pixel_values = self.gpu_preprocessor.preprocess_batch(frames)
            return self.vision_encoder(pixel_values).pooler_output
        else:
            # Fallback to CPU path via PIL
            images = [Image.fromarray(f) for f in frames]
            return self.get_latents_for_images(images)

    @torch.no_grad()
    def get_latents_for_images(self, images: list[Image.Image]) -> torch.Tensor:
        """Legacy CPU path using HuggingFace processor (slower)."""
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
#  NEWER ARCHITECTURE: DECOUPLED FRAME PARSING 'KERNELS'
# =====================================================================================

import heapq

class EventContext:
    def __init__(self):
        self.priority_events = []
        # --- NEW: Add a counter for tie-breaking ---
        self.counter = 0
    def publish(self, z_score: float, event: dict):
        # --- MODIFIED: Push a 3-element tuple ---
        # The tuple is now (-z_score, count, event).
        # If z-scores are equal, Python will compare the counts, which are always unique.
        # The dictionaries will never be compared.
        heapq.heappush(self.priority_events, (-z_score, self.counter, event))
        self.counter += 1
    def get_new_events_for_agent(self, agent_last_seen_count: int) -> list[dict]:
        """Returns new events since the agent last checked."""
        # We now return the third element (index 2) of the stored tuple
        return [item[2] for item in self.priority_events[agent_last_seen_count:]]

class SearchAgent:
    """An agent that launches GPU work WITHOUT BLOCKING."""
    def __init__(self, kernel, agent_config, frame_accessor, timestamps, event_context, stream):
        self.kernel = kernel
        self.config = agent_config
        self.frame_accessor = frame_accessor
        self.timestamps = np.array(timestamps)
        self.event_context = event_context
        self.stream = stream
        self.work_queue = deque([(0, len(frame_accessor) - 1, 0, True)])
        self.computed_latents_for_update = []
        self.pending_results = deque()
        self.last_seen_event_count = 0
        self.cross_pollination_window = self.config.get("cross_pollination_window_seconds", 1.0)
        # --- NEW: Give each agent its own budget and counter ---
        self.search_log = [] # Give each agent a log
        self.budget = int(len(frame_accessor) * self.config.get("max_p", 0.33))
        self.frames_processed = 0

    def has_work(self) -> bool:
        return bool(self.work_queue) or bool(self.pending_results)
    
    def _poll_event_context(self):
        new_events = self.event_context.get_new_events_for_agent(self.last_seen_event_count)
        self.last_seen_event_count = len(self.event_context.priority_events)
        if not new_events: return
        for event in new_events:
            if event['source_kernel'] != self.kernel.kernel_name:
                start_time = event['timestamp'] - self.cross_pollination_window
                end_time = event['timestamp'] + self.cross_pollination_window
                start_idx = np.searchsorted(self.timestamps, start_time, side='left')
                end_idx = np.searchsorted(self.timestamps, end_time, side='right')
                start_idx, end_idx = max(0, start_idx), min(len(self.timestamps) - 1, end_idx)
                if start_idx < end_idx: self.work_queue.appendleft((start_idx, end_idx, 0, True))
    
    def launch_work(self):
        if self.frames_processed >= self.budget:
            # Budget exhausted, clear the remaining work queue to stop the search.
            self.work_queue.clear()
            return False 
        if len(self.pending_results) >= 4: return False
        self._poll_event_context()
        if not self.work_queue: return False
        
        start_idx, end_idx, depth, is_hot = self.work_queue.popleft()
        branching_factor = self.config['branching_factor']
        if is_hot:
            k_to_use, max_d_to_use = self.config.get("top_k", 2), self.config.get("max_d", 3)
        else:
            k_to_use, max_d_to_use = self.config.get("top_k_lazy", 1), self.config.get("max_d_lazy", 2)
        
        if depth >= max_d_to_use: 
            self.search_log.append({'type': 'terminate_branch', 'reason': 'max_depth/size', 'depth': depth})
            return False
        
        sentinel_indices = np.linspace(start_idx, end_idx, branching_factor, dtype=int).tolist()
        self.frames_processed += len(sentinel_indices)

        with torch.cuda.stream(self.stream):
            # --- STAGE 1: GPU WORK ---
            # Use GPU preprocessing path if available (much faster)
            if hasattr(self.kernel, 'get_latents_for_frames') and hasattr(self.kernel, 'gpu_preprocessor') and self.kernel.gpu_preprocessor is not None:
                # Fast path: raw numpy -> GPU preprocess -> model (no PIL!)
                raw_frames = self.frame_accessor.get_raw_batch(sentinel_indices)
                latents_gpu = self.kernel.get_latents_for_frames(raw_frames)
            else:
                # Legacy path: PIL preprocessing on CPU
                images = self.frame_accessor.get_batch(sentinel_indices)
                latents_gpu = self.kernel.get_latents_for_images(images)
            scores = self.kernel.tracker.get_novelty_scores(latents_gpu)
            is_novel_mask, z_scores = self.kernel.tracker.is_novel(scores)
            
            # --- STAGE 2: ASYNC CPU TRANSFER ---
            # Now, move the original latents to the CPU for the end-of-chunk
            # statistical update. This happens in the background.
            latents_cpu = latents_gpu.to('cpu', non_blocking=True)
        
        # --- STAGE 3: QUEUE RESULTS ---
        # The pending result now includes the pre-calculated scores and masks.
        self.pending_results.append({
            'sentinel_indices': sentinel_indices, 
            'latents_for_update': latents_cpu, # Keep CPU version for the update list
            'scores': scores,
            'is_novel_mask': is_novel_mask,
            'z_scores': z_scores,
            'start_idx': start_idx, 'end_idx': end_idx, 'depth': depth,
            'is_hot': is_hot, 'k_to_use': k_to_use
        })
        return True
    
    def poll_results(self):
        """
        --- REFACTORED FOR EFFICIENCY ---
        Processes results from the GPU. It no longer needs to move data back to
        the GPU as the novelty scores have already been calculated.
        """
        if not self.pending_results or not self.stream.query():
            return False
        
        result = self.pending_results.popleft()
        
        # --- MODIFICATION: Directly use the pre-calculated results ---
        scores = result['scores']
        is_novel_mask = result['is_novel_mask']
        z_scores = result['z_scores']
        
        # --- FIX: This now correctly appends a CPU tensor, preventing VRAM leaks ---
        self.computed_latents_for_update.append(result['latents_for_update'])
        
        sub_spans = []
        for i in range(len(result['sentinel_indices']) - 1):
            sub_span_start = int(result['sentinel_indices'][i])
            sub_span_end = int(result['sentinel_indices'][i+1])
            is_novel = is_novel_mask[i].item() or is_novel_mask[i+1].item()
            max_z = max(z_scores[i].item(), z_scores[i+1].item())
            sub_spans.append({'start': sub_span_start, 'end': sub_span_end, 'is_hot': is_novel, 'z_score': max_z})
            
            if is_novel:
                event_idx_in_batch = i if z_scores[i] > z_scores[i+1] else i + 1
                # Use the pre-calculated z_score directly from the result
                z_score_value = z_scores[event_idx_in_batch].item()
                # --- FIX: Use the novelty threshold from the agent's current config ---
                if z_score_value > self.config.get('novelty_z_score_threshold', 0.1):
                    event = {
                        "source_kernel": self.kernel.kernel_name,
                        "timestamp": self.timestamps[result['sentinel_indices'][event_idx_in_batch]],
                        "score": scores[event_idx_in_batch].item(),
                        "z_score": z_score_value,
                    }
                    self.event_context.publish(z_score_value, event)
        
        sub_spans.sort(key=lambda x: x['z_score'], reverse=True)
        k = result['k_to_use'] if any(s['is_hot'] for s in sub_spans) else self.config.get("top_k_lazy", 1)
        for i in range(min(k, len(sub_spans))):
            span = sub_spans[i]
            self.work_queue.append((span['start'], span['end'], result['depth'] + 1, span['is_hot']))
            self.search_log.append({'type': 'queue_span', 'depth': result['depth'] + 1, 'z_score': span['z_score']})
        return True # --- FIX: Signal that a result was processed ---
class HierarchicalSearchCoordinator:
    def __init__(self, kernels: list[SalienceKernel], config: dict):
        self.kernels = kernels
        self.config = config
        self.streams = {k.kernel_name: torch.cuda.Stream() for k in kernels}

        # FIX: Instantiate the scheduler so it can be used.
        if "scheduled_hyperparams" in self.config:
            self.scheduler = ConfigScheduler(self.config["scheduled_hyperparams"])
        else:
            self.scheduler = None
    
    def process_chunk(self, frame_accessor: LazyFrameAccessor, timestamps_chunk: list, shutdown_event: Event, progress_queue: Queue) -> dict:
        event_context = EventContext()
        agents = []
        
        if self.scheduler:
            scheduled_values = self.scheduler.get_current_values()
            print(f"Coordinator (step {self.scheduler.current_step}): Using scheduled params: {scheduled_values}")
            self.config.update(scheduled_values)
            self.scheduler.step()

            # --- FIX: Propagate updated config to each kernel's tracker ---
            # This ensures the `is_novel` method uses the correct z-score threshold.
            for kernel in self.kernels:
                kernel.tracker.update_config(scheduled_values)
            
        for kernel in self.kernels:
            agent_config = self.config.copy()
            kernel_specific_config = self.config.get("kernel_configs", {}).get(kernel.kernel_name.lower(), {})
            agent_config.update(kernel_specific_config)
            agents.append(SearchAgent(
                kernel=kernel, agent_config=agent_config, frame_accessor=frame_accessor,
                timestamps=timestamps_chunk, event_context=event_context, stream=self.streams[kernel.kernel_name]
            ))
        iteration_count, concurrent_launches = 0, 0
        shm_name = frame_accessor.shm.name

        # --- CORRECTED COORDINATOR LOOP ---
        while any(agent.has_work() for agent in agents):
            if shutdown_event.is_set(): break
            
            work_done_this_loop, launches_this_loop = False, 0
            for agent in agents:
                if agent.launch_work():
                    work_done_this_loop, launches_this_loop = True, launches_this_loop + 1
                if agent.poll_results():
                    work_done_this_loop = True
                    progress_queue.put({'shm_name': shm_name})
            
            if launches_this_loop > 1: concurrent_launches += 1
            if not work_done_this_loop: time.sleep(0.001)

            iteration_count += 1

        # flex concurrent launches
        print(f"Chunk stats: {iteration_count} iterations, "
          f"{concurrent_launches} had concurrent launches "
          f"({100*concurrent_launches/iteration_count:.1f}% parallelism)")
        
        for agent in agents: agent.stream.synchronize()
        
        for agent in agents:
            if agent.computed_latents_for_update:
                # 1. Concatenate all CPU tensors into one large CPU tensor.
                all_latents_cpu = torch.cat(agent.computed_latents_for_update, dim=0)
                # 2. Move the entire batch to the GPU in one efficient transfer.
                all_latents_gpu = all_latents_cpu.to(agent.kernel.device)
                # 3. Update the tracker model.
                agent.kernel.tracker.update(all_latents_gpu)
        
        final_keyframes = []
        # --- MODIFIED: Unpack the 3-element tuple blocking heapq deadlocks ---
        for z_score, count, event in event_context.priority_events:
            source_kernel = next(k for k in self.kernels if k.kernel_name == event["source_kernel"])
            if event["z_score"] > self.config['novelty_z_score_threshold']:
                 final_keyframes.append(
                    source_kernel.create_keyframe_event(
                        timestamp=event["timestamp"], score=event["score"],
                        reason="cooperative_search_spike", z_score=event["z_score"]
                    )
                )
        final_search_log = {}
        for agent in agents:
            final_search_log[agent.kernel.kernel_name] = agent.search_log

        return {'keyframes': final_keyframes, 'search_log': final_search_log}

# =====================================================================================
#  KERNEL REGISTRY
# =====================================================================================
KERNEL_REGISTRY = {"siglip": SigLIPKernel, "ocr": OCRLatentKernel}


# =====================================================================================
#  TWO-PHASE TRIAGE WORKER (PAGED MEMORY MODEL)
# =====================================================================================

class CoarseTriageResult:
    """Result of coarse triage on a page."""
    def __init__(self, page_id: str, is_interesting: bool, max_novelty_score: float,
                 sentinel_indices: list, hot_regions: list, latents_for_update: torch.Tensor):
        self.page_id = page_id
        self.is_interesting = is_interesting
        self.max_novelty_score = max_novelty_score
        self.sentinel_indices = sentinel_indices
        self.hot_regions = hot_regions  # List of (start_idx, end_idx) tuples
        self.latents_for_update = latents_for_update


class TwoPhaseTriageCoordinator:
    """
    Coordinator for two-phase triage:
    1. COARSE TRIAGE: Few sentinels (6), quick decision: retain or evict
    2. FINE REFINEMENT: Only for retained pages, preemptible

    Key insight: Optimize for EVICTION LATENCY, not precision.
    Most pages are boring - decide that fast and free the memory.
    """

    def __init__(self, kernels: list, config: dict):
        self.kernels = kernels
        self.config = config
        self.coarse_sentinels = config.get('coarse_triage_sentinels', 6)
        self.coarse_threshold = config.get('coarse_triage_novelty_threshold', 0.3)
        self.fine_max_p = config.get('fine_refinement_max_p', 0.15)

        # Scheduler for hyperparameter warmup
        from hyperparameter_schedulers import ConfigScheduler
        if "scheduled_hyperparams" in self.config:
            self.scheduler = ConfigScheduler(self.config["scheduled_hyperparams"])
        else:
            self.scheduler = None

    @torch.no_grad()
    def coarse_triage(self, frame_accessor, timestamps: list, kernel) -> CoarseTriageResult:
        """
        Fast coarse triage using few sentinel frames.

        Returns:
            CoarseTriageResult with decision: is_interesting, hot_regions, etc.
        """
        num_frames = len(frame_accessor)
        page_id = frame_accessor.shm.name

        # Sample few sentinel indices (e.g., 6 evenly spaced)
        sentinel_indices = np.linspace(0, num_frames - 1, self.coarse_sentinels, dtype=int).tolist()

        # Get frames and compute latents (use GPU path if available)
        if hasattr(kernel, 'get_latents_for_frames') and hasattr(kernel, 'gpu_preprocessor') and kernel.gpu_preprocessor is not None:
            raw_frames = frame_accessor.get_raw_batch(sentinel_indices)
            latents = kernel.get_latents_for_frames(raw_frames)
        else:
            images = frame_accessor.get_batch(sentinel_indices)
            latents = kernel.get_latents_for_images(images)

        # Get novelty scores
        scores = kernel.tracker.get_novelty_scores(latents)
        is_novel_mask, z_scores = kernel.tracker.is_novel(scores)

        # Decision: is ANY sentinel novel enough to warrant retention?
        max_z_score = z_scores.max().item() if len(z_scores) > 0 else 0.0
        is_interesting = max_z_score > self.coarse_threshold

        # Identify hot regions (pairs of adjacent sentinels where either is novel)
        hot_regions = []
        if is_interesting:
            for i in range(len(sentinel_indices) - 1):
                if is_novel_mask[i].item() or is_novel_mask[i + 1].item():
                    hot_regions.append((sentinel_indices[i], sentinel_indices[i + 1]))

        # Latents go to CPU for model update
        latents_cpu = latents.to('cpu', non_blocking=True)

        return CoarseTriageResult(
            page_id=page_id,
            is_interesting=is_interesting,
            max_novelty_score=max_z_score,
            sentinel_indices=sentinel_indices,
            hot_regions=hot_regions,
            latents_for_update=latents_cpu
        )

    @torch.no_grad()
    def fine_refinement(self, frame_accessor, timestamps: list, kernel,
                        hot_regions: list, check_preemption: callable) -> dict:
        """
        Fine refinement of interesting page. PREEMPTIBLE.

        Args:
            frame_accessor: Access to page frames
            timestamps: Frame timestamps
            kernel: Salience kernel to use
            hot_regions: List of (start_idx, end_idx) from coarse triage
            check_preemption: Callable that returns True if we should yield

        Returns:
            Dict with keyframes, was_preempted flag, latents_for_update
        """
        num_frames = len(frame_accessor)
        budget = int(num_frames * self.fine_max_p)
        frames_processed = 0

        keyframes = []
        all_latents = []
        was_preempted = False

        timestamps_arr = np.array(timestamps)

        # Work queue: (start_idx, end_idx, depth)
        work_queue = deque(hot_regions)

        while work_queue and frames_processed < budget:
            # Check for preemption (new page needs triage)
            if check_preemption():
                was_preempted = True
                break

            start_idx, end_idx = work_queue.popleft()
            span_size = end_idx - start_idx

            if span_size < 4:
                # Span too small, just sample the midpoint
                mid_idx = (start_idx + end_idx) // 2
                indices = [mid_idx]
            else:
                # Sample 4 sentinels within this span
                indices = np.linspace(start_idx, end_idx, 4, dtype=int).tolist()

            frames_processed += len(indices)

            # Use GPU preprocessing path if available
            if hasattr(kernel, 'get_latents_for_frames') and hasattr(kernel, 'gpu_preprocessor') and kernel.gpu_preprocessor is not None:
                raw_frames = frame_accessor.get_raw_batch(indices)
                latents = kernel.get_latents_for_frames(raw_frames)
            else:
                images = frame_accessor.get_batch(indices)
                latents = kernel.get_latents_for_images(images)
            scores = kernel.tracker.get_novelty_scores(latents)
            is_novel_mask, z_scores = kernel.tracker.is_novel(scores)

            all_latents.append(latents.to('cpu', non_blocking=True))

            # Record keyframes and queue sub-regions
            for i, idx in enumerate(indices):
                if is_novel_mask[i].item():
                    keyframes.append({
                        "type": f"{kernel.kernel_name}_KEYFRAME",
                        "timestamp": timestamps_arr[idx],
                        "frame_index": idx,
                        "z_score": z_scores[i].item(),
                        "score": scores[i].item(),
                        "reason": "fine_refinement"
                    })

            # Queue sub-regions around novel points (if budget allows)
            if span_size > 8:
                for i in range(len(indices) - 1):
                    if is_novel_mask[i].item() or is_novel_mask[i + 1].item():
                        sub_start, sub_end = indices[i], indices[i + 1]
                        if sub_end - sub_start > 2:
                            work_queue.append((sub_start, sub_end))

        latents_for_update = torch.cat(all_latents, dim=0) if all_latents else None

        return {
            "keyframes": keyframes,
            "was_preempted": was_preempted,
            "frames_processed": frames_processed,
            "latents_for_update": latents_for_update
        }


def paged_triage_worker_loop(work_queue, result_queue, config, output_path, shutdown_event, preemption_flag):
    """
    Two-phase triage worker for the paged memory model.

    Handles both:
    1. COARSE_TRIAGE: Quick decision on new pages (6 sentinels)
    2. FINE_REFINEMENT: Detailed search on interesting pages (preemptible)

    The worker checks preemption_flag periodically during refinement.
    If a new page arrives for triage, refinement yields.
    """
    worker_pid = os.getpid()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config['device'] = device

    # Initialize kernels
    kernels = [KERNEL_REGISTRY[name](config) for name in config.get("salience_kernels", ["siglip"])]
    coordinator = TwoPhaseTriageCoordinator(kernels, config)
    preprocessor = lambda frame: _downscale_image(Image.fromarray(frame))

    print(f"[{worker_pid}] Paged triage worker started with {len(kernels)} kernels")

    with sdp_kernel(enable_flash=True, enable_mem_efficient=True, enable_math=True):
        log_attention_backend_status(worker_pid)

        while True:
            try:
                if shutdown_event.is_set():
                    break

                task = work_queue.get(timeout=0.1)
                if task is None:
                    break

                work_type = task.get('work_type', 'COARSE_TRIAGE')
                page_id = task.get('page_id')

                frame_accessor = LazyFrameAccessor(
                    shm_name=task['shm_name'],
                    shape=task['shape'],
                    dtype=task['dtype'],
                    preprocess_fn=preprocessor
                )

                if work_type == 'COARSE_TRIAGE':
                    # Fast coarse triage
                    results = []
                    for kernel in kernels:
                        result = coordinator.coarse_triage(
                            frame_accessor, task['timestamps'], kernel
                        )
                        results.append(result)

                        # Update kernel's tracker with coarse latents
                        if result.latents_for_update is not None:
                            latents_gpu = result.latents_for_update.to(device)
                            kernel.tracker.update(latents_gpu)

                    # Combine results: interesting if ANY kernel says so
                    is_interesting = any(r.is_interesting for r in results)
                    max_novelty = max(r.max_novelty_score for r in results)
                    hot_regions = []
                    for r in results:
                        hot_regions.extend(r.hot_regions)

                    result_message = {
                        "work_type": "COARSE_TRIAGE",
                        "page_id": page_id,
                        "shm_name": task['shm_name'],
                        "shape": task['shape'],
                        "dtype": task['dtype'],
                        "timestamps": task['timestamps'],
                        "is_interesting": is_interesting,
                        "max_novelty": max_novelty,
                        "hot_regions": hot_regions,
                    }
                    result_queue.put(result_message)

                elif work_type == 'FINE_REFINEMENT':
                    # Detailed refinement (preemptible)
                    def check_preemption():
                        return preemption_flag.is_set()

                    all_keyframes = []
                    was_preempted = False

                    for kernel in kernels:
                        result = coordinator.fine_refinement(
                            frame_accessor,
                            task['timestamps'],
                            kernel,
                            task.get('hot_regions', []),
                            check_preemption
                        )

                        all_keyframes.extend(result['keyframes'])
                        was_preempted = was_preempted or result['was_preempted']

                        # Update kernel's tracker
                        if result['latents_for_update'] is not None:
                            latents_gpu = result['latents_for_update'].to(device)
                            kernel.tracker.update(latents_gpu)

                        if was_preempted:
                            break

                    result_message = {
                        "work_type": "FINE_REFINEMENT",
                        "page_id": page_id,
                        "shm_name": task['shm_name'],
                        "shape": task['shape'],
                        "dtype": task['dtype'],
                        "timestamps": task['timestamps'],
                        "keyframes": all_keyframes,
                        "was_preempted": was_preempted,
                    }
                    result_queue.put(result_message)

                frame_accessor.close()

            except queue.Empty:
                continue
            except Exception as e:
                print(f"[{worker_pid}] ERROR in paged triage worker: {e}")
                import traceback; traceback.print_exc()


# =====================================================================================
#  LEGACY MAIN WORKER LOOP (for backwards compatibility)
# =====================================================================================
def analysis_worker_loop(task_queue, return_queue, config, output_path, shutdown_event, progress_queue):
    worker_pid = os.getpid()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config['device'] = device
    kernels = [KERNEL_REGISTRY[name](config) for name in config.get("salience_kernels", ["siglip"])]
    search_coordinator = HierarchicalSearchCoordinator(kernels, config)
    preprocessor = lambda frame: _downscale_image(Image.fromarray(frame))

    # This context manager will globally set the most efficient attention backend
    # for all operations within this worker process's lifetime.
    # The priority is Flash > Memory-Efficient > Eager Math.
    with sdp_kernel(enable_flash=True, enable_mem_efficient=True, enable_math=True):
        # Log the chosen backend for confirmation.
        log_attention_backend_status(worker_pid)

        while True:
            try:
                if shutdown_event.is_set(): break
                task = task_queue.get(timeout=0.1)
                if task is None: break

                frame_accessor = LazyFrameAccessor(
                    shm_name=task['shm_name'], shape=task['shape'],
                    dtype=task['dtype'], preprocess_fn=preprocessor
                )
                # --- Pass progress_queue to the coordinator ---
                result_data = search_coordinator.process_chunk(frame_accessor, task['timestamps'], shutdown_event, progress_queue)
                return_message = {
                    "shm_name": task['shm_name'], "shape": task['shape'],
                    "dtype": task['dtype'], "timestamps": task['timestamps'],
                    "source": "HierarchicalSearchCoordinator", "data": result_data
                }
                #way too big to print lol
                #print(f"repr:{repr(result_data)}")
                return_queue.put(return_message)
                frame_accessor.close()
            except queue.Empty:
                continue
            except Exception as e:
                print(f"[{worker_pid}] CRITICAL ERROR in worker loop: {e}")
                import traceback; traceback.print_exc()
                if 'task' in locals() and task:
                    failure_message = {
                        "shm_name": task['shm_name'], "shape": task.get('shape'),
                        "dtype": task.get('dtype'), "timestamps": task.get('timestamps'),
                        "source": "HierarchicalSearchCoordinator", "data": {"error": str(e)}
                    }
                    return_queue.put(failure_message)
                    if 'shm' in locals() and shm: shm.close()