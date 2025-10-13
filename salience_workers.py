# salience_workers.py

import numpy as np
import torch
import torch.nn.functional as F
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
        #max 4 in-flight operations
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
        # --- NEW: Increment the counter ---
        self.frames_processed += len(sentinel_indices)

        with torch.cuda.stream(self.stream):
            images = self.frame_accessor.get_batch(sentinel_indices)
            latents = self.kernel.get_latents_for_images(images)
            latents_cpu = latents.to('cpu', non_blocking=True)
        
        self.pending_results.append({
            'sentinel_indices': sentinel_indices, 'latents': latents_cpu,
            'start_idx': start_idx, 'end_idx': end_idx, 'depth': depth,
            'is_hot': is_hot, 'k_to_use': k_to_use
        })
        return True
    
    def poll_results(self):
        if not self.pending_results or not self.stream.query():
            return False # --- FIX: Signal that no results were processed ---
        
        result = self.pending_results.popleft()
        latents_cpu = result['latents']
        latents_gpu = latents_cpu.to(self.kernel.device)
        scores = self.kernel.tracker.get_novelty_scores(latents_gpu)
        is_novel_mask, z_scores = self.kernel.tracker.is_novel(scores)
        
        self.computed_latents_for_update.append(latents_gpu)
        
        sub_spans = []
        for i in range(len(result['sentinel_indices']) - 1):
            sub_span_start = int(result['sentinel_indices'][i])
            sub_span_end = int(result['sentinel_indices'][i+1])
            is_novel = is_novel_mask[i].item() or is_novel_mask[i+1].item()
            max_z = max(z_scores[i].item(), z_scores[i+1].item())
            sub_spans.append({'start': sub_span_start, 'end': sub_span_end, 'is_hot': is_novel, 'z_score': max_z})
            
            if is_novel:
                event_idx_in_batch = i if z_scores[i] > z_scores[i+1] else i + 1
                event = {
                    "source_kernel": self.kernel.kernel_name,
                    "timestamp": self.timestamps[result['sentinel_indices'][event_idx_in_batch]],
                    "score": scores[event_idx_in_batch].item(),
                    "z_score": z_scores[event_idx_in_batch].item(),
                }
                self.event_context.publish(z_scores[event_idx_in_batch].item(), event)
        
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
    
    def process_chunk(self, frame_accessor: LazyFrameAccessor, timestamps_chunk: list, shutdown_event: Event, progress_queue: Queue) -> dict:
        event_context = EventContext()
        agents = []
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
                all_latents = torch.cat(agent.computed_latents_for_update, dim=0)
                agent.kernel.tracker.update(all_latents)
        
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
#  MAIN WORKER LOOP
# =====================================================================================
def analysis_worker_loop(task_queue, return_queue, config, output_path, shutdown_event, progress_queue):
    worker_pid = os.getpid()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config['device'] = device
    kernels = [KERNEL_REGISTRY[name](config) for name in config.get("salience_kernels", ["siglip"])]
    search_coordinator = HierarchicalSearchCoordinator(kernels, config)
    preprocessor = lambda frame: _downscale_image(Image.fromarray(frame))

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