# recording_stuff.py

import mss
import pynput
import torch
import torch.nn.functional as F
import pyarrow as pa
import pyarrow.parquet as pq
import orjson
import time
import uuid
import os
import io
import sys
import numpy as np
import subprocess
import queue 
from multiprocessing import Process, Queue, shared_memory, Event
from threading import Thread, Lock
from PIL import Image
from collections import deque, defaultdict
from dataclasses import dataclass
import psutil

# We now need one of the salience workers in the Triage process itself
import salience_workers as sal_wo
from window_utils import get_window_finder, WindowNotFoundError
import hmi_utils #breaking off gui and keylog capture into an import

# OBS-based capture (preferred over fragile window title matching)
try:
    from obs_capture import CaptureManager, obs_capture_process_loop
    OBS_AVAILABLE = True
except ImportError:
    OBS_AVAILABLE = False
    print("NOTE: obs_capture not available, using legacy window capture") 

# --- Configuration ---
#RECORDING_WINDOW_NAME = "Yunyun Syndrome!? Rhythm Psychosis Demo Version 1.0.1"
RECORDING_WINDOW_NAME = "Sulfur"
CAPTURE_REGION = None
OUTPUT_PATH = f"./capture_run_{int(time.time())}"
CAPTURE_FPS = 60

# --- PAGED MEMORY MODEL ---
# Small pages for fast eviction decisions (was 12s chunks)
from paged_memory import (
    PAGE_SECONDS, PAGE_FRAMES_AT_60FPS, COARSE_TRIAGE_SENTINELS,
    COARSE_TRIAGE_NOVELTY_THRESHOLD, FINE_REFINEMENT_MAX_P,
    MAX_RETAINED_PAGES, MAX_PENDING_TRIAGE, BASELINE_TARGET_FPS,
    PageState, PageMetadata, GlobalScheduler, WorkType, WorkItem,
    create_page_from_frames, get_decimated_frames_for_baseline
)

PAGE_DEADLINE_SECONDS = 30.0  # Much shorter deadline for small pages
KEYFRAME_PADDING_SECONDS = 0.5 # How many seconds before and after a keyframe to include in a clip

# Legacy compatibility (will be removed)
CHUNK_SECONDS = PAGE_SECONDS
UNHANDLED_SHM_DEADLINE_SECONDS = PAGE_DEADLINE_SECONDS
MAX_PIPELINE_DEPTH = MAX_RETAINED_PAGES + MAX_PENDING_TRIAGE
MIN_PIPELINE_DEPTH = 2

# --- Triage Configuration ---
TRIAGE_THRESHOLD = COARSE_TRIAGE_NOVELTY_THRESHOLD

# --- NEW: Salience Strategy Configuration ---
# Choose which algorithm to use for novelty detection.
# Options: "naive_cos_dissimilarity", "pca_mahalanobis"
SALIENCE_STRATEGY = "pca_mahalanobis" 
SALIENCE_KERNELS = ["siglip", "ocr"] # Add "ocr" to this list to enable the OCR kernel

# --- NEW: Hyperparameters for the strategies ---
# These are passed to the salience worker config.
STRATEGY_CONFIGS = {
    "naive_cos_dissimilarity": {
        "z_score_threshold": 0.5, # The threshold for the simple Z-score method.
        "branching_factor": 8, # Override the worker's default of 8
        "min_exhaustive_size": 16, # Override the worker's default of 16 
        "max_batch_size": 16,
        "optimistic_top_k": 1,       # How many of the "most interesting" non-novel branches to explore.
        "optimistic_max_depth": 1,   # How many consecutive optimistic steps are allowed before a branch must prove its novelty.
    },
    "pca_mahalanobis": {
        "pca_n_components": 16,      # Max components (top-k guardrail).
        "pca_variance_threshold": 0.95, # The variance to capture (top-p).
        "novelty_z_score_threshold": 0.1, # Z-score threshold for the Mahalanobis scores themselves.
        "branching_factor": 8, # Override the worker's default of 8
        "min_exhaustive_size": 8, # Override the worker's default of 16
        "max_batch_size": 8,
        "top_k": 1,          # Explore the top 2 sub-spans in a "hot" region.
        "max_d": 4,          # Max recursion depth for "hot" regions.
        "top_k_lazy": 1,     # Explore only the top sub-span in a "cold" region.
        "max_d_lazy": 2,     # Give up on "cold" regions faster.
        "max_p": 0.33,       # Hard budget: process at most 33% of total frames in a chunk.
        "kernel_configs": {
            "siglip": {
                "max_batch_size": 8 # SigLIP is efficient
            },
            "ocr": {
                "max_batch_size": 4   # OCR model is huge, process one by one
            }
    }
}
}
SCHEDULED_HYPERPARAMS = {
    "max_p": {
        "initial_value": 0.05,   # Start with a modest 5% search budget...
        "sustain_value": 0.12,   # need an extremely light encoder which finishes super fast to use ~1/3 max_p
        "warmup_steps": 2,       # Hold the initial value for the first 2 chunks.
        "ramp_steps": 24,        # Linearly ramp up over the next 12 chunks (2 minutes @ 10s/chunk).
    },
    "novelty_z_score_threshold": {
        "initial_value": 0.75,   # Start with a higher threshold (less sensitive)...
        "sustain_value": 0.01,    # ...and ramp down to the normal, more sensitive value.
        "warmup_steps": 4,       # Hold for 4 chunks.
        "ramp_steps": 24,        # Ramp down over the next 8 chunks.
    }
}

class ProgressLogger:
    """
    A utility to consolidate repetitive log lines into a single, updating line,
    similar to tqdm's postfix.
    """
    def __init__(self):
        self.last_message_key = None
        self.message_counts = defaultdict(int)
        self.last_line_len = 0

    def log(self, message_key: str, message_template: str):
        """
        Logs a message. If the key is the same as the last one, it updates
        the previous line. If it's different, it finalizes the old line
        and starts a new one.
        """
        # If the message type has changed, finalize the previous line with a newline.
        if message_key != self.last_message_key:
            if self.last_message_key is not None:
                sys.stdout.write("\n")
            self.last_message_key = message_key
            # Reset the counter for this new message type
            self.message_counts[message_key] = 0

        # Increment the count for the current message
        self.message_counts[message_key] += 1
        count = self.message_counts[message_key]

        # Format the display string with the counter
        display_string = f"{message_template} (x{count})"
        
        # Calculate padding to overwrite any previous, longer line
        padding = ' ' * max(0, self.last_line_len - len(display_string))
        
        # Use carriage return `\r` to return to the start of the line and overwrite
        sys.stdout.write(f"\r{display_string}{padding}")
        sys.stdout.flush()
        
        self.last_line_len = len(display_string)

    def finalize(self):
        """Call at the end of the program to ensure the last log line gets a newline."""
        if self.last_message_key is not None:
            sys.stdout.write("\n")
            sys.stdout.flush()

def video_encoding_worker_loop(task_queue, result_queue, output_path):
    # This worker gets its own encoder instance
    video_encoder = hmi_utils.VideoEncoder(output_path)
    print(f"[{os.getpid()}] Encoding worker started.")
    
    while True:
        task = task_queue.get()
        if task is None:
            break

        frames = task['frames']
        timestamps = task['timestamps']
        quality = task['quality']

        # This call now happens inside the isolated worker process
        proc = video_encoder.start_encode_slice(frames, timestamps, quality)
        
        if proc:
            # --- MODIFICATION: Capture and print stderr on failure ---
            # proc.wait() returns the exit code. We also want the error output.
            stdout_data, stderr_data = proc.communicate()
            return_code = proc.returncode
            if return_code != 0:
                print(f"[{os.getpid()}] FFMPEG encoding job failed with code {return_code}.")
                # Decode stderr from bytes to string for printing
                error_message = stderr_data.decode('utf-8', errors='ignore')
                if error_message:
                    print(f"--- FFMPEG Error Output ---\n{error_message}\n---------------------------")
            print(f"[{os.getpid()}] Encoding job finished with code {return_code}.")

        # Signal completion back to the main loop
        result_queue.put({'status': 'complete'}) # CHANGED

    print(f"[{os.getpid()}] Encoding worker finished.")

# =====================================================================================
#  NEW: DEDICATED PROCESS LOOPS
# =====================================================================================

def capture_process_loop(raw_frame_queue: Queue, config: dict, shutdown_event: Event):
    """
    The Collector & Baseline Recorder: This single process has two jobs:
    1. Grabs frames at high frequency and puts them onto the salience queue.
    2. Uses a non-blocking internal queue to pass frames to a dedicated writer
       thread that handles the potentially blocking write to the FFMPEG pipe.
    """
    pid = os.getpid()
    print(f"[{pid}] Capture & Baseline Recorder process started.")

    # --- Local State for this Process ---
    video_encoder = hmi_utils.VideoEncoder(config['output_path'])
    window_finder = get_window_finder()

 # --- STATE MACHINE VARIABLES ---
    ffmpeg_proc, ffmpeg_pipe = None, None
    stdin_writer_thread = None
    stderr_reader_thread = None # <-- NEW
    current_dimensions = None
    # This is our new, explicit state management flag.
    baseline_restart_required = True  # Start in a state that requires initialization.

    # --- NEW: Internal queue and thread for non-blocking writes ---
    baseline_write_queue = queue.Queue(maxsize=config['capture_fps']) # Buffer up to 1s of frames

    def _pipe_writer_loop(pipe, internal_queue, shutdown):
        """This function runs in a separate thread. Its only job is to
        pull frames from the internal queue and perform the blocking write."""
        while not shutdown.is_set():
            try:
                frame = internal_queue.get(timeout=0.1)
                # --- THIS IS THE FIX ---
                # Ensure the frame data is in a packed, C-contiguous memory layout
                # before writing its bytes. This removes any stride/padding.
                contiguous_frame = np.ascontiguousarray(frame)
                pipe.write(contiguous_frame.tobytes())
                # --- END FIX ---
            except queue.Empty:
                continue
            except (IOError, BrokenPipeError):
                print(f"[{pid}] FFMPEG writer thread detected a broken pipe. Exiting.")
                break
        print(f"[{pid}] FFMPEG writer thread finished.")

    
    def _pipe_reader_loop(pipe, log_prefix, shutdown):
        """Drains ffmpeg's stderr pipe to prevent deadlocks and logs output."""
        # Use iter() to create a blocking iterator over the pipe's lines.
        # This is a clean way to read until the pipe is closed.
        for line in iter(pipe.readline, b''):
            if shutdown.is_set(): break
            # Decode and print/log the line from ffmpeg
            print(f"[{log_prefix}] {line.decode('utf-8', errors='ignore').strip()}")
        pipe.close()
        print(f"[{pid}] FFMPEG stderr reader thread finished.")

    def _finalize_encoder():
        nonlocal ffmpeg_proc, ffmpeg_pipe_in, ffmpeg_pipe_err, stdin_writer_thread, stderr_reader_thread, current_dimensions
                # Wait for threads to finish their work
        if stdin_writer_thread and stdin_writer_thread.is_alive():
            stdin_writer_thread.join(timeout=1.0)
        if stderr_reader_thread and stderr_reader_thread.is_alive():
            stderr_reader_thread.join(timeout=1.0)

        stdin_writer_thread, stderr_reader_thread = None, None

        if ffmpeg_proc:
            print(f"[{pid}] Finalizing baseline video segment...")
            # Close stdin pipe to signal end of stream
            if ffmpeg_pipe_in:
                try: ffmpeg_pipe_in.close()
                except (IOError, BrokenPipeError): pass
            
            # Wait for the process to terminate and get the final exit code
            ffmpeg_proc.wait(timeout=2.0)
        ffmpeg_proc, ffmpeg_pipe_in, ffmpeg_pipe_err, current_dimensions = None, None, None, None

    try:
        with mss.mss() as sct:
            while not shutdown_event.is_set():
                start_time = time.time()
                try:
                    is_writer_dead = stdin_writer_thread and not stdin_writer_thread.is_alive()
                    is_reader_dead = stderr_reader_thread and not stderr_reader_thread.is_alive()
                    if (is_writer_dead or is_reader_dead) and ffmpeg_proc:
                        print(f"[{pid}] FFMPEG helper thread died. Flagging for restart.")
                        baseline_restart_required = True
                    if baseline_restart_required:
                        _finalize_encoder()
                        baseline_restart_required = False
                    monitor = window_finder.find_window_by_title(config['window_name'])
                    if monitor is None:
                        if ffmpeg_proc: _finalize_encoder()
                        time.sleep(0.5)
                        continue

                    img = sct.grab(monitor)
                    frame_rgb = np.array(img)[:, :, :3][:, :, ::-1]

                    # --- 3. ACTION LOGIC ---
                    # This block handles starting a new encoder if one is needed,
                    # either from a clean slate or due to a dimension change.
                    if ffmpeg_proc is None or frame_rgb.shape != current_dimensions:
                        if ffmpeg_proc is not None:
                             print(f"[{pid}] Detected dimension change to {frame_rgb.shape}.")
                        _finalize_encoder() # Ensure clean state before starting new.
                        print(f"[{pid}] Initializing new baseline encoder.")
                        # Get all three pipes from the new process
                        ffmpeg_proc, ffmpeg_pipe_in, ffmpeg_pipe_err = video_encoder.start_continuous_encode(
                            dimensions=frame_rgb.shape, fps=config['capture_fps']
                        )
                        current_dimensions = frame_rgb.shape
                        stdin_writer_thread = Thread(target=_pipe_writer_loop, args=(ffmpeg_pipe_in, baseline_write_queue, shutdown_event))
                        stdin_writer_thread.start()
                        
                        # Start a reader thread for stderr
                        log_prefix = f"ffmpeg_baseline_{int(start_time)}"
                        stderr_reader_thread = Thread(target=_pipe_reader_loop, args=(ffmpeg_pipe_err, log_prefix, shutdown_event))
                        stderr_reader_thread.start()

                    # --- 4. STEADY-STATE OPERATION ---
                    try:
                        raw_frame_queue.put_nowait((start_time, frame_rgb))
                    except queue.Full:
                        # This is now a recoverable condition, not a fatal error.
                        print(f"[{pid}] WARNING: raw_frame_queue is full. Dropping a frame.")
                        pass # Continue the loop without crashing
                    if stdin_writer_thread:
                        try:
                            baseline_write_queue.put_nowait(frame_rgb)
                        except queue.Full:
                            print(f"[{pid}] WARNING: base_recording queue is fillies!")
                            pass

                except WindowNotFoundError:
                    if ffmpeg_proc: _finalize_encoder()
                    time.sleep(0.5)
                    continue

                elapsed = time.time() - start_time
                sleep_time = (1.0 / config['capture_fps']) - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)
    finally:
        print(f"[{pid}] Capture process shutting down...")
        _finalize_encoder()
        print(f"[{pid}] Capture process finished.")

def _stage_and_dispatch_page(frames: list, timestamps: list, page_queue: Queue, baseline_queue: Queue, shm_registry: dict):
    """
    Handles the creation of a PAGE (small memory block), copying frame data into it,
    and dispatching to both triage and baseline encoding queues.

    Key difference from old chunk model:
    - Pages are small (1.5s instead of 12s)
    - Pages feed BOTH triage AND baseline encoder (no duplication)
    - Decimated frames are extracted here for baseline encoding

    IMPORTANT: shm_registry is used to keep SHM handles alive on Windows!
    Without this, Python garbage collects the handle when this function returns,
    causing the SHM to be destroyed before other processes can access it.
    """
    try:
        # 1. Create the page using the paged memory model
        page, shm = create_page_from_frames(frames, timestamps)

        # 2. Extract decimated frames for baseline encoder BEFORE dispatching
        #    This happens synchronously to avoid SHM lifetime issues
        decimated_frames, decimated_timestamps = get_decimated_frames_for_baseline(
            shm_name=page.shm_name,
            shape=page.shape,
            dtype=page.dtype,
            timestamps=page.timestamps,
            target_fps=BASELINE_TARGET_FPS
        )

        # 3. Dispatch page metadata to triage queue (the page_queue is for GlobalScheduler)
        page_task = {
            "page_id": page.page_id,
            "shm_name": page.shm_name,
            "shape": page.shape,
            "dtype": page.dtype,
            "timestamps": page.timestamps,
            "page_metadata": page,
        }
        page_queue.put(page_task)

        # 4. Dispatch baseline encoding task (metadata only - encoder reads from SHM)
        #    We pass SHM info instead of copying frames through the queue.
        #    The baseline encoder will read and decimate directly from SHM.
        baseline_task = {
            "page_id": page.page_id,
            "shm_name": page.shm_name,
            "shape": page.shape,
            "dtype": "uint8",  # Always uint8 for frame data
            "timestamps": list(page.timestamps),
        }
        baseline_queue.put(baseline_task)

        # 5. Store handle in registry to keep it alive!
        # On Windows, SharedMemory is reference-counted. If we let the handle go
        # out of scope (garbage collected), the SHM is destroyed before the
        # orchestrator can open it. Store in registry to keep alive.
        shm_registry[page.page_id] = shm

    except Exception as e:
        print(f"[Dispatcher] CRITICAL ERROR during page dispatch: {e}")
        import traceback; traceback.print_exc()
        # On error, we DO need to clean up since orchestrator won't receive the page
        if 'shm' in locals() and shm:
            try:
                shm.close()
                shm.unlink()
            except (FileNotFoundError, BufferError):
                pass


# Legacy wrapper for compatibility
def _stage_and_dispatch_chunk(frames: list, timestamps: list, salience_task_queue: Queue, shm_notification_queue: Queue):
    """DEPRECATED: Use _stage_and_dispatch_page instead."""
    # Create a dummy baseline queue and registry (for legacy code paths)
    class DummyQueue:
        def put(self, x): pass
    _legacy_shm_registry = {}  # Will leak handles, but legacy code is deprecated anyway
    _stage_and_dispatch_page(frames, timestamps, salience_task_queue, DummyQueue(), _legacy_shm_registry)
    # Also notify the old shm_notification_queue for legacy orchestrator
    # (This will be removed once PagedOrchestrator is fully integrated)


def dispatcher_loop(raw_frame_queue: Queue, page_queue: Queue, baseline_queue: Queue, config: dict, shutdown_event: Event):
    """
    The Paged Dispatcher: Buffers frames into small PAGES (1.5s instead of 12s chunks)
    and dispatches them to both:
    1. Triage queue (for salience analysis)
    2. Baseline queue (for continuous low-quality recording)

    Key insight: Pages are small enough that eviction decisions happen fast.
    Most pages are boring and can be dropped after baseline encoding.
    """
    worker_pid = os.getpid()
    print(f"[{worker_pid}] Paged Dispatcher started (page_size={PAGE_SECONDS}s, {PAGE_FRAMES_AT_60FPS} frames)")

    frames_per_page = int(config.get('page_seconds', PAGE_SECONDS) * config['capture_fps'])
    frame_buffer = deque(maxlen=frames_per_page)

    # CRITICAL: Keep SHM handles alive on Windows!
    # Without this registry, handles get garbage collected when _stage_and_dispatch_page
    # returns, destroying the SHM before the orchestrator can access it.
    shm_registry = {}

    pages_dispatched = 0

    while not shutdown_event.is_set():
        try:
            timestamp, frame = raw_frame_queue.get(timeout=0.1)
            frame_buffer.append((timestamp, frame))

            if len(frame_buffer) == frames_per_page:
                timestamps, frames = zip(*frame_buffer)

                _stage_and_dispatch_page(
                    frames=list(frames),
                    timestamps=list(timestamps),
                    page_queue=page_queue,
                    baseline_queue=baseline_queue,
                    shm_registry=shm_registry
                )

                pages_dispatched += 1
                if pages_dispatched % 10 == 0:
                    print(f"[{worker_pid}] Dispatched {pages_dispatched} pages (holding {len(shm_registry)} SHM handles)")

                frame_buffer.clear()

        except queue.Empty:
            continue

    # Cleanup: close all SHM handles on shutdown
    print(f"[{worker_pid}] Paged Dispatcher shutting down, releasing {len(shm_registry)} SHM handles...")
    for page_id, shm in shm_registry.items():
        try:
            shm.close()
        except Exception:
            pass

    print(f"[{worker_pid}] Paged Dispatcher finished. Total pages: {pages_dispatched}")

class Orchestrator:
    """
    The Conductor: Manages the lifecycle of all processes and orchestrates the
    high-level flow of information, but does NOT handle raw frame data directly.
    """
    def __init__(self, config):
        self.config = config
        self.is_running = True
        self.shutdown_event = Event()
        self.num_salience_workers = 1   #wtf lol
        self.num_encoding_workers = 2

        # --- NEW: A set of specialized queues for a decoupled system ---
        self.raw_frame_queue_max_size = config['capture_fps'] * 3
        self.raw_frame_queue = Queue(maxsize=self.raw_frame_queue_max_size) # Buffer 3s of raw frames
        self.salience_task_queue = Queue()
        self.salience_results_queue = Queue()
        self.encoding_task_queue = Queue()
        self.encoding_results_queue = Queue()
        
        # --- State for managing temporary shared memory blocks ---
        self.shm_notification_queue = Queue() 
        self.encoding_task_queue = Queue()
        self.encoding_results_queue = Queue()

        # --- ADD THIS LINE ---
        self.progress_queue = Queue()
        
        # --- MODIFIED: This dictionary will now hold active SHM handles ---
        self.active_shm_blocks = {} # {shm_name: shm_instance}
        self.shm_lock = Lock()
        self.max_pipeline_depth = MAX_PIPELINE_DEPTH # Max number of concurrent SHM blocks
        self.min_pipeline_depth = MIN_PIPELINE_DEPTH  # Target to shrink back towards
        
        # High-level components
        self.persistence_manager = AsyncPersistenceManager(config['output_path'], Queue()) # Give it its own queue
        self.guardian_thread = Thread(target=self._guardian_loop, daemon=True)
        self.frame_archiver = hmi_utils.FrameArchiver(config['output_path'])
        self.video_encoder = hmi_utils.VideoEncoder(config['output_path'])
        
        self.processes = []
        self.threads = []
        self.logger = ProgressLogger()

        # --- NEW: Add a timer for periodic status logging ---
        self.last_log_time = time.time()
        self.log_interval_seconds = 4.0 # Log status every 4 seconds

    def _log_pipeline_status(self):
        """Prints a comprehensive, single-line status of all major queues."""
        now = time.time()
        if (now - self.last_log_time) < self.log_interval_seconds:
            return

        with self.shm_lock:
            active_chunks = len(self.active_shm_blocks)

        # Using qsize() is generally safe for logging/monitoring purposes
        pending_salience = self.salience_task_queue.qsize()
        buffered_raw_frames = self.raw_frame_queue.qsize()
        max_raw_frames = self.raw_frame_queue_max_size
        pending_encodes = self.encoding_task_queue.qsize()

        status_line = (
            f"[Pipeline Status] "
            f"Active Chunks: {active_chunks} | "
            f"Pending Salience: {pending_salience} | "
            f"Raw Frames: {buffered_raw_frames}/{max_raw_frames} | "
            f"Pending Encodes: {pending_encodes}"
        )

        # Use the ProgressLogger to avoid spamming the console with newlines
        self.logger.log("pipeline_status", status_line)
        self.last_log_time = now

    def start(self):
        """Starts all worker processes and the main orchestration loop."""
        self.start_workers()
        self.persistence_manager.start()
        self.guardian_thread.start() # Start the guardian
        
        print("Starting main orchestration loop... Press Ctrl+C to stop.")
        try:
            while self.is_running:
                # --- Main Orchestration Loop ---
                # 1. Claim new SHM blocks from the Dispatcher
                self.claim_new_shm_blocks()
                # 2. Process results from salience workers
                self.process_salience_results_queue()
                # 3. Check for completed video encodes
                self.process_encoding_results_queue()
                # 3.1. homebrew pbar i guess
                self.process_progress_queue()
                # 4. NEW: Audit active SHM blocks for timeouts
                self.audit_shm_blocks()
                # 5. NEW: Adjust pipeline parameters based on load
                self.adapt_pipeline()

                # --- NEW: Call the status logger on every loop ---
                self._log_pipeline_status()
                
                time.sleep(0.012)

        except (KeyboardInterrupt, SystemExit):
            self.is_running = False
        finally:
            self.shutdown()

    def process_salience_results_queue(self):
        """
        Processes keyframe events and search logs from a salience worker.
        This version has a single, robust, race-free cleanup path.
        """
        while not self.salience_results_queue.empty():
            result = self.salience_results_queue.get_nowait()
            shm_name = result['shm_name']

            # --- MODIFICATION: Update status before processing ---
            with self.shm_lock:
                if shm_name in self.active_shm_blocks:
                    # We just heard from the worker. Reset the timeout clock for this block.
                    self.active_shm_blocks[shm_name]['timestamp'] = time.time() 
                    self.active_shm_blocks[shm_name]['status'] = 'processing_complete' # Update status
                else:
                    # This can still happen if the worker was *extremely* slow, but it's much less likely.
                    print(f"Orchestrator: Received result for {shm_name}, but it's no longer active. Discarding.")
                    continue

            shm_name = result['shm_name']
            data = result.get('data', {})
            keyframes = data.get('keyframes', [])
            search_log = data.get('search_log', [])

            # We DO NOT clean up the SHM block yet.

            # --- STEP 1: Process and persist all metadata ---
            ##print(f"Orchestrator: Received {len(keyframes)} keyframes and {len(search_log)} search log entries from {shm_name}.")
            if search_log:
                log_event = {
                    "event_id": str(uuid.uuid4()),
                    "stream_type": "SALIENCE_SEARCH_LOG",
                    "start_timestamp": result['timestamps'][0],
                    "delta_timestamp": result['timestamps'][-1] - result['timestamps'][0],
                    "payload_json": orjson.dumps({'shm_name': shm_name, 'log': search_log}).decode('utf-8')
                }
                self.persistence_manager.event_queue.put(log_event)

            # --- STEP 2: If keyframes exist, dispatch encoding jobs that need the SHM block ---
            if keyframes:
                # Merge time intervals
                padding = self.config.get('keyframe_padding_seconds', 2.0) # Default to 2.0 if not set
                intervals = [[event['timestamp'] - padding, event['timestamp'] + padding] for event in keyframes]
                intervals.sort(key=lambda x: x[0])
                merged_intervals = []
                if intervals:
                    current_merge = intervals[0]
                    for next_interval in intervals[1:]:
                        if next_interval[0] <= current_merge[1]:
                            current_merge[1] = max(current_merge[1], next_interval[1])
                        else:
                            merged_intervals.append(current_merge)
                            current_merge = next_interval
                    merged_intervals.append(current_merge)

                video_encode_jobs = [{'start_time': s, 'end_time': e, 'quality': 'HIGH'} for s, e in merged_intervals]
                
                if video_encode_jobs:
                    ##print(f"Orchestrator: Merged {len(keyframes)} events into {len(video_encode_jobs)} encoding job(s).")
                    try:
                        # Open the SHM block for reading. This should now succeed.
                        shm = shared_memory.SharedMemory(name=shm_name)
                        buffer = np.ndarray(result['shape'], dtype=result['dtype'], buffer=shm.buf)
                        timestamps = np.array(result['timestamps'])
                        
                        # Dispatch frames to the encoder
                        for job in video_encode_jobs:
                            mask = (timestamps >= job['start_time']) & (timestamps <= job['end_time'])
                            indices = np.where(mask)[0]
                            if len(indices) > 0:
                                frames_to_encode = buffer[indices]
                                timestamps_to_encode = timestamps[indices]
                                self.encoding_task_queue.put({
                                    'frames': frames_to_encode,
                                    'timestamps': timestamps_to_encode,
                                    'quality': job['quality'],
                                })
                        # We are done READING. Close our handle.
                        shm.close()
                    except FileNotFoundError:
                        pass
                        ##print(f"WARNING: Could not find SHM block {shm_name} for encoding. This might be a shutdown race condition.")

            # --- FINAL STEP: The block has served all purposes. Destroy it. ---
            self.cleanup_shm_block(shm_name)

    # --- FIX: Add the missing method definition here ---
    def process_encoding_results_queue(self):
        """Checks for and logs completed video encoding jobs."""
        while not self.encoding_results_queue.empty():
            result = self.encoding_results_queue.get_nowait()
            # In a more complex system, you might log this completion.
            # For now, just confirming it's done is enough.
            print(f"Orchestrator: Confirmed completion of encode job.")

    def process_progress_queue(self):
        """
        Processes heartbeat messages from Salience workers. A heartbeat
        indicates that a chunk is still being actively processed, so we
        reset its timeout clock to prevent it from being prematurely cleaned up.
        """
        while not self.progress_queue.empty():
            try:
                msg = self.progress_queue.get_nowait()
                shm_name = msg.get('shm_name')

                if shm_name:
                    with self.shm_lock:
                        if shm_name in self.active_shm_blocks:
                            # We received a sign of life! Reset the timeout timer.
                            self.active_shm_blocks[shm_name]['timestamp'] = time.time()
                            
                            # Optional: Log the heartbeat for debugging
                            # The key is the shm_name itself
                            key = f"heartbeat_{shm_name}"
                            # The template is the string without the counter
                            template = f"Orchestrator: Received heartbeat for {shm_name}. Resetting timeout."
                            self.logger.log(key, template)
            except queue.Empty:
                break
            except Exception as e:
                print(f"Orchestrator: Error processing progress queue: {e}")

    def claim_new_shm_blocks(self):
        """Claims new SHM blocks announced by the Dispatcher."""
        while not self.shm_notification_queue.empty():
            shm_name = self.shm_notification_queue.get_nowait()
            try:
                shm = shared_memory.SharedMemory(name=shm_name)
                with self.shm_lock:
                    self.active_shm_blocks[shm_name] = {
                        "shm": shm,
                        "timestamp": time.time(),
                        "status": "pending_analysis" 
                    }
            except FileNotFoundError:
                print(f"Orchestrator WARNING: Notified of {shm_name}, but it disappeared.")

    def audit_shm_blocks(self):
        """Finds and cleans up abandoned SHM blocks."""
        now = time.time()
        abandoned_blocks = []
        with self.shm_lock:
            for name, meta in self.active_shm_blocks.items():
                if (now - meta['timestamp']) > UNHANDLED_SHM_DEADLINE_SECONDS:
                    print(f"Orchestrator: SHM block {name} has been unhandled for too long. Declaring abandoned.")
                    abandoned_blocks.append(name)
        
        for name in abandoned_blocks:
            self.cleanup_shm_block(name)

    def cleanup_shm_block(self, name: str):
        """Safely closes, unlinks, and removes an SHM block from tracking."""
        with self.shm_lock:
            if name in self.active_shm_blocks:
                meta = self.active_shm_blocks.pop(name)
                meta['shm'].close()
                meta['shm'].unlink()
                key = f"cleanup_{name}"
                template = f"Orchestrator: Cleaned up SHM block {name}."
                self.logger.log(key, template)

    def adapt_pipeline(self):
        """Dynamically adjusts capture parameters based on processing load."""
        with self.shm_lock:
            pipeline_depth = len(self.active_shm_blocks)

        # Get the number of tasks waiting to even be started
        pending_analysis_tasks = self.salience_task_queue.qsize()
        total_backlog = pipeline_depth + pending_analysis_tasks

        if total_backlog > self.max_pipeline_depth:
            # We are backlogged. Increase chunk size to reduce analysis overhead per second of video.
            if self.config['chunk_seconds'] < 20:
                self.config['chunk_seconds'] += 1
                print(f"PIPELINE BACKLOG DETECTED (depth: {pipeline_depth}). Increasing chunk size to {self.config['chunk_seconds']}s.")
        elif total_backlog < self.min_pipeline_depth:
            # We have spare capacity.
            pass
    def start_workers(self):
        """Creates and starts all the decoupled processes and threads."""
        # --- Start Processes ---
        # Capture process - use OBS if preferred and available
        if self.config.get('prefer_obs') and OBS_AVAILABLE:
            capture_target = obs_capture_process_loop
            capture_name = "OBS_Capture"
        else:
            capture_target = capture_process_loop
            capture_name = "Legacy_Capture"

        process_map = {
            capture_name: (capture_target, (self.raw_frame_queue, self.config, self.shutdown_event)),
            "Triage": (dispatcher_loop, (self.raw_frame_queue, self.salience_task_queue, self.shm_notification_queue, self.config, self.shutdown_event)),
        }
        for name, (target, args) in process_map.items():
            p = Process(target=target, args=args, name=name, daemon=True)
            self.processes.append(p)
        for i in range(self.num_salience_workers):
            # --- MODIFIED: The arguments are now simpler and more generic ---
            p = Process(target=sal_wo.analysis_worker_loop, 
                        args=(self.salience_task_queue, self.salience_results_queue, 
                        self.config, self.config['output_path'], self.shutdown_event,
                        self.progress_queue), 
                        name=f"Salience-{i}", daemon=True)
            self.processes.append(p)
        for i in range(self.num_encoding_workers):
            p = Process(target=video_encoding_worker_loop, args=(self.encoding_task_queue, self.encoding_results_queue, self.config['output_path']), name=f"Encoder-{i}", daemon=True)
            self.processes.append(p)
        for p in self.processes:
            p.start()

        # Input capture thread (only if input_target is set)
        input_target = self.config.get('input_target')
        if input_target is not None:
            print("-> Starting background threads (Input Capture)...")
            input_thread = Thread(
                target=hmi_utils.input_capture_worker,
                args=(self.persistence_manager.event_queue, input_target),
                daemon=True
            )
            self.threads.append(input_thread)
            input_thread.start()
            print(f"-> Input capture started for: {input_target if isinstance(input_target, str) else input_target.exe_name}")
        else:
            print("-> Input capture DISABLED (no target selected)")

    def _guardian_loop(self):
        """Monitors system resources and takes emergency action."""
        print("Guardian thread started. Monitoring system RAM.")
        while self.is_running:
            mem_percent = psutil.virtual_memory().percent
            
            if mem_percent > 95.0:
                print(f"CRITICAL: System memory at {mem_percent}%. Forcing immediate shutdown.")
                # This is the most drastic action. os._exit bypasses all cleanup.
                # In a real-world app, you might use a wrapper script to restart.
                os._exit(1)

            elif mem_percent > 90.0:
                print(f"WARNING: System memory at {mem_percent}%. Shedding pending analysis tasks.")
                blocks_to_drop = []
                with self.shm_lock:
                    for name, meta in self.active_shm_blocks.items():
                        # Only drop blocks that haven't even started analysis yet.
                        if meta['status'] == 'pending_analysis':
                            blocks_to_drop.append(name)
                
                if blocks_to_drop:
                    print(f"Guardian shedding {len(blocks_to_drop)} SHM blocks to free memory.")
                    for name in blocks_to_drop:
                        self.cleanup_shm_block(name)
            
            time.sleep(2) # Check every 2 seconds

    def shutdown(self):
        print("\nShutting down orchestrator...")
        if not self.is_running: return
        self.is_running = False
        print("-> Signaling all processes to exit via event...")
        self.shutdown_event.set()
        
        print("-> Sending shutdown signals to worker queues...")
        for _ in range(self.num_salience_workers): self.salience_task_queue.put(None)
        for _ in range(self.num_encoding_workers): self.encoding_task_queue.put(None)

        print("-> Waiting for worker processes to terminate...")
        for p in self.processes:
            p.join(timeout=5)
            if p.is_alive():
                print(f"WARNING: Process {p.name} ({p.pid}) did not terminate gracefully. Forcing.")
                p.terminate()
                
        self.video_encoder.shutdown_all()
        self.persistence_manager.shutdown()
        print("-> Cleaning up any orphaned SHM blocks...")
        with self.shm_lock:
            all_names = list(self.active_shm_blocks.keys())
        for name in all_names:
            self.cleanup_shm_block(name)
        self.logger.finalize()
        print("-> Shutdown complete.")

# =====================================================================================
#  UNCHANGED CLASSES (VideoEncoder, AsyncPersistenceManager)
# =====================================================================================

class AsyncPersistenceManager:
    """
    Manages durable persistence using a JSONL journal and periodic, finalized
    Parquet chunks. This is a crash-proof design.
    """
    def __init__(self, output_dir: str, event_queue: Queue, chunk_event_threshold: int = 10000):
        self.output_dir = output_dir
        self.event_queue = event_queue
        self.chunk_threshold = chunk_event_threshold
        
        self.is_running = True
        self.thread = Thread(target=self._run_loop)
        
        self.schema = pa.schema([
            pa.field('event_id', pa.string()), pa.field('stream_type', pa.string()),
            pa.field('start_timestamp', pa.float64()), pa.field('delta_timestamp', pa.float64()),
            pa.field('payload_json', pa.string())
        ])
        
        # --- File Handles ---
        # The journal is our source of truth. It's opened in append mode.
        self.journal_path = os.path.join(self.output_dir, "events_stream.jsonl")
        self.journal_file = open(self.journal_path, "ab") # Append, binary
        
        # State for the current in-memory chunk
        self.current_chunk_events = []
        self.chunk_counter = 0
        
        print(f"Persistence manager initialized. Journaling to {self.journal_path}")

    def start(self):
        self.thread.daemon = True
        self.thread.start()

    def _run_loop(self):
        """Drains the queue, writes to the journal, and finalizes Parquet chunks when full."""
        while self.is_running or not self.event_queue.empty():
            try:
                # Block until an event arrives, with a timeout to allow periodic checks
                event = self.event_queue.get(timeout=0.1)
                
                # 1. Write to Journal (Source of Truth) - This is fast and safe.
                self.journal_file.write(orjson.dumps(event) + b'\n')
                
                # 2. Add to in-memory batch for the next Parquet chunk
                self.current_chunk_events.append(event)
                
                # 3. Check if it's time to finalize the current chunk
                if len(self.current_chunk_events) >= self.chunk_threshold:
                    self._finalize_chunk()

            except Exception: # Queue was empty, loop continues
                continue

    def _finalize_chunk(self):
        """Writes the current batch of events to a new, finalized Parquet file."""
        if not self.current_chunk_events:
            return

        chunk_path = os.path.join(self.output_dir, f"events_chunk_{self.chunk_counter:04d}.parquet")
        print(f"Finalizing chunk of {len(self.current_chunk_events)} events to {chunk_path}...")
        
        try:
            table = pa.Table.from_pylist(self.current_chunk_events, schema=self.schema)
            pq.write_table(table, chunk_path) # write_table opens, writes, and closes/finalizes in one go.
            
            # Success! Reset the chunk.
            self.current_chunk_events = []
            self.chunk_counter += 1
        except Exception as e:
            print(f"CRITICAL ERROR: Could not write Parquet chunk {chunk_path}. Data is still safe in journal. Error: {e}")

    def shutdown(self):
        """Signals the thread to stop and finalizes the last partial chunk."""
        print("Shutting down persistence manager...")
        self.is_running = False
        
        # Wait for the writer thread to finish processing the queue
        self.thread.join(timeout=2.0)
        
        # Finalize any remaining events in the last partial chunk
        self._finalize_chunk()
        
        # Close the journal file handle
        if self.journal_file:
            self.journal_file.close()
        print("Persistence manager shut down.")

# =====================================================================================
#  PAGED ORCHESTRATOR (NEW ARCHITECTURE)
# =====================================================================================

class PagedOrchestrator:
    """
    The Paged Orchestrator: Uses the GlobalScheduler for priority-based page processing.

    Key differences from legacy Orchestrator:
    1. Pages are small (1.5s instead of 12s chunks)
    2. Two-phase triage: coarse first, fine only for interesting pages
    3. Global scheduler prioritizes new page triage over refinement
    4. Boring pages evicted immediately after baseline encode
    5. Keyframe stills are ACTUALLY SAVED (via FrameArchiver)
    6. Baseline encoding uses proper frame decimation
    """

    def __init__(self, config):
        self.config = config
        self.is_running = True
        self.shutdown_event = Event()
        self.preemption_flag = Event()  # Set when new page arrives during refinement

        # Queues
        self.raw_frame_queue_max_size = config['capture_fps'] * 3
        self.raw_frame_queue = Queue(maxsize=self.raw_frame_queue_max_size)
        self.page_queue = Queue()           # Pages from dispatcher
        self.baseline_queue = Queue()       # Decimated frames for baseline encoding
        self.triage_work_queue = Queue()    # Work items for triage worker
        self.triage_result_queue = Queue()  # Results from triage worker
        self.encoding_task_queue = Queue()
        self.encoding_results_queue = Queue()

        # Global scheduler
        self.scheduler = GlobalScheduler()

        # Components
        self.persistence_manager = AsyncPersistenceManager(config['output_path'], Queue())
        self.frame_archiver = hmi_utils.FrameArchiver(config['output_path'])
        self.video_encoder = hmi_utils.VideoEncoder(config['output_path'])
        self.guardian_thread = Thread(target=self._guardian_loop, daemon=True)
        self.baseline_encoder_thread = Thread(target=self._baseline_encoder_loop, daemon=True)

        self.processes = []
        self.threads = []
        self.logger = ProgressLogger()

        # Stats
        self.last_log_time = time.time()
        self.log_interval_seconds = 2.0

    def start(self):
        """Start all worker processes and the main orchestration loop."""
        self._start_workers()
        self.persistence_manager.start()
        self.guardian_thread.start()
        self.baseline_encoder_thread.start()

        print("Starting paged orchestration loop... Press Ctrl+C to stop.")
        try:
            while self.is_running:
                # 1. Receive new pages from dispatcher
                self._receive_new_pages()

                # 2. Dispatch work to triage worker based on scheduler priority
                self._dispatch_scheduled_work()

                # 3. Process triage results
                self._process_triage_results()

                # 4. Check encoding results
                self._process_encoding_results()

                # 5. Log status
                self._log_status()

                time.sleep(0.008)  # ~120Hz orchestration loop

        except (KeyboardInterrupt, SystemExit):
            self.is_running = False
        finally:
            self.shutdown()

    def _receive_new_pages(self):
        """Receive new pages from dispatcher and register with scheduler."""
        pages_received = 0
        while not self.page_queue.empty():
            try:
                page_task = self.page_queue.get_nowait()
                page = page_task['page_metadata']

                # CRITICAL: Open SHM handle IMMEDIATELY to keep it alive on Windows!
                # On Windows, SharedMemory is destroyed when all handles are closed.
                # The dispatcher closes its handle after putting page on queue, so we
                # MUST open our handle before that happens.
                try:
                    shm_handle = shared_memory.SharedMemory(name=page.shm_name)
                except FileNotFoundError:
                    print(f"[Orchestrator] WARNING: SHM {page.shm_name} not found, page may have been lost")
                    continue

                # Register with scheduler, passing handle to keep SHM alive
                self.scheduler.register_page(page, shm_handle=shm_handle)
                pages_received += 1

            except queue.Empty:
                break

        # DISABLED: Preemption causes more problems than it solves with single worker
        # The worker can only do one thing at a time anyway, so preemption just
        # causes thrashing. Let refinement complete, then triage will run.
        # TODO: Re-enable with smarter logic once basic flow works
        pass

    def _dispatch_scheduled_work(self):
        """Get next work item from scheduler and dispatch to triage worker."""
        # FIRST: Process ALL pending evictions (they're fast, no worker needed)
        # This prevents eviction starvation from constant triage/refinement work
        self._process_pending_evictions()

        # Only allow ONE page in TRIAGING or REFINING at a time (single worker)
        # But prioritize refinement if interesting pages are piling up
        with self.scheduler.lock:
            states = [p.state for p in self.scheduler.pages.values()]
            triaging_count = sum(1 for s in states if s == PageState.TRIAGING)
            refining_count = sum(1 for s in states if s == PageState.REFINING)
            interesting_waiting = sum(1 for s in states if s == PageState.INTERESTING_RETAINED)

        # Skip if worker is already busy
        if triaging_count >= 1 or refining_count >= 1:
            return

        # If interesting pages are piling up, force refinement by skipping triage
        # This prevents refinement starvation from constant new page arrivals
        force_refinement = interesting_waiting >= 3

        # THEN: Get next work (possibly forcing refinement)
        work = self.scheduler.get_next_work(prefer_refinement=force_refinement)
        if work is None:
            return

        page = self.scheduler.pages.get(work.page_id)
        if page is None:
            return

        if work.work_type == WorkType.COARSE_TRIAGE:
            task = {
                'work_type': 'COARSE_TRIAGE',
                'page_id': page.page_id,
                'shm_name': page.shm_name,
                'shape': page.shape,
                'dtype': page.dtype,
                'timestamps': page.timestamps,
            }
            self.triage_work_queue.put(task)

        elif work.work_type == WorkType.FINE_REFINEMENT:
            task = {
                'work_type': 'FINE_REFINEMENT',
                'page_id': page.page_id,
                'shm_name': page.shm_name,
                'shape': page.shape,
                'dtype': page.dtype,
                'timestamps': page.timestamps,
                'hot_regions': page.hot_regions or [],
            }
            self.triage_work_queue.put(task)

        elif work.work_type == WorkType.EVICTION:
            # Shouldn't get here often now, but handle just in case
            self._evict_page(work.page_id)

    def _process_pending_evictions(self):
        """Process all pages in BORING_EVICTING state without using the priority queue."""
        with self.scheduler.lock:
            pages_to_evict = [
                page_id for page_id, page in self.scheduler.pages.items()
                if page.state == PageState.BORING_EVICTING
            ]

        for page_id in pages_to_evict:
            self._evict_page(page_id)

    def _evict_page(self, page_id: str):
        """Evict a single page and clean up its SHM."""
        shm = self.scheduler.evict_page(page_id)
        if shm:
            try:
                shm.close()
                shm.unlink()
            except Exception:
                pass

    def _process_triage_results(self):
        """Process results from triage worker."""
        while not self.triage_result_queue.empty():
            try:
                result = self.triage_result_queue.get_nowait()
                work_type = result.get('work_type')
                page_id = result.get('page_id')

                if work_type == 'COARSE_TRIAGE':
                    is_interesting = result.get('is_interesting', False)
                    max_novelty = result.get('max_novelty', 0.0)
                    hot_regions = result.get('hot_regions', [])

                    if is_interesting:
                        # Schedule for fine refinement
                        page = self.scheduler.pages.get(page_id)
                        if page:
                            page.hot_regions = hot_regions
                        self.scheduler.schedule_refinement(page_id, max_novelty)
                        self.logger.log("triage", f"Page {page_id[:12]} INTERESTING (novelty={max_novelty:.2f})")
                    else:
                        # Schedule for eviction (boring page)
                        self.scheduler.schedule_eviction(page_id)
                        self.logger.log("triage", f"Page {page_id[:12]} boring, evicting")

                elif work_type == 'FINE_REFINEMENT':
                    keyframes = result.get('keyframes', [])
                    was_preempted = result.get('was_preempted', False)

                    if was_preempted:
                        # Pause and reschedule
                        self.scheduler.pause_refinement(page_id)
                        self.preemption_flag.clear()
                        self.logger.log("preempt", f"Page {page_id[:12]} refinement preempted")
                    else:
                        # Refinement complete
                        self.scheduler.complete_refinement(page_id)

                        if keyframes:
                            self._handle_keyframes(result, keyframes)
                        else:
                            # No keyframes found, just evict
                            self.scheduler.schedule_eviction(page_id)

            except queue.Empty:
                break

    def _handle_keyframes(self, result, keyframes):
        """
        Handle keyframes from refinement:
        1. Save stills via FrameArchiver (THE MISSING FUNCTIONALITY!)
        2. Dispatch high-quality video encoding
        3. Log events to persistence manager
        """
        page_id = result['page_id']
        shm_name = result['shm_name']
        shape = result['shape']
        dtype = result['dtype']
        timestamps = result['timestamps']

        try:
            # Open SHM to access frames
            shm = shared_memory.SharedMemory(name=shm_name)
            buffer = np.ndarray(shape, dtype=dtype, buffer=shm.buf)
            timestamps_arr = np.array(timestamps)

            # --- SAVE KEYFRAME STILLS (THE MISSING PIECE!) ---
            for kf in keyframes:
                frame_idx = kf.get('frame_index')
                if frame_idx is not None and 0 <= frame_idx < len(buffer):
                    frame_data = buffer[frame_idx]
                    event_type = kf.get('type', 'KEYFRAME')
                    timestamp = kf.get('timestamp', timestamps_arr[frame_idx])

                    # ACTUALLY SAVE THE STILL!
                    saved_path = self.frame_archiver.save_frame(frame_data, timestamp, event_type)

                    if saved_path:
                        # Log the keyframe event with the saved path
                        event = {
                            "event_id": str(uuid.uuid4()),
                            "stream_type": event_type,
                            "start_timestamp": timestamp,
                            "delta_timestamp": 0.0,
                            "payload_json": orjson.dumps({
                                "z_score": kf.get('z_score'),
                                "score": kf.get('score'),
                                "reason": kf.get('reason'),
                                "frame_index": frame_idx,
                                "still_path": saved_path,
                            }).decode('utf-8')
                        }
                        self.persistence_manager.event_queue.put(event)

            # --- DISPATCH HIGH-QUALITY VIDEO ENCODING ---
            # Merge keyframe intervals with padding
            padding = self.config.get('keyframe_padding_seconds', 0.5)
            intervals = []
            for kf in keyframes:
                ts = kf.get('timestamp')
                if ts is not None:
                    intervals.append([ts - padding, ts + padding])

            if intervals:
                intervals.sort(key=lambda x: x[0])
                merged = [intervals[0]]
                for interval in intervals[1:]:
                    if interval[0] <= merged[-1][1]:
                        merged[-1][1] = max(merged[-1][1], interval[1])
                    else:
                        merged.append(interval)

                # Dispatch encoding jobs
                for start_time, end_time in merged:
                    mask = (timestamps_arr >= start_time) & (timestamps_arr <= end_time)
                    indices = np.where(mask)[0]
                    if len(indices) > 0:
                        frames_to_encode = buffer[indices].copy()  # Copy before closing SHM
                        timestamps_to_encode = timestamps_arr[indices]
                        self.encoding_task_queue.put({
                            'frames': frames_to_encode,
                            'timestamps': timestamps_to_encode,
                            'quality': 'HIGH',
                        })

            shm.close()

            # Schedule eviction now that we've handled the keyframes
            self.scheduler.schedule_eviction(page_id)

        except FileNotFoundError:
            print(f"WARNING: SHM {shm_name} not found for keyframe handling")
        except Exception as e:
            print(f"ERROR handling keyframes: {e}")
            import traceback; traceback.print_exc()

    def _process_encoding_results(self):
        """Check for completed encoding jobs."""
        while not self.encoding_results_queue.empty():
            try:
                self.encoding_results_queue.get_nowait()
                # Just count completions for now
            except queue.Empty:
                break

    def _baseline_encoder_loop(self):
        """Background thread that encodes decimated page frames to ONE continuous file."""
        print("Baseline encoder thread started")

        # Continuous encoder state
        ffmpeg_proc = None
        ffmpeg_stdin = None
        current_dimensions = None

        def start_encoder(dimensions):
            """Start a new continuous encoder for the given dimensions."""
            nonlocal ffmpeg_proc, ffmpeg_stdin, current_dimensions
            proc, stdin, stderr = self.video_encoder.start_continuous_encode(
                dimensions=dimensions,
                fps=BASELINE_TARGET_FPS
            )
            ffmpeg_proc = proc
            ffmpeg_stdin = stdin
            current_dimensions = dimensions
            return proc, stdin

        def stop_encoder():
            """Gracefully stop the current encoder."""
            nonlocal ffmpeg_proc, ffmpeg_stdin, current_dimensions
            if ffmpeg_stdin:
                try:
                    ffmpeg_stdin.close()
                except Exception:
                    pass
            if ffmpeg_proc:
                try:
                    ffmpeg_proc.wait(timeout=5.0)
                except Exception:
                    ffmpeg_proc.kill()
            ffmpeg_proc = None
            ffmpeg_stdin = None
            current_dimensions = None

        try:
            while self.is_running or not self.baseline_queue.empty():
                try:
                    task = self.baseline_queue.get(timeout=0.5)

                    # Read frames from SHM
                    shm_name = task['shm_name']
                    shape = task['shape']
                    dtype = np.dtype(task['dtype'])
                    timestamps = task['timestamps']

                    try:
                        # Extract decimated frames
                        decimated_frames, decimated_timestamps = get_decimated_frames_for_baseline(
                            shm_name=shm_name,
                            shape=shape,
                            dtype=dtype,
                            timestamps=timestamps,
                            target_fps=BASELINE_TARGET_FPS
                        )

                        if len(decimated_frames) == 0:
                            continue

                        frame_dimensions = decimated_frames[0].shape

                        # Start or restart encoder if dimensions changed
                        if ffmpeg_proc is None or frame_dimensions != current_dimensions:
                            if ffmpeg_proc is not None:
                                print(f"[Baseline] Dimension change: {current_dimensions} -> {frame_dimensions}")
                                stop_encoder()
                            start_encoder(frame_dimensions)

                        # Write frames to the continuous encoder
                        if ffmpeg_stdin:
                            try:
                                frame_bytes = np.ascontiguousarray(decimated_frames).tobytes()
                                ffmpeg_stdin.write(frame_bytes)
                            except (BrokenPipeError, OSError):
                                print("[Baseline] Pipe broke, restarting encoder")
                                stop_encoder()
                                start_encoder(frame_dimensions)
                                # Retry write
                                try:
                                    frame_bytes = np.ascontiguousarray(decimated_frames).tobytes()
                                    ffmpeg_stdin.write(frame_bytes)
                                except Exception:
                                    pass

                    except FileNotFoundError:
                        # SHM was already evicted, skip this page
                        pass

                except queue.Empty:
                    continue
                except Exception as e:
                    print(f"Baseline encoder error: {e}")
                    import traceback; traceback.print_exc()

        finally:
            # Clean up encoder on exit
            stop_encoder()
            print("Baseline encoder thread finished")

    def _log_status(self):
        """Log pipeline status periodically."""
        now = time.time()
        if (now - self.last_log_time) < self.log_interval_seconds:
            return

        stats = self.scheduler.get_stats()
        state_dist = stats.get('state_distribution', {})
        # Compact state names
        state_abbrev = {
            'PENDING_TRIAGE': 'PendT',
            'TRIAGING': 'Triage',
            'BORING_EVICTING': 'BorEvict',
            'INTERESTING_RETAINED': 'IntRet',
            'REFINING': 'Refine',
            'ENCODING_INTERESTING': 'EncInt',
            'EVICTED': 'Evict',
        }
        state_str = " ".join(f"{state_abbrev.get(k, k)}:{v}" for k, v in state_dist.items() if v > 0)

        status_line = (
            f"[Paged Pipeline] "
            f"Active: {stats['active_pages']} | "
            f"Pending: {stats['pending_work']} | "
            f"Triaged: {stats['pages_triaged']} | "
            f"Evicted: {stats['pages_evicted_boring']} | "
            f"Interesting: {stats['pages_retained_interesting']} | "
            f"States: {state_str}"
        )
        self.logger.log("status", status_line)
        self.last_log_time = now

    def _guardian_loop(self):
        """Monitor system resources and take emergency action."""
        print("Guardian thread started")
        while self.is_running:
            mem_percent = psutil.virtual_memory().percent

            if mem_percent > 95.0:
                print(f"CRITICAL: Memory at {mem_percent}%. Forcing shutdown.")
                os._exit(1)

            elif mem_percent > 90.0:
                print(f"WARNING: Memory at {mem_percent}%. Shedding boring pages.")
                # Evict all pending pages
                with self.scheduler.lock:
                    for page_id, page in list(self.scheduler.pages.items()):
                        if page.state == PageState.PENDING_TRIAGE:
                            self.scheduler.schedule_eviction(page_id)

            time.sleep(2)

    def _start_workers(self):
        """Start all worker processes."""
        # Capture process - use OBS if preferred and available
        if self.config.get('prefer_obs') and OBS_AVAILABLE:
            capture_proc = Process(
                target=obs_capture_process_loop,
                args=(self.raw_frame_queue, self.config, self.shutdown_event),
                name="OBS_Capture",
                daemon=True
            )
        else:
            capture_proc = Process(
                target=capture_process_loop,
                args=(self.raw_frame_queue, self.config, self.shutdown_event),
                name="Legacy_Capture",
                daemon=True
            )
        self.processes.append(capture_proc)

        # Paged dispatcher process
        dispatcher_proc = Process(
            target=dispatcher_loop,
            args=(self.raw_frame_queue, self.page_queue, self.baseline_queue,
                  self.config, self.shutdown_event),
            name="PagedDispatcher",
            daemon=True
        )
        self.processes.append(dispatcher_proc)

        # Triage worker (uses two-phase triage)
        triage_proc = Process(
            target=sal_wo.paged_triage_worker_loop,
            args=(self.triage_work_queue, self.triage_result_queue,
                  self.config, self.config['output_path'],
                  self.shutdown_event, self.preemption_flag),
            name="TriageWorker",
            daemon=True
        )
        self.processes.append(triage_proc)

        # Video encoding workers
        for i in range(2):
            enc_proc = Process(
                target=video_encoding_worker_loop,
                args=(self.encoding_task_queue, self.encoding_results_queue,
                      self.config['output_path']),
                name=f"Encoder-{i}",
                daemon=True
            )
            self.processes.append(enc_proc)

        # Start all processes
        for p in self.processes:
            p.start()

        # Input capture thread (only if input_target is set)
        input_target = self.config.get('input_target')
        if input_target is not None:
            input_thread = Thread(
                target=hmi_utils.input_capture_worker,
                args=(self.persistence_manager.event_queue, input_target),
                daemon=True
            )
            self.threads.append(input_thread)
            input_thread.start()
            print(f"-> Input capture started for: {input_target if isinstance(input_target, str) else input_target.exe_name}")
        else:
            print("-> Input capture DISABLED (no target selected)")

    def shutdown(self):
        """Graceful shutdown."""
        print("\nShutting down paged orchestrator...")
        if not self.is_running:
            return
        self.is_running = False

        print("-> Signaling shutdown...")
        self.shutdown_event.set()

        print("-> Sending shutdown signals...")
        self.triage_work_queue.put(None)
        for _ in range(2):
            self.encoding_task_queue.put(None)

        print("-> Waiting for processes...")
        for p in self.processes:
            p.join(timeout=5)
            if p.is_alive():
                print(f"WARNING: {p.name} did not terminate, forcing")
                p.terminate()

        self.video_encoder.shutdown_all()
        self.persistence_manager.shutdown()

        # Cleanup remaining SHM blocks
        print("-> Cleaning up pages...")
        with self.scheduler.lock:
            for page_id in list(self.scheduler.pages.keys()):
                shm = self.scheduler.evict_page(page_id)
                if shm:
                    try:
                        shm.close()
                        shm.unlink()
                    except Exception:
                        pass

        self.logger.finalize()
        print("-> Shutdown complete.")


# =====================================================================================
#  MAIN ENTRY POINT
# =====================================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Salience-driven gameplay capture",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Capture Modes:
  --obs          Use OBS Virtual Camera (RECOMMENDED - anti-cheat safe)
  --window NAME  Use legacy window title matching (fragile, breaks on title change)
  --legacy       Use old 12s chunk architecture instead of 1.5s pages

Input Capture:
  --no-input     Disable keyboard/mouse input capture
  --input-exe    Executable name/path to track (skips selector)

Examples:
  python recording_stuff.py --obs                    # OBS + window selector
  python recording_stuff.py --obs --input-exe gzdoom.exe  # Skip selector
  python recording_stuff.py --obs --no-input         # Video only, no inputs
  python recording_stuff.py --window "Sulfur"        # Legacy window capture

Setup OBS first:
  python obs_setup.py --install      # Install OBS
  python obs_setup.py --setup-vcam   # Set up Virtual Camera
        """
    )
    parser.add_argument("--legacy", action="store_true",
                        help="Use legacy chunk-based orchestrator (12s chunks)")
    parser.add_argument("--window", type=str, default=None,
                        help="Window name for legacy capture (fragile!)")
    parser.add_argument("--obs", action="store_true",
                        help="Use OBS Virtual Camera capture (recommended)")
    parser.add_argument("--obs-device", type=int, default=None,
                        help="Video device index for OBS Virtual Camera")
    parser.add_argument("--no-obs", action="store_true",
                        help="Force legacy capture even if OBS is available")
    parser.add_argument("--no-input", action="store_true",
                        help="Disable keyboard/mouse input capture")
    parser.add_argument("--input-exe", type=str, default=None,
                        help="Executable name/path for input capture (skips selector)")
    parser.add_argument("--debug-windows", action="store_true",
                        help="Debug window enumeration for input selector")
    parser.add_argument("--log", type=str, default=None, metavar="FILE",
                        help="Write all output to log file (keeps terminal interactive)")
    args = parser.parse_args()

    # Set up logging to file if requested
    if args.log:
        import sys

        class TeeWriter:
            """Write to both terminal and log file."""
            def __init__(self, terminal, logfile):
                self.terminal = terminal
                self.logfile = logfile

            def write(self, message):
                self.terminal.write(message)
                self.terminal.flush()
                self.logfile.write(message)
                self.logfile.flush()

            def flush(self):
                self.terminal.flush()
                self.logfile.flush()

        log_file = open(args.log, 'w', encoding='utf-8')
        sys.stdout = TeeWriter(sys.__stdout__, log_file)
        sys.stderr = TeeWriter(sys.__stderr__, log_file)
        print(f"Logging to: {args.log}")

    # Determine capture mode
    use_obs = False
    window_name = args.window or RECORDING_WINDOW_NAME

    if args.obs:
        if not OBS_AVAILABLE:
            print("ERROR: --obs requested but obs_capture module not available")
            print("       Install opencv-python: pip install opencv-python")
            exit(1)
        use_obs = True
    elif args.no_obs:
        use_obs = False
    elif OBS_AVAILABLE and not args.window:
        # Default to OBS if available and no window specified
        use_obs = True

    # Validate we have a capture source
    if not use_obs and not window_name and not CAPTURE_REGION:
        print("ERROR: You must specify a capture source:")
        print("  --obs              Use OBS Virtual Camera")
        print("  --window NAME      Use legacy window capture")
        exit(1)

    os.makedirs(OUTPUT_PATH, exist_ok=True)

    # --- INPUT CAPTURE TARGET SELECTION ---
    input_target = None  # WindowIdentifier or string for input capture
    if args.no_input:
        print("\nInput capture disabled (--no-input)")
        input_target = None
    elif args.input_exe:
        # User specified executable directly, skip selector
        input_target = args.input_exe
        print(f"\nInput capture target: {input_target}")
    else:
        # Interactive window selector
        print("\n" + "=" * 60)
        print("  INPUT CAPTURE SETUP")
        print("  Select which window to track keyboard/mouse inputs for.")
        print("  (This is separate from OBS video capture)")
        print("=" * 60)

        input_target = hmi_utils.select_target_window(debug=args.debug_windows)
        if input_target is None:
            print("\nNo window selected. Input capture will be DISABLED.")
            print("(You can still capture video, just no keyboard/mouse events)")
            response = input("Continue without input capture? [Y/n]: ").strip().lower()
            if response == 'n':
                print("Aborted.")
                exit(0)

    config = {
        "window_name": window_name,
        "region": CAPTURE_REGION,
        "output_path": OUTPUT_PATH,
        "capture_fps": CAPTURE_FPS,
        "keyframe_padding_seconds": KEYFRAME_PADDING_SECONDS,
        "salience_kernels": SALIENCE_KERNELS,
        "salience_strategy": SALIENCE_STRATEGY,
        "scheduled_hyperparams": SCHEDULED_HYPERPARAMS,
        **STRATEGY_CONFIGS[SALIENCE_STRATEGY],

        # Paged memory model config
        "page_seconds": PAGE_SECONDS,
        "coarse_triage_sentinels": COARSE_TRIAGE_SENTINELS,
        "coarse_triage_novelty_threshold": COARSE_TRIAGE_NOVELTY_THRESHOLD,
        "fine_refinement_max_p": FINE_REFINEMENT_MAX_P,

        # Legacy compatibility
        "chunk_seconds": PAGE_SECONDS if not args.legacy else CHUNK_SECONDS,
        "triage_threshold": TRIAGE_THRESHOLD,

        # OBS capture config
        "prefer_obs": use_obs,
        "require_obs": args.obs,  # If --obs was explicitly passed, require it (no silent fallback)
        "obs_device_index": args.obs_device,

        # Input capture config (WindowIdentifier, string, or None to disable)
        "input_target": input_target,
    }

    orchestrator = None
    try:
        # Print mode info
        print("=" * 60)
        if args.legacy:
            print("  LEGACY MODE: Chunk-based orchestrator (12s chunks)")
        else:
            print("  PAGED MODE: Paged orchestrator (1.5s pages)")
            print("  - Two-phase triage (coarse -> fine)")
            print("  - Priority scheduling (triage > refinement)")
            print("  - Keyframe stills saved to ./stills/")

        if use_obs:
            print("  VIDEO: OBS Virtual Camera (anti-cheat safe)")
            if args.obs:
                print("         (explicitly requested with --obs, NO fallback)")
            if args.obs_device is not None:
                print(f"         Device index: {args.obs_device}")
            print("         Make sure OBS is running with Virtual Camera started!")
        else:
            print(f"  VIDEO: Legacy window matching (fragile!)")
            print(f"         Window: '{window_name}'")

        if input_target is None:
            print("  INPUT: DISABLED (no keyboard/mouse capture)")
        elif isinstance(input_target, hmi_utils.WindowIdentifier):
            print(f"  INPUT: {input_target.exe_name} (PID {input_target.pid})")
            print(f"         Stable ID - survives window title changes")
        else:
            print(f"  INPUT: {input_target}")
        print("=" * 60)
        print()

        if args.legacy:
            orchestrator = Orchestrator(config)
        else:
            orchestrator = PagedOrchestrator(config)

        orchestrator.start()

    except WindowNotFoundError as e:
        print(f"FATAL STARTUP ERROR: {e}")
    except Exception as e:
        print(f"An unexpected error occurred in the main block: {e}")
        import traceback; traceback.print_exc()
        if orchestrator:
            orchestrator.shutdown()