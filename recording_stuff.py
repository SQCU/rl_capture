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

# --- Configuration ---
RECORDING_WINDOW_NAME = "ATLYSS"
CAPTURE_REGION = None
OUTPUT_PATH = f"./capture_run_{int(time.time())}"
CHUNK_SECONDS = 10  # How many seconds of frames to triage at a time
CAPTURE_FPS = 60
UNHANDLED_SHM_DEADLINE_SECONDS = 150.0 # Time before we declare a block abandoned
MAX_PIPELINE_DEPTH = 15
MIN_PIPELINE_DEPTH = 5
KEYFRAME_PADDING_SECONDS = 0.5 # How many seconds before and after a keyframe to include in a clip
# --- NEW: Triage Configuration ---
TRIAGE_THRESHOLD = 0.1 # A starting point for max_delta to trigger deep analysis

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
        "sustain_value": 0.1,    # ...and ramp down to the normal, more sensitive value.
        "warmup_steps": 4,       # Hold for 4 chunks.
        "ramp_steps": 16,        # Ramp down over the next 8 chunks.
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
                    raw_frame_queue.put_nowait((start_time, frame_rgb))
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

def _stage_and_dispatch_chunk(frames: list, timestamps: list, salience_task_queue: Queue, shm_notification_queue: Queue):
    """
    Handles the creation of a shared memory block, copying frame data into it,
    and dispatching the necessary tasks to the salience workers and orchestrator.
    """
    try:
        # 1. Prepare SHM block parameters
        buffer_shape = (len(frames), *frames[0].shape)
        buffer_size = int(np.prod(buffer_shape) * np.dtype(np.uint8).itemsize)
        shm_name = f"salience_chunk_{uuid.uuid4()}"

        # 2. Create the block and copy data
        shm = shared_memory.SharedMemory(name=shm_name, create=True, size=buffer_size)
        shm_buffer = np.ndarray(buffer_shape, dtype=np.uint8, buffer=shm.buf)
        np.copyto(shm_buffer, np.array(frames))

        # 3. Create the task for the salience workers
        task = {
            "shm_name": shm_name,
            "shape": buffer_shape,
            "dtype": np.uint8,
            "timestamps": list(timestamps)
        }
        
        # 4. Dispatch tasks to the respective queues
        salience_task_queue.put(task)
        shm_notification_queue.put(shm_name)

        # 5. Close our local handle. The Orchestrator now owns the block.
        shm.close()

    except Exception as e:
        print(f"[Dispatcher] CRITICAL ERROR during chunk dispatch: {e}")
        # Attempt to clean up a partially created SHM block if something went wrong
        if 'shm' in locals() and shm:
            shm.close()
            try:
                shm.unlink() # This might fail if it was never fully created, that's okay.
            except FileNotFoundError:
                pass

def dispatcher_loop(raw_frame_queue: Queue, salience_task_queue: Queue, shm_notification_queue: Queue, config: dict, shutdown_event: Event):
    """
    The Dispatcher: A "dumb" process that only buffers frames, stages them
    to shared memory, and puts a generic analysis task on the queue.
    It performs ZERO analysis itself.
    """
    worker_pid = os.getpid()
    print(f"[{worker_pid}] Dispatcher started.")

    frames_per_chunk = int(config['chunk_seconds'] * config['capture_fps'])
    frame_buffer = deque(maxlen=frames_per_chunk)

    while not shutdown_event.is_set():
        try:
            timestamp, frame = raw_frame_queue.get(timeout=0.1)
            frame_buffer.append((timestamp, frame))

            # --- MODIFIED: The logic is now radically simpler ---
            if len(frame_buffer) == frames_per_chunk:
                #print(f"[{worker_pid}] Dispatcher: Staging {len(frame_buffer)}-frame chunk for analysis.")
                
                timestamps, frames = zip(*frame_buffer)
                
                # The dispatcher's ONLY job is to stage the data and create a generic task.
                _stage_and_dispatch_chunk(
                    frames=list(frames), 
                    timestamps=list(timestamps), 
                    salience_task_queue=salience_task_queue, 
                    shm_notification_queue=shm_notification_queue
                )
                
                frame_buffer.clear()

        except queue.Empty:
            continue

    print(f"[{worker_pid}] Dispatcher finished.")

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
        self.raw_frame_queue = Queue(maxsize=config['capture_fps'] * 3) # Buffer 3s of raw frames
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
        process_map = {
            "Capture": (capture_process_loop, (self.raw_frame_queue, self.config, self.shutdown_event)),
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

        print("-> Starting background threads (Input Capture)...")
        input_thread = Thread(target=hmi_utils.input_capture_worker, args=(self.persistence_manager.event_queue, self.config['window_name']), daemon=True)
        self.threads.append(input_thread)
        input_thread.start()

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
#  MAIN ENTRY POINT
# =====================================================================================

if __name__ == "__main__":
    if RECORDING_WINDOW_NAME is None and CAPTURE_REGION is None:
        print("ERROR: You must set RECORDING_WINDOW_NAME or CAPTURE_REGION.")
        exit(1)

    os.makedirs(OUTPUT_PATH, exist_ok=True)
    
    config = {
        "window_name": RECORDING_WINDOW_NAME,
        "region": CAPTURE_REGION,
        "output_path": OUTPUT_PATH,
        "chunk_seconds": CHUNK_SECONDS,
        "capture_fps": CAPTURE_FPS,
        "triage_threshold": TRIAGE_THRESHOLD,
        "keyframe_padding_seconds": KEYFRAME_PADDING_SECONDS, 
        "salience_kernels": SALIENCE_KERNELS, 
        "salience_strategy": SALIENCE_STRATEGY,
        "scheduled_hyperparams": SCHEDULED_HYPERPARAMS,
        **STRATEGY_CONFIGS[SALIENCE_STRATEGY]
    }

    orchestrator = None
    try:
        orchestrator = Orchestrator(config)
        orchestrator.start()
    except WindowNotFoundError as e:
        print(f"FATAL STARTUP ERROR: {e}")
    except Exception as e:
        print(f"An unexpected error occurred in the main block: {e}")
        # In case of an early crash, we still want to try to shut down
        if orchestrator:
            orchestrator.shutdown()