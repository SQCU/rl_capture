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
import numpy as np
import subprocess
import queue 
from multiprocessing import Process, Queue, shared_memory, Event
from threading import Thread, Lock
from PIL import Image
from collections import deque

# We now need one of the salience workers in the Triage process itself
import salience_workers as sal_wo
from window_utils import get_window_finder, WindowNotFoundError

# --- Configuration ---
RECORDING_WINDOW_NAME = "ToramOnline"
CAPTURE_REGION = None
OUTPUT_PATH = f"./capture_run_{int(time.time())}"
CHUNK_SECONDS = 5  # How many seconds of frames to triage at a time
CAPTURE_FPS = 30
# --- NEW: Triage Configuration ---
TRIAGE_THRESHOLD = 0.1 # A starting point for max_delta to trigger deep analysis

# --- NEW: Salience Strategy Configuration ---
# Choose which algorithm to use for novelty detection.
# Options: "naive_cos_dissimilarity", "pca_mahalanobis"
SALIENCE_STRATEGY = "pca_mahalanobis" 
SALIENCE_KERNELS = ["siglip"] # Add "ocr" to this list to enable the OCR kernel

# --- NEW: Hyperparameters for the strategies ---
# These are passed to the salience worker config.
STRATEGY_CONFIGS = {
    "naive_cos_dissimilarity": {
        "z_score_threshold": 0.5, # The threshold for the simple Z-score method.
        "branching_factor": 8, # Override the worker's default of 8
        "min_exhaustive_size": 16, # Override the worker's default of 16 
    },
    "pca_mahalanobis": {
        "pca_n_components": 16,      # Max components (top-k guardrail).
        "pca_variance_threshold": 0.95, # The variance to capture (top-p).
        "novelty_z_score_threshold": 2.0, # Z-score threshold for the Mahalanobis scores themselves.
        "branching_factor": 8, # Override the worker's default of 8
        "min_exhaustive_size": 16, # Override the worker's default of 16
    }
}

### helper functions 


class FrameArchiver:
    """Handles the saving of individual frames as high-quality images."""
    def __init__(self, output_path: str):
        self.stills_path = os.path.join(output_path, "stills")
        os.makedirs(self.stills_path, exist_ok=True)
        print(f"Frame archiver will save stills to: {self.stills_path}")

    def save_frame(self, frame_data: np.ndarray, timestamp: float, event_type: str) -> str:
        """
        Saves a single frame to a PNG file.
        
        Returns:
            The relative path to the saved image file.
        """
        try:
            img = Image.fromarray(frame_data)
            filename = f"{event_type}_{timestamp:.4f}.png"
            filepath = os.path.join(self.stills_path, filename)
            img.save(filepath, "PNG")
            
            # Return the relative path for the metadata payload
            return os.path.join("stills", filename)
        except Exception as e:
            print(f"Error saving frame at timestamp {timestamp}: {e}")
            return None

def input_capture_worker(event_queue: Queue):
    """Listens for keyboard and mouse events and puts them on the event queue."""
    key_states = {} # Tracks the start time of key presses

    def on_press(key):
        key_id = str(key)
        if key_id not in key_states: # Avoid auto-repeat events
            key_states[key_id] = time.time()

    def on_release(key):
        key_id = str(key)
        if key_id in key_states:
            start_time = key_states.pop(key_id)
            end_time = time.time()
            event = {
                "event_id": str(uuid.uuid4()),
                "stream_type": "USER_INPUT",
                "start_timestamp": start_time,
                "delta_timestamp": end_time - start_time,
                "payload_json": orjson.dumps({"type": "key", "key": key_id}).decode('utf-8')
            }
            event_queue.put(event)

    # In a real app, you would add mouse listeners as well (on_click, on_move)
    keyboard_listener = pynput.keyboard.Listener(on_press=on_press, on_release=on_release)
    keyboard_listener.start()
    print("Input listener started.")
    keyboard_listener.join() # This blocks until the listener stops

def video_encoding_worker_loop(task_queue, result_queue, output_path):
    # This worker gets its own encoder instance
    video_encoder = VideoEncoder(output_path)
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
            # This is the crucial part: the WORKER waits, not the main loop
            return_code = proc.wait() 
            print(f"[{os.getpid()}] Encoding job finished with code {return_code}.")

        # Signal completion back to the main loop
        result_queue.put({'status': 'complete'}) # CHANGED

    print(f"[{os.getpid()}] Encoding worker finished.")

# =====================================================================================
#  NEW: DEDICATED PROCESS LOOPS
# =====================================================================================

def capture_process_loop(raw_frame_queue: Queue, config: dict, shutdown_event: Event):
    """
    The Collector: The fastest, dumbest part of the pipeline.
    Its only job is to grab frames and put them on a queue. It cannot be blocked.
    """
    print(f"[{os.getpid()}] Capture process started.")
    window_finder = get_window_finder()
    
    with mss.mss() as sct:
        while not shutdown_event.is_set():
            start_time = time.time()
            
            # Find the window to capture
            monitor = window_finder.find_window_by_title(config['window_name'])
            if monitor is None:
                time.sleep(0.5)
                continue

            img = sct.grab(monitor)
            frame_bgr = np.array(img)[:,:,:3]
            
            # --- FIX: Convert from BGR to RGB ---
            # This uses numpy slicing to reverse the order of the last dimension (the color channels)
            frame_rgb = frame_bgr[:, :, ::-1]

            try:
                # Put the raw frame and timestamp on the queue
                raw_frame_queue.put_nowait((start_time, frame_rgb))
            except queue.Full:
                # This is our safety valve. If the downstream Triage is falling behind,
                # we drop frames here to protect the capture process.
                print(f"[{os.getpid()}] WARNING: Raw frame queue is full. Triage is not keeping up. Dropping frame.")

            # Sleep to maintain target FPS
            elapsed = time.time() - start_time
            sleep_time = (1.0 / config['capture_fps']) - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
                
    print(f"[{os.getpid()}] Capture process finished.")

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
                print(f"[{worker_pid}] Dispatcher: Staging {len(frame_buffer)}-frame chunk for analysis.")
                
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
        self.num_salience_workers = 2
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
        
        # --- MODIFIED: This dictionary will now hold active SHM handles ---
        self.active_shm_blocks = {} # {shm_name: shm_instance}
        
        # High-level components
        self.persistence_manager = AsyncPersistenceManager(config['output_path'], Queue()) # Give it its own queue
        self.frame_archiver = FrameArchiver(config['output_path'])
        self.video_encoder = VideoEncoder(config['output_path'])
        
        self.processes = []
        self.threads = []

    def start(self):
        """Starts all worker processes and the main orchestration loop."""
        self.start_workers()
        self.persistence_manager.start()
        
        print("Starting main orchestration loop... Press Ctrl+C to stop.")
        try:
            while self.is_running:
                # --- Main Orchestration Loop ---
                # This loop is now very simple and fast.

                # --- NEW: Check for and claim new SHM blocks ---
                if not self.shm_notification_queue.empty():
                    shm_name = self.shm_notification_queue.get_nowait()
                    try:
                        # Open our own handle to the SHM block, keeping it alive.
                        shm = shared_memory.SharedMemory(name=shm_name)
                        self.active_shm_blocks[shm_name] = shm
                        print(f"Orchestrator: Registered and claimed SHM block {shm_name}.")
                    except FileNotFoundError:
                        print(f"Orchestrator WARNING: Triage notified of {shm_name}, but it disappeared before we could claim it.")

                # 1. Check for results from salience workers
                if not self.salience_results_queue.empty():
                    result = self.salience_results_queue.get_nowait()
                    self.process_salience_results(result)

                # 2. Check for completed video encodes
                if not self.encoding_results_queue.empty():
                    result = self.encoding_results_queue.get_nowait()
                    print(f"Orchestrator: Confirmed completion of encode job.")
                    # In a more complex system, you might log this completion.
                
                time.sleep(0.05)

        except (KeyboardInterrupt, SystemExit):
            self.is_running = False
        finally:
            self.shutdown()

    def process_salience_results(self, result: dict):
        """
        Processes keyframe events and search logs from a salience worker.
        """
        shm_name = result['shm_name']
        data = result.get('data', {})
        keyframes = data.get('keyframes', [])
        search_log = data.get('search_log', [])

        print(f"Orchestrator: Received {len(keyframes)} keyframes and {len(search_log)} search log entries from {shm_name}.")

        # --- NEW: Persist the search log for a detailed timeline ---
        if search_log:
            log_event = {
                "event_id": str(uuid.uuid4()),
                "stream_type": "SALIENCE_SEARCH_LOG",
                "start_timestamp": result['timestamps'][0],
                "delta_timestamp": result['timestamps'][-1] - result['timestamps'][0],
                "payload_json": orjson.dumps({'shm_name': shm_name, 'log': search_log}).decode('utf-8')
            }
            self.persistence_manager.event_queue.put(log_event)

        # --- Rule Application & Video Encoding (only if keyframes were found) ---
        if keyframes:
            # --- FIX: Aggregate and merge overlapping time intervals ---
            intervals = []
            for event in keyframes:
                intervals.append([event['timestamp'] - 2.0, event['timestamp'] + 2.0])
            
            # Sort intervals by start time
            intervals.sort(key=lambda x: x[0])
            
            merged_intervals = []
            if intervals:
                current_merge = intervals[0]
                for next_interval in intervals[1:]:
                    # If the next interval overlaps with the current one, merge them
                    if next_interval[0] <= current_merge[1]:
                        current_merge[1] = max(current_merge[1], next_interval[1])
                    else:
                        # Otherwise, the current merge is finished. Start a new one.
                        merged_intervals.append(current_merge)
                        current_merge = next_interval
                merged_intervals.append(current_merge)

            video_encode_jobs = []
            for start_time, end_time in merged_intervals:
                video_encode_jobs.append({
                    'start_time': start_time,
                    'end_time': end_time,
                    'quality': 'HIGH'
                })
            
            if video_encode_jobs:
                print(f"Orchestrator: Merged {len(keyframes)} events into {len(video_encode_jobs)} encoding job(s).")
                # We need to re-attach to the SHM to get frame data for the encoder
                try:
                    shm = shared_memory.SharedMemory(name=shm_name)
                    shape = result['shape']
                    timestamps = result['timestamps']
                    buffer = np.ndarray(shape, dtype=result['dtype'], buffer=shm.buf)
                    
                    # For now, process each job separately. Could be optimized.
                    for job in video_encode_jobs:
                        mask = (np.array(timestamps) >= job['start_time']) & (np.array(timestamps) <= job['end_time'])
                        indices = np.where(mask)[0]
                        if len(indices) > 0:
                            sorted_indices = sorted(indices, key=lambda i: timestamps[i])
                            frames_to_encode = buffer[sorted_indices]
                            timestamps_to_encode = np.array(timestamps)[sorted_indices]
                            
                            self.encoding_task_queue.put({
                                'frames': frames_to_encode,
                                'timestamps': timestamps_to_encode,
                                'quality': job['quality'],
                            })
                    shm.close()

                except FileNotFoundError:
                    print(f"WARNING: Could not find SHM block {shm_name} for encoding. It may have been cleaned up already.")

            # --- Cleanup ---
            # The salience worker is done, and we have dispatched encoding.
            # Now we can release our handle and unlink the memory.
            shm_to_clean = self.active_shm_blocks.pop(shm_name, None)
            if shm_to_clean:
                shm_to_clean.close()
                shm_to_clean.unlink()
                print(f"Orchestrator: Cleaned up SHM block {shm_name}.")
            else:
                print(f"Orchestrator WARNING: Tried to clean up {shm_name}, but it was not in our active registry.")
                # As a fallback, try to unlink it anyway in case of a state mismatch
                try:
                    shm_fallback = shared_memory.SharedMemory(name=shm_name)
                    shm_fallback.close()
                    shm_fallback.unlink()
                except FileNotFoundError:
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
                        args=(self.salience_task_queue, self.salience_results_queue, self.config, self.config['output_path'], self.shutdown_event), 
                        name=f"Salience-{i}", daemon=True)
            self.processes.append(p)
        for i in range(self.num_encoding_workers):
            p = Process(target=video_encoding_worker_loop, args=(self.encoding_task_queue, self.encoding_results_queue, self.config['output_path']), name=f"Encoder-{i}", daemon=True)
            self.processes.append(p)
        for p in self.processes:
            p.start()

        print("-> Starting background threads (Input Capture)...")
        input_thread = Thread(target=input_capture_worker, args=(self.persistence_manager.event_queue,), daemon=True)
        self.threads.append(input_thread)
        input_thread.start()

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
        for name, shm in self.active_shm_blocks.items():
            print(f"   - Cleaning {name}")
            shm.close()
            shm.unlink()
        print("-> Shutdown complete.")

# =====================================================================================
#  UNCHANGED CLASSES (VideoEncoder, AsyncPersistenceManager)
# =====================================================================================

class VideoEncoder:
    """
    Manages ffmpeg subprocesses for encoding video clips from raw frames.
    """
    def __init__(self, output_path: str):
        self.output_path = output_path
        os.makedirs(os.path.join(self.output_path, "videos"), exist_ok=True)
        self.active_processes = [] # Keep track of all processes
        self.lock = Lock() # To safely append to the list from multiple workers

    def start_encode_slice(self, frames: np.ndarray, timestamps: np.ndarray, quality: str) -> subprocess.Popen:
        """
        Launches an ffmpeg process asynchronously to encode a set of frames.
        
        Args:
            frames: A numpy array of shape (N, H, W, 3) with RGB frame data.
            timestamps: A numpy array of shape (N,) with the timestamp for each frame.
            quality: A string ('HIGH' or 'LOW') to determine encoding parameters.
        
        Returns:
            A Popen object representing the running ffmpeg process.
        """
        if len(frames) == 0:
            return None

        # Calculate a stable framerate for the clip
        durations = np.diff(timestamps)
        avg_fps = 1.0 / np.mean(durations) if len(durations) > 0 else 30.0
        
        height, width, _ = frames[0].shape
        output_filename = f"clip_{timestamps[0]:.2f}_{quality.lower()}.mp4"
        output_filepath = os.path.join(self.output_path, "videos", output_filename)

        # Configure ffmpeg parameters based on quality
        if quality == 'HIGH':
            crf = 20  # Lower CRF is higher quality
            preset = 'fast'
            resolution = f"{width}x{height}"
        else: # VERY_LOW
            crf = 35
            preset = 'ultrafast'
            # Downscale for low quality
            resolution = f"{width//4}x{height//4}"

        command = [
            'ffmpeg',
            '-y',  # Overwrite output file if it exists
            '-f', 'rawvideo',
            '-vcodec', 'rawvideo',
            '-s', f'{width}x{height}',  # Input size
            '-pix_fmt', 'rgb24',
            '-r', str(avg_fps), # Input framerate
            '-i', '-',  # The input comes from stdin
            '-c:v', 'libx264',
            '-preset', preset,
            '-crf', str(crf),
            '-vf', f'scale={resolution}', # Apply scaling
            '-pix_fmt', 'yuv420p',  # For compatibility
            output_filepath,
        ]

        # Start the process with stdin piped
        proc = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        with self.lock:
            self.active_processes.append(proc)
        
        # Write all frame data to the pipe in a separate thread to avoid blocking
        def pipe_frames():
            try:
                # Convert to bytes and write
                proc.stdin.write(frames.tobytes())
            except (IOError, BrokenPipeError):
                print(f"Warning: ffmpeg pipe broke for {output_filename}. The process may have terminated early.")
            finally:
                proc.stdin.close()
        
        Thread(target=pipe_frames).start()
        
        print(f"Started encoding {output_filename}...")
        return proc

    def start_continuous_encode(self, dimensions: tuple, fps: int, quality: str) -> (subprocess.Popen, io.BufferedWriter):
        """
        Launches a persistent ffmpeg process for a continuous, low-quality recording.
        
        Returns:
            A tuple containing the Popen object and its stdin pipe.
        """
        height, width, _ = dimensions
        output_filename = f"baseline_record_{time.time():.0f}.mp4"
        output_filepath = os.path.join(self.output_path, "videos", output_filename)

        # A very low quality but fast configuration
        crf = 40
        preset = 'ultrafast'
        resolution = f"{width//4}x{height//4}"

        command = [
            'ffmpeg',
            '-y',
            '-f', 'rawvideo',
            '-vcodec', 'rawvideo',
            '-s', f'{width}x{height}',
            '-pix_fmt', 'rgb24',
            '-r', str(fps), # Tell ffmpeg the rate of the incoming stream
            '-i', '-',
            '-an', # No audio
            '-c:v', 'libx264',
            '-preset', preset,
            '-crf', str(crf),
            '-vf', f'scale={resolution}',
            '-pix_fmt', 'yuv420p',
            output_filepath,
        ]

        proc = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        
        return proc, proc.stdin

    def shutdown_all(self):
        with self.lock:
            for proc in self.active_processes:
                if proc.poll() is None: # If the process is still running
                    try:
                        proc.terminate() # Send SIGTERM
                    except Exception as e:
                        print(f"Error terminating ffmpeg process {proc.pid}: {e}")

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
        "salience_kernels": SALIENCE_KERNELS, # <-- Add this line
        "salience_strategy": SALIENCE_STRATEGY,
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