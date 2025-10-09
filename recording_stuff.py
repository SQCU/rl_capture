# recording_stuff.py

import mss
import pynput
import torch
import pyarrow as pa
import pyarrow.parquet as pq
import orjson
import time
import uuid
import os
import io
import numpy as np
import subprocess
from multiprocessing import Process, Queue, shared_memory
from threading import Thread, Lock
from PIL import Image

from salience_workers import analysis_worker_loop, SigLIPSalienceWorker, OCRLatentWorker
from window_utils import get_window_finder, WindowNotFoundError

# --- Configuration ---
# Find a window by title (leave None to capture primary monitor)
RECORDING_WINDOW_NAME = "ToramOnline" # e.g., "Cyberpunk 2077"
# Or define a specific region if window title matching fails
CAPTURE_REGION = None # e.g., {"top": 40, "left": 0, "width": 800, "height": 600}
OUTPUT_PATH = f"./capture_run_{int(time.time())}"
BUFFER_SECONDS = 10
CHUNK_SECONDS = 0.5
CAPTURE_FPS = 30

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

# --- UNSTUBBED: Input Logging Worker ---
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

class MemoryAndDispatchManager:
    def __init__(self, config):
        self.config = config
        self.is_running = True
        self.is_resizing = False

        # --- One-Time Setup ---
        self.task_queues = {"ocr": Queue(), "visual": Queue()}
        self.return_queue = Queue()
        # This queue is now shared with the persistence manager
        self.event_persistence_queue = Queue() 

        self.pipeline_lock = Lock()
        self.window_finder = get_window_finder()
        
        # --- NEW: Instantiate the Persistence Manager ---
        parquet_path = os.path.join(self.config['output_path'], "events.parquet")
        self.persistence_manager = AsyncPersistenceManager(self.config['output_path'], self.event_persistence_queue)

        initial_geometry = self.window_finder.find_window_by_title(self.config['window_name'])
        if initial_geometry is None:
            raise WindowNotFoundError(f"Could not find window '{self.config['window_name']}' on startup.")
        
        self.video_encoder = VideoEncoder(self.config['output_path'])
        self.frame_archiver = FrameArchiver(self.config['output_path'])
        self._initialize_capture_pipeline(initial_geometry)
        
        self.processes = []
        self.threads = []

        # --- NEW: State for Time-Based Baseline Recorder ---
        self.baseline_encoder_pipe = None
        # This now stores the timestamp of the last saved frame.
        self.last_baseline_frame_ts = 0.0
        # The interval in seconds between baseline frames.
        self.baseline_interval_sec = 5.0 

        
    def _initialize_capture_pipeline(self, geometry):
        """Initializes all components that depend on capture dimensions."""
        print(f"Initializing capture pipeline for geometry: {geometry}")
        self.capture_monitor = geometry
        h, w = self.capture_monitor['height'], self.capture_monitor['width']
        if h <= 0 or w <= 0: raise ValueError("Invalid window geometry")
        self.capture_dims = (h, w, 3)

        # This queue is for the producer(capture) -> consumer(dispatch) pattern
        # and should be reset with the pipeline.
        self._internal_chunk_queue = Queue()

        # Shared Memory Buffer Setup
        buffer_frames = int(self.config['buffer_seconds'] * self.config['capture_fps'])
        self.buffer_shape = (buffer_frames, *self.capture_dims)
        # Calculate the size and explicitly cast it to a standard Python int.
        buffer_size = int(np.prod(self.buffer_shape) * np.dtype(np.uint8).itemsize)
        
        # A unique name is critical for each new shared memory block
        shm_name = f"salience_capture_{uuid.uuid4()}"
        self.shm = shared_memory.SharedMemory(name=shm_name, create=True, size=buffer_size)
        self.buffer = np.ndarray(self.buffer_shape, dtype=np.uint8, buffer=self.shm.buf)
        self.timestamps = np.zeros(buffer_frames, dtype=np.float64)
        
        # Reset state variables for the new pipeline
        self.write_head = 0
        self.chunk_size = int(self.config['chunk_seconds'] * self.config['capture_fps'])
        self.ref_counts = {}
        self.dispatch_times = {}
        self.results = {}
        self.next_chunk_id = 0

        # --- NEW: State for Buffer Management ---
        # This array tracks which frames are "checked out" by an encoder.
        self.frame_ref_counts = np.zeros(buffer_frames, dtype=np.int32)
        # This dict tracks running ffmpeg jobs. {job_id: (future, locked_indices)}
        self.pending_encodes = {}
        self.next_encode_job_id = 0

    def _teardown_capture_pipeline(self):
        """Gracefully shuts down workers and releases shared memory."""
        print("Tearing down capture pipeline...")
        # Signal workers to stop by putting None on their queues
        for q in self.task_queues.values():
            q.put(None)
        
        # Wait for worker processes to finish
        for p in self.processes:
            p.join(timeout=5)
            if p.is_alive(): p.terminate()
        self.processes = []

        # Release the shared memory block
        self.shm.close()
        self.shm.unlink()
        print("Pipeline torn down.")

    # --- FIXED: Correct Logic and Daemon Threads ---
    def run(self):
        self.start_workers()
        self.persistence_manager.start()
        self._start_baseline_recorder()

        capture_t = Thread(target=self._capture_loop)
        
        # Start Input Thread as a Daemon
        input_t = Thread(target=input_capture_worker, args=(self.event_persistence_queue,))
        input_t.daemon = True
        
        self.threads = [capture_t, input_t]
        for t in self.threads:
            t.start()

        print("Starting main dispatch loop... Press Ctrl+C to stop.")
        while self.is_running:
            try:
                if self.is_resizing:
                    self._handle_resize_event()
                    continue

                if not self._internal_chunk_queue.empty():
                    chunk_info = self._internal_chunk_queue.get_nowait()
                    self.dispatch_chunk(**chunk_info)

                if not self.return_queue.empty():
                    result = self.return_queue.get_nowait()
                    chunk_id = result.get('chunk_id')
                    if chunk_id in self.ref_counts:
                        if chunk_id not in self.results: self.results[chunk_id] = []
                        self.results[chunk_id].append(result)
                        self.ref_counts[chunk_id] -= 1
                # --- NEW: Section for Monitoring Pending Encodes ---
                self._check_pending_encodes()

                for chunk_id in list(self.ref_counts.keys()):
                    is_timed_out = (time.time() - self.dispatch_times.get(chunk_id, time.time())) > 10.0
                    if self.ref_counts.get(chunk_id, -1) <= 0 or is_timed_out:
                        self.finalize_chunk(chunk_id)
                
                # REMOVED: Redundant input event polling. The persistence manager handles this now.

                time.sleep(0.01)

            except (KeyboardInterrupt, SystemExit):
                self.is_running = False
                break
            except Exception: # Catch Empty exceptions from get_nowait
                continue
        
        print("Main loop finished.")

    def _check_pending_encodes(self):
        """Polls running ffmpeg processes and releases frame locks upon completion."""
        completed_jobs = []
        for job_id, (future, locked_indices) in self.pending_encodes.items():
            # poll() is non-blocking and returns the process exit code, or None if still running.
            if future.poll() is not None:
                print(f"Encoding job {job_id} completed with code {future.returncode}.")
                # Release the frame locks
                with self.pipeline_lock: # Protect ref counts from race conditions
                    self.frame_ref_counts[locked_indices] -= 1
                completed_jobs.append(job_id)
        
        # Clean up completed jobs from the dictionary
        for job_id in completed_jobs:
            del self.pending_encodes[job_id]

    def _handle_resize_event(self):
        """The core logic for the "hot swap"."""
        with self.pipeline_lock:
            if not self.is_resizing: return # Another thread might have handled it
            
            print("Resize detected! Pausing and re-initializing...")
            
            # 1. Teardown
            self._teardown_capture_pipeline()

            # 2. Re-initialize
            new_geometry = self.window_finder.find_window_by_title(self.config['window_name'])
            if new_geometry is None:
                print("Window lost during resize. Shutting down.")
                self.is_running = False
                self.is_resizing = False
                return
            
            self._initialize_capture_pipeline(new_geometry)

            # 3. Relaunch workers
            self.start_workers()

            # 4. Resume
            self.is_resizing = False
            print("Pipeline re-initialized. Resuming capture.")

    def _capture_loop(self):
        frames_since_last_chunk = 0
        with mss.mss() as sct:
            while self.is_running:
                # Pause capture if a resize is in progress
                if self.is_resizing:
                    time.sleep(0.1)
                    continue
                
                # --- NEW: CRITICAL Buffer Full Check ---
                # Check if the next write position is locked by a pending encode.
                if self.frame_ref_counts[self.write_head] > 0:
                    print(f"WARNING: Buffer full! Dropping frame. Encoder is not keeping up.")
                    # Sleep briefly to avoid a tight spinning loop that burns CPU
                    time.sleep(1.0 / self.config['capture_fps'])
                    continue # Skip this capture iteration

                start_time = time.time()
                current_geometry = None
                if self.config['window_name']:
                    current_geometry = self.window_finder.find_window_by_title(self.config['window_name'])
                
                if current_geometry is None:
                    # Window lost, pause and retry
                    print(f"Window '{self.config['window_name']}' lost. Pausing capture...")
                    time.sleep(1.0)
                    continue

                # --- UNSTUBBED: Resize Detection Logic ---
                if (current_geometry['width'] != self.capture_dims[1] or
                    current_geometry['height'] != self.capture_dims[0]):
                    # CRITICAL: Signal the resize and stop capturing
                    self.is_resizing = True
                    continue # Immediately stop this loop's iteration

                self.capture_monitor = current_geometry
                img = sct.grab(self.capture_monitor)
                # Convert BGRA from mss to RGB for our models
                frame_rgb = np.array(img)[:,:,:3]

                # Write to circular buffer
                current_head = self.write_head
                self.buffer[current_head] = frame_rgb
                self.timestamps[current_head] = start_time
                self.write_head = (self.write_head + 1) % self.buffer_shape[0]
                
                frames_since_last_chunk += 1
                if frames_since_last_chunk >= self.chunk_size:
                    if not self.is_running:
                            break # Exit the loop immediately
                    chunk_start_index = (self.write_head - self.chunk_size) % self.buffer_shape[0]
                    self._internal_chunk_queue.put({
                        "chunk_id": f"chunk_{self.next_chunk_id}",
                        "start_index": chunk_start_index,
                        "num_frames": self.chunk_size
                    })
                    self.next_chunk_id += 1
                    frames_since_last_chunk = 0
                
                # --- CORRECTED: Time-Based Trigger for Baseline Recorder ---
                if self.baseline_encoder_pipe and (start_time - self.last_baseline_frame_ts >= self.baseline_interval_sec):
                    try:
                        self.baseline_encoder_pipe.write(frame_rgb.tobytes())
                        # IMPORTANT: Update the timestamp of the last successful write.
                        self.last_baseline_frame_ts = start_time
                    except (IOError, BrokenPipeError):
                        print("Error: Baseline recorder pipe has broken. Attempting to restart...")
                        self._restart_baseline_recorder()
                    except Exception as e:
                        if self.is_running:
                            print(f"Error in capture loop: {e}")
                            time.sleep(0.5)
                        else:
                            print(f"Error,  breaking: {e}")
                            break # Exit if shutdown was initiated
                

                # Sleep to maintain target FPS
                elapsed = time.time() - start_time
                sleep_time = (1.0 / self.config['capture_fps']) - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

    def dispatch_chunk(self, chunk_id, start_index, num_frames):
        # Get timestamps for the chunk, handling wraparound
        indices = np.arange(start_index, start_index + num_frames) % self.buffer_shape[0]
        chunk_timestamps = self.timestamps[indices].tolist()

        task = {
            "chunk_id": chunk_id, "start": start_index, "num": num_frames,
            "shm_name": self.shm.name, "shape": self.buffer_shape, "dtype": self.buffer.dtype,
            "timestamps": chunk_timestamps
        }
        
        num_workers = 0
        for queue in self.task_queues.values():
            queue.put(task)
            num_workers += 1
        
        self.ref_counts[chunk_id] = num_workers
        self.dispatch_times[chunk_id] = time.time()
        print(f"Dispatched {chunk_id} to {num_workers} workers.")

    # In class MemoryAndDispatchManager:

    def finalize_chunk(self, chunk_id):
        """
        Processes all analysis results for a completed chunk, orchestrating
        video encoding, still image archiving, and metadata logging.
        """
        collected_results = self.results.pop(chunk_id, [])
        if not collected_results:
            # If there are no results, there's nothing to do.
            del self.ref_counts[chunk_id]
            del self.dispatch_times[chunk_id]
            return

        print(f"Finalizing chunk {chunk_id} with results from {len(collected_results)} worker(s)...")

        # --- Step 1: Initialize Action Plans ---
        # We will scan all results once and decide what actions to take.
        video_encode_jobs = []
        still_archive_jobs = []
        events_to_log = []

        # --- Step 2: Rule Application Loop ---
        # Scan all results and populate the action plans.
        for result_group in collected_results:
            source_worker = result_group.get('source', 'unknown_worker')
            for event in result_group.get('data', []):
                event_type = event.get('type')
                timestamp = event.get('timestamp')

                if not timestamp:
                    continue # Skip malformed events

                # Rule A: Visual keyframe triggers a high-quality video encode job.
                if event_type == 'VISUAL_KEYFRAME':
                    start_time = timestamp - 2.0
                    end_time = timestamp + 2.0
                    video_encode_jobs.append({'start_time': start_time, 'end_time': end_time, 'quality': 'HIGH'})
                    print(f"  - Plan: Encode HIGH quality video around {timestamp:.2f}s due to {source_worker}.")

                # Rule B: OCR keyframe triggers a still image archive job.
                if event_type == 'OCR_LATENT_KEYFRAME':
                    still_archive_jobs.append({'timestamp': timestamp, 'event_type': 'ocr_keyframe', 'original_event': event})
                    print(f"  - Plan: Archive still image at {timestamp:.2f}s due to {source_worker}.")
                
                # All valid events are candidates for logging.
                events_to_log.append(event)
        
        # --- Step 3: Execute Action Plans ---

        # Execute Still Image Archiving (Synchronous, Fast)
        for job in still_archive_jobs:
            with self.pipeline_lock: # Lock to safely read from the buffer
                time_deltas = np.abs(self.timestamps - job['timestamp'])
                closest_index = np.argmin(time_deltas)
                
                if time_deltas[closest_index] < 0.1: # 100ms tolerance
                    frame_to_save = self.buffer[closest_index]
                    image_path = self.frame_archiver.save_frame(frame_to_save, job['timestamp'], job['event_type'])
                    
                    # IMPORTANT: Mutate the original event to include the image path
                    if image_path:
                        job['original_event']['payload']['image_pointer'] = image_path
                else:
                    print(f"  - Warning: Could not find a matching frame for still at {job['timestamp']:.2f}s.")

        # Execute Video Encoding (Asynchronous, Slow)
        # Note: We might have multiple overlapping jobs; a more advanced implementation could merge them.
        # For now, we process them as they came.
        for job in video_encode_jobs:
            with self.pipeline_lock:
                valid_indices_mask = (self.timestamps >= job['start_time']) & (self.timestamps <= job['end_time'])
                locked_indices = np.where(valid_indices_mask)[0]

                if len(locked_indices) > 0:
                    self.frame_ref_counts[locked_indices] += 1
                    
                    sorted_indices = sorted(locked_indices, key=lambda i: self.timestamps[i])
                    frames_to_encode = self.buffer[sorted_indices]
                    timestamps_to_encode = self.timestamps[sorted_indices]

                    future = self.video_encoder.start_encode_slice(frames_to_encode, timestamps_to_encode, job['quality'])
                    if future:
                        job_id = self.next_encode_job_id
                        self.pending_encodes[job_id] = (future, sorted_indices)
                        self.next_encode_job_id += 1
                else:
                    print(f"  - Warning: No frames found for video encode job in range [{job['start_time']:.2f}s, {job['end_time']:.2f}s].")

        # --- Step 4: Log Metadata ---
        # Log all events from the chunk, some of which may now be enriched with image pointers.
        self.log_events_from_results(events_to_log)
        
        # --- Step 5: Cleanup Chunk State ---
        del self.ref_counts[chunk_id]
        del self.dispatch_times[chunk_id]

    def log_events_from_results(self, collected_results):
        """
        Processes results from workers and puts the resulting events onto the
        persistence queue instead of a local list.
        """
        for result_group in collected_results:
            for event_data in result_group['data']:
                event = {
                    "event_id": str(uuid.uuid4()),
                    "stream_type": event_data.get("type", "UNKNOWN"),
                    "start_timestamp": event_data.get("timestamp"),
                    "delta_timestamp": 0.0, # Instantaneous salience event
                    "payload_json": orjson.dumps(event_data).decode('utf-8')
                }
                self.event_persistence_queue.put(event)

    def _start_baseline_recorder(self):
        """Launches the persistent ffmpeg process for the low-quality baseline video."""
        print("Starting low-quality baseline recorder...")
        try:
            # The VideoEncoder needs a new method to handle this.
            # It will return the process object and its stdin pipe.
            self.baseline_proc, self.baseline_encoder_pipe = self.video_encoder.start_continuous_encode(
                dimensions=self.capture_dims,
                fps=self.config['capture_fps'], # The pipe needs a target FPS
                quality='VERY_LOW'
            )
        except Exception as e:
            print(f"CRITICAL ERROR: Failed to start baseline recorder: {e}")

    def _restart_baseline_recorder(self):
        """Handles the case where the persistent ffmpeg process crashes."""
        if self.baseline_proc:
            self.baseline_proc.kill() # Ensure the old process is gone
        self.baseline_encoder_pipe = None
        self.last_baseline_frame_ts = 0.0
        self._start_baseline_recorder() # Try to launch a new one

    def start_workers(self):
        # OCR Worker
        self.processes.append(
            Process(target=analysis_worker_loop, args=(
                self.task_queues['ocr'], self.return_queue, 'OCRLatentWorker',
                None, self.config['output_path']
            ))
        )
        # Visual Salience Worker
        self.processes.append(
            Process(target=analysis_worker_loop, args=(
                self.task_queues['visual'], self.return_queue, 'SigLIPSalienceWorker',
                None, self.config['output_path']
            ))
        )
        for p in self.processes:
            p.start()

    def shutdown(self):
        print("Shutting down manager...")
        if not self.is_running: return

        self.is_running = False
        
        for t in self.threads:
            if t.is_alive() and not t.isDaemon():
                t.join(timeout=2)

        if self.baseline_encoder_pipe:
            try:
                self.baseline_encoder_pipe.close()
            except Exception as e:
                print(f"Error closing baseline encoder pipe: {e}")
        
        # Wait for the baseline process to finish writing its file
        if hasattr(self, 'baseline_proc') and self.baseline_proc.poll() is None:
            self.baseline_proc.wait(timeout=5)
            if self.baseline_proc.poll() is None:
                self.baseline_proc.kill()

        self._teardown_capture_pipeline()
        
        # --- NEW: Shutdown the persistence manager ---
        # This will handle the final write and close the file.
        self.persistence_manager.shutdown()

# Add this class to recording_stuff.py

class VideoEncoder:
    """
    Manages ffmpeg subprocesses for encoding video clips from raw frames.
    """
    def __init__(self, output_path: str):
        self.output_path = output_path
        os.makedirs(os.path.join(self.output_path, "videos"), exist_ok=True)

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

if __name__ == "__main__":
    # Ensure a window name is set if not using a fallback
    if RECORDING_WINDOW_NAME is None and CAPTURE_REGION is None:
        print("ERROR: You must set RECORDING_WINDOW_NAME or CAPTURE_REGION.")
        # In a real app, you might fall back to primary monitor capture,
        # but for this specific design, we require a target.
        exit(1)

    os.makedirs(OUTPUT_PATH, exist_ok=True)
    
    config = {
        "window_name": RECORDING_WINDOW_NAME,
        "region": CAPTURE_REGION,
        "output_path": OUTPUT_PATH,
        "buffer_seconds": BUFFER_SECONDS,
        "chunk_seconds": CHUNK_SECONDS,
        "capture_fps": CAPTURE_FPS,
    }

    manager = None
    try:
        manager = MemoryAndDispatchManager(config)
        manager.run()
    except WindowNotFoundError as e:
        print(f"FATAL STARTUP ERROR: {e}")
    except Exception as e:
        print(f"An unexpected error occurred in the main block: {e}")

    finally:
        if manager:
            manager.shutdown()