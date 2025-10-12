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
from dataclasses import dataclass
import psutil

# We now need one of the salience workers in the Triage process itself
import salience_workers as sal_wo
from window_utils import get_window_finder, WindowNotFoundError

# --- Configuration ---
RECORDING_WINDOW_NAME = "ATLYSS"
CAPTURE_REGION = None
OUTPUT_PATH = f"./capture_run_{int(time.time())}"
CHUNK_SECONDS = 10  # How many seconds of frames to triage at a time
CAPTURE_FPS = 30
UNHANDLED_SHM_DEADLINE_SECONDS = 75.0 # Time before we declare a block abandoned
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
        "novelty_z_score_threshold": 0.5, # Z-score threshold for the Mahalanobis scores themselves.
        "branching_factor": 8, # Override the worker's default of 8
        "min_exhaustive_size": 8, # Override the worker's default of 16
        "max_batch_size": 8,
        "top_k": 3,          # Explore the top 2 sub-spans in a "hot" region.
        "max_d": 3,          # Max recursion depth for "hot" regions.
        "top_k_lazy": 1,     # Explore only the top sub-span in a "cold" region.
        "max_d_lazy": 1,     # Give up on "cold" regions faster.
        "max_p": 0.33,       # Hard budget: process at most 33% of total frames in a chunk.
        "kernel_configs": {
            "siglip": {
                "max_batch_size": 8 # SigLIP is efficient
            },
            "ocr": {
                "max_batch_size": 8   # OCR model is huge, process one by one
            }
    }
}
}

### helper functions 
import pywinctl as pwc
from threading import Thread
import time
# claude hates the unfocused general purpose keylogger, they RLHFed that fellow to shreds
class FocusTracker:
    """Tracks which window currently has focus."""
    def __init__(self, target_window_name: str):
        self.target_window_name = target_window_name
        self.has_focus = False
        self.is_running = True
        self.thread = Thread(target=self._poll_loop, daemon=True)
        self.thread.start()
    
    def _poll_loop(self):
        """Poll the active window every 50ms."""
        while self.is_running:
            active = pwc.getActiveWindow()
            if active:
                self.has_focus = (self.target_window_name in active.title)
            time.sleep(0.05)  # 20Hz polling rate
    
    def stop(self):
        self.is_running = False
        self.thread.join()

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

@dataclass
class MouseTrajectory:
    """Represents a complete mouse gesture."""
    start_time: float
    end_time: float
    start_pos: tuple[int, int]
    end_pos: tuple[int, int]
    sample_points: list[tuple[int, int]]  # Downsampled path
    button_held: str | None  # 'left', 'right', or None for hover
    total_distance: float
    linearity: float  # 0.0 = very curved, 1.0 = perfectly straight

class MouseTrajectoryTracker:
    """
    Buffers mouse movements and emits trajectory events when motion stops
    or button state changes.
    """
    def __init__(self, sample_interval: float = 0.05):
        """
        Args:
            sample_interval: Time in seconds between position samples (20Hz default)
        """
        self.sample_interval = sample_interval
        self.last_sample_time = 0.0
        
        # Current trajectory being built
        self.current_trajectory = []
        self.trajectory_start_time = None
        self.trajectory_button = None
        
        # Motion detection
        self.last_position = None
        self.motion_timeout = 0.2  # If no movement for 200ms, trajectory ends
        self.last_motion_time = None
    
    def _calculate_linearity(self, points: list[tuple[int, int]]) -> float:
        """
        Measures how straight a path is.
        Returns 1.0 for perfectly straight line, lower for curved paths.
        """
        if len(points) < 3:
            return 1.0
        
        start = np.array(points[0])
        end = np.array(points[-1])
        
        # Direct distance (straight line)
        direct_distance = np.linalg.norm(end - start)
        
        if direct_distance < 1:  # Tiny movement, consider it straight
            return 1.0
        
        # Actual path length
        path_length = 0.0
        for i in range(len(points) - 1):
            p1 = np.array(points[i])
            p2 = np.array(points[i + 1])
            path_length += np.linalg.norm(p2 - p1)
        
        # Linearity = direct_distance / path_length
        # (1.0 = straight, <1.0 = curved)
        return direct_distance / path_length if path_length > 0 else 1.0
    
    def _calculate_total_distance(self, points: list[tuple[int, int]]) -> float:
        """Sum of distances between consecutive points."""
        if len(points) < 2:
            return 0.0
        
        total = 0.0
        for i in range(len(points) - 1):
            p1 = np.array(points[i])
            p2 = np.array(points[i + 1])
            total += np.linalg.norm(p2 - p1)
        return total
    
    def on_move(self, x: int, y: int, current_button: str | None) -> MouseTrajectory | None:
        """
        Called on every mouse movement event.
        Returns a completed trajectory if motion has stopped, otherwise None.
        """
        current_time = time.time()
        
        # Initialize tracking if this is the first movement
        if self.trajectory_start_time is None:
            self.trajectory_start_time = current_time
            self.current_trajectory = [(x, y)]
            self.trajectory_button = current_button
            self.last_position = (x, y)
            self.last_motion_time = current_time
            self.last_sample_time = current_time
            return None
        
        # Check if button state changed (e.g., started/stopped dragging)
        if current_button != self.trajectory_button:
            completed = self._finalize_trajectory()
            # Start new trajectory
            self.trajectory_start_time = current_time
            self.current_trajectory = [(x, y)]
            self.trajectory_button = current_button
            self.last_position = (x, y)
            self.last_motion_time = current_time
            self.last_sample_time = current_time
            return completed
        
        # Check if motion has stopped (timeout)
        if self.last_position is not None:
            dx = x - self.last_position[0]
            dy = y - self.last_position[1]
            distance = (dx*dx + dy*dy)**0.5
            
            # If mouse hasn't moved significantly
            if distance < 2:  # 2 pixel threshold
                if current_time - self.last_motion_time > self.motion_timeout:
                    # Motion stopped, finalize trajectory
                    completed = self._finalize_trajectory()
                    # Reset for next trajectory
                    self.trajectory_start_time = None
                    self.current_trajectory = []
                    return completed
            else:
                # Motion detected, update timestamp
                self.last_motion_time = current_time
        
        # Sample position at fixed interval (avoid over-sampling)
        if current_time - self.last_sample_time >= self.sample_interval:
            self.current_trajectory.append((x, y))
            self.last_sample_time = current_time
        
        self.last_position = (x, y)
        return None
    
    def _finalize_trajectory(self) -> MouseTrajectory | None:
        """Converts the current trajectory buffer into a trajectory event."""
        if not self.current_trajectory or len(self.current_trajectory) < 2:
            return None
        
        linearity = self._calculate_linearity(self.current_trajectory)
        total_distance = self._calculate_total_distance(self.current_trajectory)
        
        # Only log if the movement was significant
        if total_distance < 10:  # Ignore tiny jitter movements
            return None
        
        trajectory = MouseTrajectory(
            start_time=self.trajectory_start_time,
            end_time=time.time(),
            start_pos=self.current_trajectory[0],
            end_pos=self.current_trajectory[-1],
            sample_points=self.current_trajectory[::5],  # Further downsample for storage
            button_held=self.trajectory_button,
            total_distance=total_distance,
            linearity=linearity
        )
        
        return trajectory
    
    def force_finalize(self) -> MouseTrajectory | None:
        """
        Call this on shutdown or focus loss to flush the current trajectory
        and reset the tracker's state.
        """
        # --- FIX: Finalize and then explicitly reset the tracker's state ---
        completed_trajectory = self._finalize_trajectory()
        
        # Reset state to prevent re-firing with stale data
        self.current_trajectory = []
        self.trajectory_start_time = None
        self.last_position = None
        self.last_motion_time = None
        
        return completed_trajectory

@dataclass
class ScrollTrajectory:
    """Represents a complete scroll gesture."""
    start_time: float
    end_time: float
    total_dx: int
    total_dy: int
    reversals: int # How many times the scroll direction changed
    duration: float

class ScrollTrajectoryTracker:
    """
    Buffers scroll events and emits a trajectory when the user stops scrolling.
    This turns a high-frequency stream of scroll ticks into a single,
    semantically meaningful event.
    """
    def __init__(self, timeout: float = 0.3):
        """
        Args:
            timeout: Time in seconds of no scrolling to consider the gesture complete.
        """
        self.timeout = timeout
        
        # State for the current trajectory
        self.start_time = None
        self.last_scroll_time = None
        self.total_dx = 0
        self.total_dy = 0
        self.history_dy = []

    def on_scroll(self, dx: int, dy: int) -> ScrollTrajectory | None:
        """
        Processes a single scroll tick. Returns a completed trajectory if a
        new gesture is starting after a pause, otherwise None.
        """
        current_time = time.time()
        completed_trajectory = None

        # If a scroll gesture was active but has timed out, finalize it.
        if self.start_time and (current_time - self.last_scroll_time > self.timeout):
            completed_trajectory = self._finalize_trajectory()
            self._reset()

            # --- ADD THIS GUARD ---
        if dx == 0 and dy == 0: # This is a dummy event just to check for timeouts
            return completed_trajectory

        # Start a new trajectory if one isn't active
        if not self.start_time:
            self.start_time = current_time
        
        # Update the current trajectory's state
        self.last_scroll_time = current_time
        self.total_dx += dx
        self.total_dy += dy
        self.history_dy.append(dy)

        return completed_trajectory

    def _finalize_trajectory(self) -> ScrollTrajectory | None:
        """Analyzes the buffered scroll data and creates a trajectory event."""
        if self.start_time is None:
            return None

        # Analyze history for direction reversals (e.g., overshot and corrected)
        reversals = 0
        if len(self.history_dy) > 1:
            # Get the signs of each scroll tick (+1 for up, -1 for down)
            signs = np.sign(self.history_dy)
            # Find where the sign changes
            sign_changes = np.diff(signs)
            # Count non-zero changes
            reversals = np.count_nonzero(sign_changes)

        trajectory = ScrollTrajectory(
            start_time=self.start_time,
            end_time=self.last_scroll_time,
            total_dx=self.total_dx,
            total_dy=self.total_dy,
            reversals=int(reversals),
            duration=self.last_scroll_time - self.start_time
        )
        return trajectory

    def _reset(self):
        """Resets the state to prepare for the next gesture."""
        self.start_time = None
        self.last_scroll_time = None
        self.total_dx = 0
        self.total_dy = 0
        self.history_dy = []

    def force_finalize(self) -> ScrollTrajectory | None:
        """Called on shutdown or focus loss to flush any pending scroll gesture."""
        completed = self._finalize_trajectory()
        self._reset()
        return completed


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

        # --- FIX: Build a video filter chain to ensure even dimensions ---
        filter_chain = ["crop=trunc(iw/2)*2:trunc(ih/2)*2"]

        if quality == 'HIGH':
            crf = 20
            preset = 'fast'
        else: # VERY_LOW
            crf = 35
            preset = 'ultrafast'
            # Also ensure the downscaled resolution is even
            out_w = (width // 4) - ((width // 4) % 2)
            out_h = (height // 4) - ((height // 4) % 2)
            filter_chain.append(f"scale={out_w}:{out_h}")

        command = [
            'ffmpeg', '-y', '-f', 'rawvideo', '-vcodec', 'rawvideo',
            '-s', f'{width}x{height}', '-pix_fmt', 'rgb24', '-r', str(avg_fps),
            '-i', '-', '-c:v', 'libx264', '-preset', preset, '-crf', str(crf),
            '-vf', ",".join(filter_chain), # Apply the filter chain
            '-pix_fmt', 'yuv420p', output_filepath,
        ]
        # --- END FIX ---

        # Start the process with stdin piped
        proc = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        ps_proc = psutil.Process(proc.pid)

        ps_proc.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS) # For Windows
        # ps_p.nice(10) # For Linux (higher number is lower priority)

        with self.lock:
            self.active_processes.append(proc)
        
        # Write all frame data to the pipe in a separate thread to avoid blocking
        def pipe_frames():
            try:
                contiguous_frames = np.ascontiguousarray(frames)
                proc.stdin.write(contiguous_frames.tobytes())
            except (IOError, BrokenPipeError):
                print(f"Warning: ffmpeg pipe broke for {output_filename}. The process may have terminated early.")
            finally:
                proc.stdin.close()
        
        Thread(target=pipe_frames).start()
        
        print(f"Started encoding {output_filename}...")
        return proc

    def start_continuous_encode(self, dimensions: tuple, fps: int) -> (subprocess.Popen, io.BufferedWriter):
        """
        Launches a persistent ffmpeg process for a continuous, low-quality recording.
        
        Returns:
            A tuple containing the Popen object and its stdin pipe.
        """
        height, width, _ = dimensions
        output_filename = f"baseline_record_{time.time():.0f}.mp4"
        output_filepath = os.path.join(self.output_path, "videos", output_filename)

        # A very low quality but fast configuration for the baseline
        crf = 38  # Higher CRF is lower quality, smaller file. 38 is quite low.
        preset = 'ultrafast' # Prioritize low CPU usage
        # Downscale to a 'potato quality' resolution like 480p
        height_out = 480
        width_out = int(width * (height_out / height))
        if width_out % 2 != 0: width_out += 1 # Ensure final width is even

        # The crop filter runs first on the input, then the scale filter runs on the result.
        filter_chain = [
            "crop=trunc(iw/2)*2:trunc(ih/2)*2",
            f"scale={width_out}:{height_out}"
        ]

        command = [
            'ffmpeg', '-y', '-f', 'rawvideo', '-vcodec', 'rawvideo',
            '-s', f'{width_out}x{height_out}', '-pix_fmt', 'rgb24', '-r', str(fps),
            '-i', '-', '-an', '-c:v', 'libx264', '-preset', preset, '-crf', str(crf),
            '-vf', ",".join(filter_chain), # Apply the filter chain
            '-pix_fmt', 'yuv420p', output_filepath,
        ]

        print(f"Starting continuous baseline recording to {output_filepath} at {width_out}:{height_out}...")
        proc = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        
        # Set low priority to not interfere with the game or capture
        ps_proc = psutil.Process(proc.pid)
        ps_proc.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS) # For Windows
        # On Linux/macOS, you'd use: ps_proc.nice(15)

        # We don't add this to self.active_processes because it has a special shutdown lifecycle
        return proc, proc.stdin

    def shutdown_all(self):
        with self.lock:
            for proc in self.active_processes:
                if proc.poll() is None: # If the process is still running
                    try:
                        proc.terminate() # Send SIGTERM
                    except Exception as e:
                        print(f"Error terminating ffmpeg process {proc.pid}: {e}")

def input_capture_worker(event_queue: Queue):
    """Listens for keyboard and mouse events and puts them on the event queue."""
    key_states = {} # Tracks the start time of key presses
    focus_tracker = FocusTracker(RECORDING_WINDOW_NAME)
    trajectory_tracker = MouseTrajectoryTracker(sample_interval=0.0083)  # 120Hz sampling
    scroll_tracker = ScrollTrajectoryTracker(timeout=0.3)

    # Track which mouse buttons are currently held
    buttons_held = set()

    def _emit_mouse_trajectory(traj: MouseTrajectory, queue: Queue):
        """Helper to convert trajectory to event and emit."""
        event = {
            "event_id": str(uuid.uuid4()),
            "stream_type": "USER_INPUT",
            "start_timestamp": traj.start_time,
            "delta_timestamp": traj.end_time - traj.start_time,
            "payload_json": orjson.dumps({
                "type": "mouse_trajectory",
                "start_pos": traj.start_pos,
                "end_pos": traj.end_pos,
                "sample_points": traj.sample_points,  # Downsampled path
                "button_held": traj.button_held,
                "total_distance": float(traj.total_distance),
                "linearity": float(traj.linearity),
            }).decode('utf-8')
        }
        queue.put(event)

    def _emit_scroll_trajectory(traj: ScrollTrajectory, queue: Queue):
        """Helper to convert scroll trajectory to event and emit."""
        event = {
            "event_id": str(uuid.uuid4()),
            "stream_type": "USER_INPUT",
            "start_timestamp": traj.start_time,
            "delta_timestamp": traj.duration,
            "payload_json": orjson.dumps({
                "type": "scroll_trajectory",
                "total_dx": traj.total_dx,
                "total_dy": traj.total_dy,
                "reversals": traj.reversals,
            }).decode('utf-8')
        }
        queue.put(event)

    def on_press(key):
        if not focus_tracker.has_focus:
            return  # Silently drop
        
        key_id = str(key)
        if key_id not in key_states:
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
    
    def on_click(x, y, button, pressed):
        if not focus_tracker.has_focus:
            return  # Silently drop
        button_str = str(button)
        if pressed:
            buttons_held.add(button_str)
            event = {
                "event_id": str(uuid.uuid4()),
                "stream_type": "USER_INPUT",
                "start_timestamp": time.time(),
                "delta_timestamp": 0.0,
                "payload_json": orjson.dumps({
                    "type": "mouse_click",
                    "button": button_str,
                    "x": x,
                    "y": y
                }).decode('utf-8')
            }
            event_queue.put(event)
        else: # This handles the button release
            # --- FIX #3: Remove the button from the set when it's released ---
            # Using .discard() is safer than .remove() as it won't error if the key is missing.
            buttons_held.discard(button_str)

    def on_scroll(x, y, dx, dy):
        """Logs mouse scroll events."""
        if not focus_tracker.has_focus:
            # Finalize any pending gesture on focus loss
            completed_scroll = scroll_tracker.force_finalize()
            if completed_scroll:
                _emit_scroll_trajectory(completed_scroll, event_queue)
            return
        
        # The tracker may return a completed trajectory from the *previous* gesture
        completed_scroll = scroll_tracker.on_scroll(dx, dy)
        if completed_scroll:
            _emit_scroll_trajectory(completed_scroll, event_queue)

    def on_move(x, y):
        # --- FIX: Add a periodic check for timed-out scroll gestures ---
        # This is a good place because on_move fires frequently.
        timed_out_scroll = scroll_tracker.on_scroll(0, 0) # Sending a dummy event triggers the timeout check
        if timed_out_scroll:
            _emit_scroll_trajectory(timed_out_scroll, event_queue)

        if not focus_tracker.has_focus:
            # If we lose focus during a trajectory, finalize it
            if trajectory_tracker.trajectory_start_time is not None:
                completed_trajectory = trajectory_tracker.force_finalize()
                if completed_trajectory:
                    _emit_mouse_trajectory(completed_trajectory, event_queue)
            return
        
        # Determine current button state
        current_button = list(buttons_held)[0] if buttons_held else None
        
        # Update trajectory tracker
        completed_trajectory = trajectory_tracker.on_move(x, y, current_button)
        
        if completed_trajectory:
            _emit_mouse_trajectory(completed_trajectory, event_queue)
    
    # In a real app, you would add mouse listeners as well (on_click, on_move)
    keyboard_listener = pynput.keyboard.Listener(on_press=on_press, on_release=on_release)
    mouse_listener = pynput.mouse.Listener(on_click=on_click, on_move=on_move, on_scroll=on_scroll)
    keyboard_listener.start()
    mouse_listener.start()  # ← ADD THIS
    print("Input listener started.")
    keyboard_listener.join() # This blocks until the listener stops
    mouse_listener.join()

    # Clean up focus tracker
    focus_tracker.stop()

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

def baseline_recorder_loop(baseline_queue: Queue, output_path: str, config: dict, shutdown_event: Event):
    """
    A dedicated process that consumes frames and writes them to a continuous,
    low-quality FFMPEG process.
    """
    # --- THIS IS THE FIX ---
    # The worker gets its own encoder instance, created after the process starts.
    video_encoder = VideoEncoder(output_path)
    # --- END FIX ---

    worker_pid = os.getpid()
    print(f"[{worker_pid}] Baseline Recorder process started.")
    
    ffmpeg_proc = None
    ffmpeg_pipe = None

    try:
        while not shutdown_event.is_set():
            try:
                timestamp, frame = baseline_queue.get(timeout=0.5)

                # On the very first frame, initialize the encoder
                if ffmpeg_proc is None:
                    frame_shape = frame.shape
                    ffmpeg_proc, ffmpeg_pipe = video_encoder.start_continuous_encode(
                        dimensions=frame_shape,
                        fps=config['capture_fps']
                    )
                
                # Write frame to the pipe
                ffmpeg_pipe.write(frame.tobytes())

            except queue.Empty:
                # If the queue is empty, check if we should shut down
                if shutdown_event.is_set():
                    break
                continue
            except (IOError, BrokenPipeError):
                print(f"[{worker_pid}] FFMPEG pipe broke. Exiting baseline recorder.")
                break

    finally:
        if ffmpeg_proc and ffmpeg_pipe:
            print(f"[{worker_pid}] Closing FFMPEG pipe and finalizing baseline video...")
            ffmpeg_pipe.close()
            # Log ffmpeg's output for debugging if it failed
            stderr_output = ffmpeg_proc.stderr.read().decode(errors='ignore')
            if ffmpeg_proc.wait() != 0:
                print(f"[{worker_pid}] FFMPEG process exited with non-zero status.")
                print(f"FFMPEG stderr:\n{stderr_output}")
        print(f"[{worker_pid}] Baseline Recorder process finished.")
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
    video_encoder = VideoEncoder(config['output_path'])
    window_finder = get_window_finder()

    # FFMPEG process management
    ffmpeg_proc = None
    ffmpeg_pipe = None
    current_dimensions = None

    # --- NEW: Internal queue and thread for non-blocking writes ---
    baseline_write_queue = queue.Queue(maxsize=config['capture_fps']) # Buffer up to 1s of frames
    writer_thread = None

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

    def _finalize_encoder():
        nonlocal ffmpeg_proc, ffmpeg_pipe, current_dimensions, writer_thread
        if writer_thread and writer_thread.is_alive():
            writer_thread.join(timeout=1.0) # Wait for the writer to finish
        writer_thread = None

        if ffmpeg_proc and ffmpeg_pipe:
            print(f"[{pid}] Finalizing baseline video segment...")
            ffmpeg_pipe.close()
            stderr = ffmpeg_proc.stderr.read().decode(errors='ignore')
            if ffmpeg_proc.wait() != 0 and stderr:
                print(f"[{pid}] FFMPEG stderr:\n{stderr}")
        ffmpeg_proc, ffmpeg_pipe, current_dimensions = None, None, None

    try:
        with mss.mss() as sct:
            while not shutdown_event.is_set():
                start_time = time.time()
                try:
                    monitor = window_finder.find_window_by_title(config['window_name'])
                    if monitor is None:
                        if ffmpeg_proc: _finalize_encoder()
                        time.sleep(0.5)
                        continue

                    img = sct.grab(monitor)
                    frame_rgb = np.array(img)[:, :, :3][:, :, ::-1]

                    if frame_rgb.shape != current_dimensions:
                        print(f"[{pid}] Detected dimension change to {frame_rgb.shape}. Restarting baseline encoder.")
                        _finalize_encoder()
                        ffmpeg_proc, ffmpeg_pipe = video_encoder.start_continuous_encode(
                            dimensions=frame_rgb.shape, fps=config['capture_fps']
                        )
                        current_dimensions = frame_rgb.shape
                        # Start a new writer thread for the new pipe
                        writer_thread = Thread(target=_pipe_writer_loop, args=(ffmpeg_pipe, baseline_write_queue, shutdown_event))
                        writer_thread.start()

                    # --- 1. Push frame to Salience Pipeline (NON-BLOCKING) ---
                    try:
                        raw_frame_queue.put_nowait((start_time, frame_rgb))
                    except queue.Full:
                        print(f"[{pid}] WARNING: Salience raw frame queue is full. Dropping frame for analysis.")

                    # --- 2. Push frame to Baseline Writer Thread (NON-BLOCKING) ---
                    if writer_thread:
                        try:
                            baseline_write_queue.put_nowait(frame_rgb)
                        except queue.Full:
                            # This is now safe. FFMPEG is behind, so we just drop a baseline frame.
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
        
        # --- MODIFIED: This dictionary will now hold active SHM handles ---
        self.active_shm_blocks = {} # {shm_name: shm_instance}
        self.shm_lock = Lock()
        self.max_pipeline_depth = MAX_PIPELINE_DEPTH # Max number of concurrent SHM blocks
        self.min_pipeline_depth = MIN_PIPELINE_DEPTH  # Target to shrink back towards
        
        # High-level components
        self.persistence_manager = AsyncPersistenceManager(config['output_path'], Queue()) # Give it its own queue
        self.guardian_thread = Thread(target=self._guardian_loop, daemon=True)
        self.frame_archiver = FrameArchiver(config['output_path'])
        self.video_encoder = VideoEncoder(config['output_path'])
        
        self.processes = []
        self.threads = []

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
                    self.active_shm_blocks[shm_name]['status'] = 'encoding_pending'
                else:
                    # This can happen if the block timed out and was already cleaned up
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
                print(f"Orchestrator: Cleaned up SHM block {name}.")

    def adapt_pipeline(self):
        """Dynamically adjusts capture parameters based on processing load."""
        with self.shm_lock:
            pipeline_depth = len(self.active_shm_blocks)

        if pipeline_depth > self.max_pipeline_depth:
            # We are backlogged. Increase chunk size to reduce analysis overhead per second of video.
            if self.config['chunk_seconds'] < 20:
                self.config['chunk_seconds'] += 1
                print(f"PIPELINE BACKLOG DETECTED (depth: {pipeline_depth}). Increasing chunk size to {self.config['chunk_seconds']}s.")
        elif pipeline_depth < self.min_pipeline_depth:
            # We have spare capacity. Decrease chunk size for lower latency.
            #if self.config['chunk_seconds'] > 5:
            #    self.config['chunk_seconds'] -= 1
            #    print(f"PIPELINE CAPACITY AVAILABLE (depth: {pipeline_depth}). Decreasing chunk size to {self.config['chunk_seconds']}s.")
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