# hmi_utils.py
### helper functions for human-machine interfaces
import io
import os
import time
import uuid
import numpy as np
import pywinctl as pwc
from threading import Thread, Lock
from dataclasses import dataclass
from multiprocessing import Process, Queue, shared_memory, Event
import subprocess
import psutil
import pynput
import orjson
from PIL import Image # <--- for framearchiver.
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
            '-s', f'{width}x{height}', '-pix_fmt', 'rgb24', '-r', str(fps),
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
        return proc, proc.stdin, proc.stderr

    def shutdown_all(self):
        with self.lock:
            for proc in self.active_processes:
                if proc.poll() is None: # If the process is still running
                    try:
                        proc.terminate() # Send SIGTERM
                    except Exception as e:
                        print(f"Error terminating ffmpeg process {proc.pid}: {e}")

def input_capture_worker(event_queue: Queue, window_name: str):
    """Listens for keyboard and mouse events and puts them on the event queue."""
    key_states = {} # Tracks the start time of key presses
    focus_tracker = FocusTracker(window_name) #revised to take operand windowname
    trajectory_tracker = MouseTrajectoryTracker(sample_interval=0.0111)  # 90Hz sampling
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
