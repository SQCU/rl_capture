# obs_capture.py
"""
OBS-based frame capture for RL recording.

This module provides frame capture via OBS Studio, replacing the fragile
window title-based capture in window_utils.py.

Capture Methods (in order of preference):
1. OBS Virtual Camera - Works with any game, anti-cheat safe
2. ZeroMQ IPC - Zero-copy, requires OBS plugin (future)
3. Legacy window capture - Fallback, fragile

Why OBS?
- Game Capture is BLESSED SOFTWARE - explicitly whitelisted by anti-cheat
- Captures by HWND/PID, not window title (immune to title changes)
- Virtual Camera provides standardized output
- No hooks/injections that trigger moderation APIs
"""

import numpy as np
import time
import sys
import platform
from typing import Optional, Tuple, Generator
from dataclasses import dataclass
from abc import ABC, abstractmethod
from threading import Thread, Event
from queue import Queue, Empty


@dataclass
class CapturedFrame:
    """A captured frame with metadata."""
    data: np.ndarray  # RGB frame data (H, W, 3)
    timestamp: float
    width: int
    height: int
    source: str  # "obs_vcam", "obs_zmq", "legacy"


class FrameCaptureBackend(ABC):
    """Abstract base class for frame capture backends."""

    @abstractmethod
    def is_available(self) -> bool:
        """Check if this backend can be used."""
        pass

    @abstractmethod
    def start(self) -> bool:
        """Start capturing."""
        pass

    @abstractmethod
    def stop(self):
        """Stop capturing."""
        pass

    @abstractmethod
    def get_frame(self, timeout: float = 1.0) -> Optional[CapturedFrame]:
        """Get the next frame."""
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Backend name for logging."""
        pass


# =============================================================================
#  OBS VIRTUAL CAMERA BACKEND
# =============================================================================

class OBSVirtualCameraBackend(FrameCaptureBackend):
    """
    Capture frames from OBS Virtual Camera.

    On Windows: Uses DirectShow via opencv-python
    On Linux: Uses V4L2 via opencv-python (/dev/video*)
    On macOS: Uses AVFoundation via opencv-python

    The Virtual Camera appears as a standard webcam device, making it
    universally compatible and immune to anti-cheat detection.
    """

    def __init__(self, device_index: Optional[int] = None, device_name: str = "OBS Virtual Camera"):
        self.device_index = device_index
        self.device_name = device_name
        self.cap = None
        self._cv2 = None

    @property
    def name(self) -> str:
        return "OBS Virtual Camera"

    def is_available(self) -> bool:
        """Check if OpenCV and a virtual camera device are available."""
        try:
            import cv2
            self._cv2 = cv2
        except ImportError:
            print("WARNING: opencv-python not installed. Install with: pip install opencv-python")
            return False

        # Try to find the OBS Virtual Camera device
        device_idx = self._find_virtual_camera()
        if device_idx is not None:
            self.device_index = device_idx
            return True

        return False

    def _find_virtual_camera(self) -> Optional[int]:
        """Find the OBS Virtual Camera device index."""
        if self.device_index is not None:
            return self.device_index

        system = platform.system()

        if system == "Linux":
            # On Linux, OBS Virtual Camera is typically /dev/video10
            # (configured in obs_setup.py via v4l2loopback)
            import os
            for video_dev in [10, 11, 12, 2, 3, 4]:
                if os.path.exists(f"/dev/video{video_dev}"):
                    # Try to open it
                    cap = self._cv2.VideoCapture(video_dev)
                    if cap.isOpened():
                        # Check if it's OBS by reading a frame
                        ret, _ = cap.read()
                        cap.release()
                        if ret:
                            print(f"Found video device at /dev/video{video_dev}")
                            return video_dev
            return None

        elif system == "Windows":
            # On Windows, enumerate devices and look for "OBS Virtual Camera"
            # OpenCV doesn't provide device names directly, so we try indices
            for idx in range(10):
                cap = self._cv2.VideoCapture(idx, self._cv2.CAP_DSHOW)
                if cap.isOpened():
                    ret, _ = cap.read()
                    cap.release()
                    if ret:
                        # Can't easily get device name, just return first working
                        # User should configure device_index explicitly if needed
                        print(f"Found video device at index {idx}")
                        return idx
            return None

        elif system == "Darwin":
            # On macOS, similar approach
            for idx in range(10):
                cap = self._cv2.VideoCapture(idx)
                if cap.isOpened():
                    ret, _ = cap.read()
                    cap.release()
                    if ret:
                        print(f"Found video device at index {idx}")
                        return idx
            return None

        return None

    def start(self) -> bool:
        """Start capturing from the virtual camera."""
        if self.device_index is None:
            print("ERROR: No virtual camera device found")
            return False

        backend = self._cv2.CAP_DSHOW if platform.system() == "Windows" else self._cv2.CAP_ANY
        self.cap = self._cv2.VideoCapture(self.device_index, backend)

        if not self.cap.isOpened():
            print(f"ERROR: Failed to open video device {self.device_index}")
            return False

        # Set buffer size to 1 to minimize latency
        self.cap.set(self._cv2.CAP_PROP_BUFFERSIZE, 1)

        # Try to get native resolution
        width = int(self.cap.get(self._cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(self._cv2.CAP_PROP_FRAME_HEIGHT))
        fps = self.cap.get(self._cv2.CAP_PROP_FPS)

        print(f"Virtual Camera opened: {width}x{height} @ {fps:.1f}fps")
        return True

    def stop(self):
        """Stop capturing."""
        if self.cap:
            self.cap.release()
            self.cap = None

    def get_frame(self, timeout: float = 1.0) -> Optional[CapturedFrame]:
        """Get the next frame from the virtual camera."""
        if not self.cap or not self.cap.isOpened():
            return None

        # Grab frame (with timeout via read)
        ret, frame = self.cap.read()
        timestamp = time.time()

        if not ret or frame is None:
            return None

        # OpenCV returns BGR, convert to RGB
        frame_rgb = self._cv2.cvtColor(frame, self._cv2.COLOR_BGR2RGB)

        return CapturedFrame(
            data=frame_rgb,
            timestamp=timestamp,
            width=frame_rgb.shape[1],
            height=frame_rgb.shape[0],
            source="obs_vcam"
        )


# =============================================================================
#  ZEROMQ IPC BACKEND (FUTURE)
# =============================================================================

class OBSZeroMQBackend(FrameCaptureBackend):
    """
    Zero-copy frame capture via ZeroMQ IPC.

    This backend requires an OBS plugin that publishes frame metadata
    to a ZeroMQ socket. Frames are shared via OS shared memory.

    See hypothetical_zero_copy_impl.fake for the plugin design.

    Status: NOT YET IMPLEMENTED - requires OBS plugin development
    """

    def __init__(self, endpoint: str = "ipc:///tmp/obs-frames"):
        self.endpoint = endpoint
        self.context = None
        self.subscriber = None

    @property
    def name(self) -> str:
        return "OBS ZeroMQ IPC"

    def is_available(self) -> bool:
        """Check if ZeroMQ is available and OBS plugin is running."""
        try:
            import zmq
        except ImportError:
            return False

        # TODO: Check if OBS plugin is publishing
        # For now, return False as plugin doesn't exist yet
        return False

    def start(self) -> bool:
        """Start ZeroMQ subscriber."""
        try:
            import zmq
            self.context = zmq.Context()
            self.subscriber = self.context.socket(zmq.SUB)
            self.subscriber.connect(self.endpoint)
            self.subscriber.setsockopt(zmq.SUBSCRIBE, b"")
            self.subscriber.setsockopt(zmq.RCVTIMEO, 1000)  # 1s timeout
            return True
        except Exception as e:
            print(f"ERROR: Failed to start ZeroMQ: {e}")
            return False

    def stop(self):
        """Stop ZeroMQ subscriber."""
        if self.subscriber:
            self.subscriber.close()
        if self.context:
            self.context.term()
        self.subscriber = None
        self.context = None

    def get_frame(self, timeout: float = 1.0) -> Optional[CapturedFrame]:
        """Get frame via ZeroMQ (zero-copy via shared memory)."""
        # TODO: Implement when OBS plugin is ready
        return None


# =============================================================================
#  LEGACY WINDOW CAPTURE BACKEND
# =============================================================================

class LegacyWindowCaptureBackend(FrameCaptureBackend):
    """
    Fallback to legacy window title-based capture.

    WARNING: This is fragile and will break if:
    - Game changes window title (e.g., stage changes in DOOM clones)
    - Window title contains dynamic content (FPS, map name)
    - Multiple windows have similar titles

    Only use as a fallback when OBS is not available.
    """

    def __init__(self, window_name: str):
        self.window_name = window_name
        self.sct = None
        self.window_finder = None
        self._mss = None

    @property
    def name(self) -> str:
        return "Legacy Window Capture"

    def is_available(self) -> bool:
        """Check if mss and window finding are available."""
        try:
            import mss
            self._mss = mss
            from window_utils import get_window_finder
            self.window_finder = get_window_finder()
            return self.window_finder.is_available()
        except ImportError:
            return False

    def start(self) -> bool:
        """Start mss capture."""
        try:
            self.sct = self._mss.mss()
            return True
        except Exception as e:
            print(f"ERROR: Failed to start mss: {e}")
            return False

    def stop(self):
        """Stop mss capture."""
        if self.sct:
            self.sct.close()
            self.sct = None

    def get_frame(self, timeout: float = 1.0) -> Optional[CapturedFrame]:
        """Get frame via legacy window capture."""
        if not self.sct or not self.window_finder:
            return None

        try:
            # Find window by title (FRAGILE!)
            monitor = self.window_finder.find_window_by_title(self.window_name)
            if monitor is None:
                return None

            # Capture
            img = self.sct.grab(monitor)
            timestamp = time.time()

            # Convert to numpy RGB
            frame = np.array(img)[:, :, :3][:, :, ::-1]  # BGRA -> RGB

            return CapturedFrame(
                data=frame,
                timestamp=timestamp,
                width=frame.shape[1],
                height=frame.shape[0],
                source="legacy"
            )
        except Exception:
            return None


# =============================================================================
#  UNIFIED CAPTURE MANAGER
# =============================================================================

class CaptureManager:
    """
    Unified capture manager that selects the best available backend.

    Priority:
    1. OBS Virtual Camera (if available and OBS is running)
    2. OBS ZeroMQ IPC (future, for zero-copy)
    3. Legacy window capture (fallback, only if OBS not explicitly required)
    """

    def __init__(self, window_name: str = "", prefer_obs: bool = True,
                 obs_device_index: Optional[int] = None,
                 require_obs: bool = False):
        """
        Args:
            window_name: Window title for legacy fallback capture
            prefer_obs: Try OBS backends first (default True)
            obs_device_index: Specific video device index for OBS Virtual Camera
            require_obs: If True, FAIL if OBS isn't available (no silent fallback)
        """
        self.window_name = window_name
        self.prefer_obs = prefer_obs
        self.obs_device_index = obs_device_index
        self.require_obs = require_obs

        self.backend: Optional[FrameCaptureBackend] = None
        self.running = False

    def _select_backend(self) -> Optional[FrameCaptureBackend]:
        """Select the best available capture backend."""
        obs_backends = []
        legacy_backends = []

        if self.prefer_obs:
            obs_backends.append(OBSVirtualCameraBackend(device_index=self.obs_device_index))
            obs_backends.append(OBSZeroMQBackend())

        if self.window_name:
            legacy_backends.append(LegacyWindowCaptureBackend(self.window_name))

        # Try OBS backends first
        for backend in obs_backends:
            if backend.is_available():
                print(f"[CAPTURE] Selected backend: {backend.name}")
                return backend

        # OBS backends failed
        if obs_backends:
            print("[CAPTURE] WARNING: OBS backends not available:")
            print("  - Is OBS Studio running?")
            print("  - Is Virtual Camera started? (Tools -> Start Virtual Camera)")
            print("  - Is opencv-python installed? (pip install opencv-python)")

        # If OBS was explicitly required, don't fall back silently
        if self.require_obs:
            print("[CAPTURE] ERROR: --obs was specified but OBS Virtual Camera is not available!")
            print("         Cannot fall back to legacy capture when OBS is explicitly requested.")
            return None

        # Try legacy backends as fallback
        for backend in legacy_backends:
            if backend.is_available():
                print(f"[CAPTURE] WARNING: Falling back to legacy capture: {backend.name}")
                print(f"          This is fragile and will break if window title changes!")
                return backend

        print("[CAPTURE] ERROR: No capture backend available!")
        return None

    def start(self) -> bool:
        """Start capture with the best available backend."""
        self.backend = self._select_backend()
        if not self.backend:
            return False

        if not self.backend.start():
            return False

        self.running = True
        return True

    def stop(self):
        """Stop capture."""
        self.running = False
        if self.backend:
            self.backend.stop()
            self.backend = None

    def get_frame(self, timeout: float = 1.0) -> Optional[CapturedFrame]:
        """Get the next frame."""
        if not self.backend or not self.running:
            return None
        return self.backend.get_frame(timeout)

    def capture_loop(self, target_fps: int = 60) -> Generator[CapturedFrame, None, None]:
        """
        Generator that yields frames at the target FPS.

        Usage:
            manager = CaptureManager(prefer_obs=True)
            manager.start()
            for frame in manager.capture_loop(fps=60):
                process(frame)
        """
        frame_interval = 1.0 / target_fps

        while self.running:
            start = time.time()

            frame = self.get_frame(timeout=frame_interval)
            if frame:
                yield frame

            elapsed = time.time() - start
            sleep_time = frame_interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)


# =============================================================================
#  CAPTURE PROCESS LOOP (DROP-IN REPLACEMENT)
# =============================================================================

def obs_capture_process_loop(raw_frame_queue: Queue, config: dict, shutdown_event: Event):
    """
    OBS-based capture process loop.

    Drop-in replacement for capture_process_loop in recording_stuff.py.
    Uses OBS Virtual Camera instead of fragile window title matching.
    """
    import os
    pid = os.getpid()
    print(f"[{pid}] OBS Capture process started")

    # Create capture manager
    # require_obs=True when user explicitly requested OBS (prevents silent fallback)
    manager = CaptureManager(
        window_name=config.get('window_name', ''),
        prefer_obs=config.get('prefer_obs', True),
        obs_device_index=config.get('obs_device_index'),
        require_obs=config.get('require_obs', False),
    )

    if not manager.start():
        print(f"[{pid}] ERROR: Failed to start capture")
        return

    print(f"[{pid}] Capture started via: {manager.backend.name}")

    target_fps = config.get('capture_fps', 60)
    frame_count = 0

    try:
        for frame in manager.capture_loop(target_fps=target_fps):
            if shutdown_event.is_set():
                break

            # Put frame on queue (same interface as legacy capture)
            try:
                raw_frame_queue.put_nowait((frame.timestamp, frame.data))
                frame_count += 1
            except:
                # Queue full, drop frame
                pass

            if frame_count % 600 == 0:  # Log every 10 seconds at 60fps
                print(f"[{pid}] Captured {frame_count} frames via {manager.backend.name}")

    finally:
        manager.stop()
        print(f"[{pid}] OBS Capture finished. Total frames: {frame_count}")


# =============================================================================
#  CLI FOR TESTING
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test OBS capture backends")
    parser.add_argument("--backend", choices=["obs", "zmq", "legacy", "auto"],
                        default="auto", help="Capture backend to use")
    parser.add_argument("--window", type=str, default="",
                        help="Window name for legacy capture")
    parser.add_argument("--device", type=int, default=None,
                        help="Video device index for OBS Virtual Camera")
    parser.add_argument("--fps", type=int, default=60, help="Target FPS")
    parser.add_argument("--duration", type=int, default=10, help="Test duration in seconds")

    args = parser.parse_args()

    print("=" * 60)
    print("  OBS Capture Test")
    print("=" * 60)

    manager = CaptureManager(
        window_name=args.window,
        prefer_obs=(args.backend in ["obs", "auto"]),
        obs_device_index=args.device,
    )

    if not manager.start():
        print("Failed to start capture!")
        sys.exit(1)

    print(f"\nCapturing for {args.duration} seconds at {args.fps} FPS...")
    print("Press Ctrl+C to stop early.\n")

    frame_count = 0
    start_time = time.time()

    try:
        for frame in manager.capture_loop(target_fps=args.fps):
            frame_count += 1

            if frame_count % args.fps == 0:
                elapsed = time.time() - start_time
                actual_fps = frame_count / elapsed
                print(f"Frames: {frame_count}, Actual FPS: {actual_fps:.1f}, "
                      f"Size: {frame.width}x{frame.height}")

            if time.time() - start_time > args.duration:
                break

    except KeyboardInterrupt:
        print("\nInterrupted!")

    finally:
        manager.stop()

    elapsed = time.time() - start_time
    print(f"\nCaptured {frame_count} frames in {elapsed:.1f}s ({frame_count/elapsed:.1f} FPS)")
