# visualize_session.py
"""
Overlay visualizer for action-state recordings.

Composites mouse trajectories, clicks, and key events onto the baseline video.
"""

import argparse
import json
import os
import cv2
import numpy as np
from dataclasses import dataclass
from typing import Optional
from collections import deque


@dataclass
class TrajectoryEvent:
    """A mouse trajectory with timing info."""
    start_time: float
    end_time: float
    points: list[tuple[int, int]]
    button_held: Optional[str]
    linearity: float


@dataclass
class ClickEvent:
    """A mouse click."""
    timestamp: float
    x: int
    y: int
    button: str


@dataclass
class KeyEvent:
    """A key press with duration."""
    start_time: float
    duration: float
    key: str


@dataclass
class ScrollEvent:
    """A scroll gesture."""
    start_time: float
    end_time: float
    total_dx: int
    total_dy: int


def load_events(jsonl_path: str) -> tuple[list, list, list, list]:
    """Load and parse events from JSONL file."""
    trajectories = []
    clicks = []
    keys = []
    scrolls = []

    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                event = json.loads(line.strip())
                if event.get('stream_type') != 'USER_INPUT':
                    continue

                payload = json.loads(event.get('payload_json', '{}'))
                start_ts = event.get('start_timestamp', 0)
                delta = event.get('delta_timestamp', 0)

                event_type = payload.get('type')

                if event_type == 'mouse_trajectory':
                    trajectories.append(TrajectoryEvent(
                        start_time=start_ts,
                        end_time=start_ts + delta,
                        points=[(p[0], p[1]) for p in payload.get('sample_points', [])],
                        button_held=payload.get('button_held'),
                        linearity=payload.get('linearity', 1.0),
                    ))

                elif event_type == 'mouse_click':
                    clicks.append(ClickEvent(
                        timestamp=start_ts,
                        x=payload.get('x', 0),
                        y=payload.get('y', 0),
                        button=payload.get('button', 'unknown'),
                    ))

                elif event_type == 'key':
                    key_str = payload.get('key', '?').strip("'")
                    keys.append(KeyEvent(
                        start_time=start_ts,
                        duration=delta,
                        key=key_str,
                    ))

                elif event_type == 'scroll_trajectory':
                    scrolls.append(ScrollEvent(
                        start_time=start_ts,
                        end_time=start_ts + delta,
                        total_dx=payload.get('total_dx', 0),
                        total_dy=payload.get('total_dy', 0),
                    ))

            except (json.JSONDecodeError, KeyError):
                continue

    return trajectories, clicks, keys, scrolls


def find_video_file(capture_dir: str) -> Optional[str]:
    """Find the baseline video in the capture directory."""
    videos_dir = os.path.join(capture_dir, 'videos')
    if not os.path.exists(videos_dir):
        return None

    # Look for baseline video
    for f in os.listdir(videos_dir):
        if f.startswith('baseline_') and f.endswith('.mp4'):
            return os.path.join(videos_dir, f)

    # Fallback to any mp4
    for f in os.listdir(videos_dir):
        if f.endswith('.mp4'):
            return os.path.join(videos_dir, f)

    return None


def get_video_time_range(video_path: str) -> tuple[float, float, float]:
    """Get video duration and estimate start time from filename."""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps if fps > 0 else 0
    cap.release()

    # Try to extract start time from filename (baseline_record_TIMESTAMP.mp4)
    basename = os.path.basename(video_path)
    try:
        if 'baseline_record_' in basename:
            ts_str = basename.replace('baseline_record_', '').replace('.mp4', '')
            start_time = float(ts_str)
            return start_time, start_time + duration, fps
    except ValueError:
        pass

    return 0, duration, fps


class OverlayRenderer:
    """Renders input events as overlays on video frames."""

    def __init__(self, frame_width: int, frame_height: int,
                 video_width: int, video_height: int):
        self.frame_w = frame_width
        self.frame_h = frame_height
        self.video_w = video_width
        self.video_h = video_height

        # Scale factors for mapping input coords to video coords
        self.scale_x = video_width / frame_width if frame_width else 1
        self.scale_y = video_height / frame_height if frame_height else 1

        # Recent keys for display (fade out over time)
        self.recent_keys: deque = deque(maxlen=10)

        # Trajectory trail (points with age for fading)
        self.trajectory_trail: deque = deque(maxlen=100)

        # Colors
        self.COLOR_TRAJECTORY = (0, 255, 255)  # Yellow
        self.COLOR_TRAJECTORY_DRAG = (255, 100, 100)  # Blue-ish (dragging)
        self.COLOR_CLICK_LEFT = (0, 255, 0)  # Green
        self.COLOR_CLICK_RIGHT = (0, 0, 255)  # Red
        self.COLOR_KEY_BG = (40, 40, 40)
        self.COLOR_KEY_TEXT = (255, 255, 255)
        self.COLOR_SCROLL = (255, 165, 0)  # Orange

    def scale_point(self, x: int, y: int) -> tuple[int, int]:
        """Scale input coordinates to video coordinates."""
        return (int(x * self.scale_x), int(y * self.scale_y))

    def render_trajectory(self, frame: np.ndarray, traj: TrajectoryEvent,
                         current_time: float, trail_duration: float = 2.0):
        """Render a mouse trajectory with fading trail."""
        if not traj.points:
            return

        # Calculate opacity based on age
        age = current_time - traj.end_time
        if age > trail_duration:
            return
        opacity = max(0, 1 - (age / trail_duration))

        # Choose color based on button state
        base_color = self.COLOR_TRAJECTORY_DRAG if traj.button_held else self.COLOR_TRAJECTORY
        color = tuple(int(c * opacity) for c in base_color)

        # Draw path
        scaled_points = [self.scale_point(p[0], p[1]) for p in traj.points]
        for i in range(len(scaled_points) - 1):
            thickness = max(1, int(3 * opacity))
            cv2.line(frame, scaled_points[i], scaled_points[i + 1], color, thickness)

        # Draw endpoint
        if scaled_points:
            end_pt = scaled_points[-1]
            radius = max(3, int(6 * opacity))
            cv2.circle(frame, end_pt, radius, color, -1)

    def render_click(self, frame: np.ndarray, click: ClickEvent,
                    current_time: float, display_duration: float = 1.0):
        """Render a click marker with fade out."""
        age = current_time - click.timestamp
        if age > display_duration or age < 0:
            return

        opacity = max(0, 1 - (age / display_duration))

        # Choose color based on button
        if 'left' in click.button.lower():
            base_color = self.COLOR_CLICK_LEFT
        else:
            base_color = self.COLOR_CLICK_RIGHT

        color = tuple(int(c * opacity) for c in base_color)

        pt = self.scale_point(click.x, click.y)

        # Expanding ring animation
        radius = int(10 + 20 * (age / display_duration))
        thickness = max(1, int(3 * opacity))
        cv2.circle(frame, pt, radius, color, thickness)

        # Center dot
        cv2.circle(frame, pt, 4, color, -1)

    def render_keys(self, frame: np.ndarray, keys: list[KeyEvent],
                   current_time: float, display_duration: float = 2.0):
        """Render recent key presses as text overlay."""
        # Filter to recent keys
        recent = [k for k in keys
                  if 0 <= current_time - k.start_time <= display_duration]

        if not recent:
            return

        # Build display string (most recent last)
        recent_sorted = sorted(recent, key=lambda k: k.start_time)
        key_strs = []
        for k in recent_sorted[-8:]:  # Last 8 keys
            key_name = k.key
            # Clean up special key names
            if key_name.startswith('Key.'):
                key_name = key_name[4:].upper()
            elif len(key_name) == 1:
                key_name = key_name.upper()
            key_strs.append(key_name)

        display_text = ' '.join(key_strs)

        # Draw background box
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        thickness = 2
        (text_w, text_h), baseline = cv2.getTextSize(display_text, font, font_scale, thickness)

        padding = 10
        box_x = 10
        box_y = frame.shape[0] - 50

        cv2.rectangle(frame,
                     (box_x, box_y - text_h - padding),
                     (box_x + text_w + padding * 2, box_y + padding),
                     self.COLOR_KEY_BG, -1)

        cv2.putText(frame, display_text,
                   (box_x + padding, box_y),
                   font, font_scale, self.COLOR_KEY_TEXT, thickness)

    def render_scroll(self, frame: np.ndarray, scroll: ScrollEvent,
                     current_time: float, display_duration: float = 1.0):
        """Render scroll indicator."""
        if current_time < scroll.start_time or current_time > scroll.end_time + display_duration:
            return

        age = current_time - scroll.end_time
        opacity = 1.0 if age < 0 else max(0, 1 - (age / display_duration))

        color = tuple(int(c * opacity) for c in self.COLOR_SCROLL)

        # Draw scroll indicator in top-right
        cx, cy = frame.shape[1] - 50, 50

        # Arrow indicating scroll direction
        if scroll.total_dy > 0:
            # Scroll up
            pts = np.array([[cx, cy - 20], [cx - 15, cy + 10], [cx + 15, cy + 10]], np.int32)
        else:
            # Scroll down
            pts = np.array([[cx, cy + 20], [cx - 15, cy - 10], [cx + 15, cy - 10]], np.int32)

        cv2.fillPoly(frame, [pts], color)


def render_video_with_overlay(capture_dir: str, output_path: Optional[str] = None,
                              show_preview: bool = True, max_frames: Optional[int] = None):
    """Main function to render video with input overlay."""

    # Find files
    events_path = os.path.join(capture_dir, 'events_stream.jsonl')
    if not os.path.exists(events_path):
        print(f"Error: No events_stream.jsonl in {capture_dir}")
        return

    video_path = find_video_file(capture_dir)
    if not video_path:
        print(f"Error: No video file found in {capture_dir}/videos/")
        return

    print(f"Loading events from: {events_path}")
    print(f"Loading video from: {video_path}")

    # Load events
    trajectories, clicks, keys, scrolls = load_events(events_path)
    print(f"Loaded: {len(trajectories)} trajectories, {len(clicks)} clicks, "
          f"{len(keys)} keys, {len(scrolls)} scrolls")

    if not any([trajectories, clicks, keys, scrolls]):
        print("Warning: No input events found!")

    # Open video
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"Video: {video_w}x{video_h} @ {fps:.1f}fps, {frame_count} frames")

    # Get time range
    video_start, video_end, _ = get_video_time_range(video_path)
    print(f"Video time range: {video_start:.1f} - {video_end:.1f}")

    # Adjust event times if video has a start timestamp
    if video_start > 1000000000:  # Looks like a unix timestamp
        # Events are in absolute time, video filename has start time
        time_offset = video_start
    else:
        # Assume events and video start at same time
        # Use first event timestamp as reference
        all_times = ([t.start_time for t in trajectories] +
                    [c.timestamp for c in clicks] +
                    [k.start_time for k in keys])
        time_offset = min(all_times) if all_times else 0

    print(f"Time offset: {time_offset:.1f}")

    # Create renderer (assume input coords match a 1920x1080-ish screen)
    # This is a guess - could be made configurable
    renderer = OverlayRenderer(
        frame_width=2560,  # Assume typical screen res
        frame_height=1440,
        video_width=video_w,
        video_height=video_h
    )

    # Set up output video if requested
    writer = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (video_w, video_h))
        print(f"Writing to: {output_path}")

    # Process frames
    frame_idx = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if max_frames and frame_idx >= max_frames:
                break

            # Calculate current time
            current_time = time_offset + (frame_idx / fps)

            # Render overlays
            for traj in trajectories:
                renderer.render_trajectory(frame, traj, current_time)

            for click in clicks:
                renderer.render_click(frame, click, current_time)

            renderer.render_keys(frame, keys, current_time)

            for scroll in scrolls:
                renderer.render_scroll(frame, scroll, current_time)

            # Write or display
            if writer:
                writer.write(frame)

            if show_preview:
                # Resize for preview if too large
                preview = frame
                if video_w > 1280:
                    scale = 1280 / video_w
                    preview = cv2.resize(frame, None, fx=scale, fy=scale)

                cv2.imshow('Session Replay', preview)

                # Playback control
                key = cv2.waitKey(int(1000 / fps)) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord(' '):
                    # Pause
                    cv2.waitKey(0)

            frame_idx += 1
            if frame_idx % 100 == 0:
                print(f"Processed {frame_idx}/{frame_count} frames...", end='\r')

    finally:
        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()

    print(f"\nDone! Processed {frame_idx} frames.")


def generate_heatmap(capture_dir: str, output_path: str,
                    resolution: tuple[int, int] = (1920, 1080)):
    """Generate a mouse activity heatmap from the session."""

    events_path = os.path.join(capture_dir, 'events_stream.jsonl')
    trajectories, clicks, _, _ = load_events(events_path)

    # Create heatmap accumulator
    heatmap = np.zeros((resolution[1], resolution[0]), dtype=np.float32)

    # Accumulate trajectory points
    for traj in trajectories:
        for x, y in traj.points:
            if 0 <= x < resolution[0] and 0 <= y < resolution[1]:
                # Add gaussian blob
                cv2.circle(heatmap, (x, y), 20, 1.0, -1)

    # Accumulate clicks (weighted higher)
    for click in clicks:
        x, y = click.x, click.y
        if 0 <= x < resolution[0] and 0 <= y < resolution[1]:
            cv2.circle(heatmap, (x, y), 30, 5.0, -1)

    # Normalize and colorize
    if heatmap.max() > 0:
        heatmap = heatmap / heatmap.max()

    heatmap_color = cv2.applyColorMap((heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET)

    cv2.imwrite(output_path, heatmap_color)
    print(f"Heatmap saved to: {output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Visualize action-state recordings with input overlay',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Preview with overlay (press Q to quit, Space to pause)
  python visualize_session.py ./capture_run_12345

  # Render to new video file
  python visualize_session.py ./capture_run_12345 -o overlay_output.mp4

  # Generate heatmap only
  python visualize_session.py ./capture_run_12345 --heatmap heatmap.png

  # No preview, just render
  python visualize_session.py ./capture_run_12345 -o out.mp4 --no-preview
"""
    )

    parser.add_argument('capture_dir',
                        help='Path to capture_run_* directory')
    parser.add_argument('-o', '--output',
                        help='Output video path (renders overlay to file)')
    parser.add_argument('--no-preview', action='store_true',
                        help='Disable preview window')
    parser.add_argument('--heatmap', metavar='PATH',
                        help='Generate mouse heatmap image')
    parser.add_argument('--max-frames', type=int,
                        help='Limit number of frames to process')
    parser.add_argument('--screen-res', default='2560x1440',
                        help='Screen resolution for coordinate scaling (default: 2560x1440)')

    args = parser.parse_args()

    if args.heatmap:
        res = tuple(map(int, args.screen_res.split('x')))
        generate_heatmap(args.capture_dir, args.heatmap, resolution=res)
    else:
        render_video_with_overlay(
            args.capture_dir,
            output_path=args.output,
            show_preview=not args.no_preview,
            max_frames=args.max_frames
        )
