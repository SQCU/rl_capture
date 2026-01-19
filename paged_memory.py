# paged_memory.py
"""
Paged memory model for salience-driven frame capture.

Key insight: The scheduler should optimize for EVICTION LATENCY of boring content,
not PRECISION on interesting content. Most frames are boring - shed them fast.

Architecture:
- Pages are small (1-2s, 60-120 frames) instead of large chunks (12s, 720 frames)
- Two-phase triage: coarse sentinel check decides retain/evict immediately
- Global scheduler prioritizes new page triage over refinement of existing pages
- Boring pages are evicted immediately after baseline encode
- Pages ARE the framebuffer - no duplication between triage and encoding
"""

import numpy as np
import time
import uuid
import heapq
from collections import deque
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple, Any
from multiprocessing import shared_memory
from threading import Lock


# =============================================================================
#  FLOW MONITOR: Derivative-based resource tracking
# =============================================================================

class FlowMonitor:
    """
    Track bytes/sec derivatives across processing stages.

    Instead of threshold-based limits ("if queue > 10, drop"):
    - Track rate of change (d_bytes/d_t) per stage
    - Detect sustained positive derivatives → "accumulating unsustainably"
    - Predict time-to-overflow based on current rate
    - Enable early intervention before hitting hard limits

    The metadata cost is trivial (few KB) vs megabytes of actual buffers.
    """

    def __init__(self, window_sec: float = 5.0):
        """
        Args:
            window_sec: Time window for derivative calculation
        """
        self.window = window_sec
        self.flows: Dict[str, deque] = {}  # stage -> [(timestamp, net_bytes)]
        self.stage_current: Dict[str, int] = {}  # stage -> current bytes held
        self.lock = Lock()

    def record(self, stage: str, bytes_in: int = 0, bytes_out: int = 0) -> None:
        """
        Record a flow event (bytes entering or leaving a stage).

        Args:
            stage: Processing stage name (e.g., 'pending', 'intret', 'encqueue')
            bytes_in: Bytes entering this stage
            bytes_out: Bytes leaving this stage
        """
        now = time.monotonic()
        net = bytes_in - bytes_out

        with self.lock:
            if stage not in self.flows:
                self.flows[stage] = deque(maxlen=500)
                self.stage_current[stage] = 0

            self.flows[stage].append((now, net))
            self.stage_current[stage] = max(0, self.stage_current.get(stage, 0) + net)

    def record_pages(self, stage: str, pages_in: int = 0, pages_out: int = 0,
                     bytes_per_page: int = 640*480*3*480) -> None:
        """
        Convenience: record page-level flow (auto-converts to bytes).

        Default bytes_per_page assumes 640x480 RGB @ 480 frames/page (8s @ 60fps).
        """
        self.record(stage,
                    bytes_in=pages_in * bytes_per_page,
                    bytes_out=pages_out * bytes_per_page)

    def get_rate(self, stage: str) -> float:
        """
        Get current bytes/sec accumulation rate for a stage.

        Returns:
            Positive = growing (bad), Negative = draining (good), Zero = stable
        """
        with self.lock:
            if stage not in self.flows:
                return 0.0

            now = time.monotonic()
            cutoff = now - self.window
            events = self.flows[stage]

            # Sum deltas within window
            net = sum(delta for ts, delta in events if ts > cutoff)
            return net / self.window

    def get_current(self, stage: str) -> int:
        """Get current bytes held at a stage."""
        with self.lock:
            return self.stage_current.get(stage, 0)

    def time_to_overflow(self, stage: str, max_bytes: int) -> float:
        """
        Estimate seconds until overflow at current accumulation rate.

        Args:
            stage: Processing stage
            max_bytes: Maximum allowed bytes for this stage

        Returns:
            Seconds until overflow. inf if draining or stable.
        """
        rate = self.get_rate(stage)
        if rate <= 0:
            return float('inf')

        current = self.get_current(stage)
        remaining = max_bytes - current
        if remaining <= 0:
            return 0.0

        return remaining / rate

    def is_unsustainable(self, stage: str, threshold_bytes_per_sec: float = 1e6,
                         min_duration_sec: float = 3.0) -> bool:
        """
        Check if a stage has been accumulating unsustainably.

        Args:
            stage: Processing stage
            threshold_bytes_per_sec: Rate above which accumulation is "unsustainable"
            min_duration_sec: How long the rate must persist to trigger

        Returns:
            True if stage is accumulating faster than threshold for min_duration
        """
        with self.lock:
            if stage not in self.flows:
                return False

            now = time.monotonic()
            events = list(self.flows[stage])

        # Check if rate has been positive for min_duration
        if len(events) < 2:
            return False

        # Find earliest event in our window
        cutoff = now - min_duration_sec
        window_events = [(ts, delta) for ts, delta in events if ts > cutoff]
        if len(window_events) < 2:
            return False

        # Check rate over this period
        duration = now - window_events[0][0]
        if duration < min_duration_sec * 0.8:  # Allow some slack
            return False

        total_net = sum(delta for _, delta in window_events)
        rate = total_net / duration

        return rate > threshold_bytes_per_sec

    def get_stats(self) -> Dict[str, Dict[str, float]]:
        """
        Get stats for all monitored stages.

        Returns:
            {stage: {"rate_mb_s": float, "current_mb": float, "tto_sec": float or inf}}
        """
        with self.lock:
            stages = list(self.flows.keys())

        stats = {}
        for stage in stages:
            rate = self.get_rate(stage)
            current = self.get_current(stage)
            # Estimate overflow assuming ~2GB max per stage
            tto = self.time_to_overflow(stage, max_bytes=2 * 1024**3)

            stats[stage] = {
                "rate_mb_s": rate / (1024**2),
                "current_mb": current / (1024**2),
                "tto_sec": tto if tto != float('inf') else -1,  # -1 = stable/draining
            }

        return stats

    def format_status(self, stage: str, max_bytes: int = 2 * 1024**3) -> str:
        """
        Format a compact status string for a stage.

        Returns something like: "+2.1MB/s ~47s" or "-0.5MB/s ok" or "stable"
        """
        rate = self.get_rate(stage)
        rate_mb = rate / (1024**2)

        if abs(rate_mb) < 0.1:
            return "stable"

        tto = self.time_to_overflow(stage, max_bytes)

        if rate > 0:
            if tto < float('inf'):
                return f"+{rate_mb:.1f}MB/s ~{int(tto)}s"
            else:
                return f"+{rate_mb:.1f}MB/s"
        else:
            return f"{rate_mb:.1f}MB/s ok"


# =============================================================================
#  CONFIGURATION: Paged Memory Model
# =============================================================================

PAGE_SECONDS = 8.0  # Match ML processing throughput (~8-10s per triage cycle)
PAGE_FRAMES_AT_60FPS = int(PAGE_SECONDS * 60)  # 480 frames per page at 60fps

# Coarse triage uses very few sentinels to decide retain/evict
COARSE_TRIAGE_SENTINELS = 6  # ~6 forwards per page for initial decision
COARSE_TRIAGE_NOVELTY_THRESHOLD = 0.3  # Higher threshold = more aggressive eviction

# Fine refinement (only for retained pages, preemptible)
FINE_REFINEMENT_MAX_P = 0.15  # Budget for detailed search on interesting pages

# Memory limits
MAX_RETAINED_PAGES = 8  # Max interesting pages held for refinement
MAX_PENDING_TRIAGE = 4  # Max pages waiting for coarse triage

# Baseline encoder decimation
BASELINE_TARGET_FPS = 30  # Decimate to 30fps for baseline recording
BASELINE_FRAME_INTERVAL = 2  # At 60fps capture, keep every 2nd frame for 30fps baseline


# =============================================================================
#  PAGE STATE MACHINE
# =============================================================================

class PageState(Enum):
    """Lifecycle states for a memory page."""
    PENDING_TRIAGE = auto()      # Just arrived, needs coarse sentinel check
    TRIAGING = auto()            # Currently being triaged (coarse pass)
    BORING_EVICTING = auto()     # Deemed boring, feeding to baseline encoder, then free
    INTERESTING_RETAINED = auto() # Deemed interesting, retained for refinement
    REFINING = auto()            # Currently being refined (fine pass, preemptible)
    ENCODING_INTERESTING = auto() # Interesting content being encoded at high quality
    EVICTED = auto()             # Memory freed


@dataclass
class PageMetadata:
    """Metadata for a single memory page."""
    page_id: str
    shm_name: str
    shape: Tuple[int, ...]
    dtype: np.dtype
    timestamps: List[float]
    state: PageState
    created_at: float
    last_activity: float

    # Triage results (populated after coarse pass)
    coarse_novelty_score: Optional[float] = None
    sentinel_indices: Optional[List[int]] = None
    hot_regions: Optional[List[Tuple[int, int]]] = None  # (start_idx, end_idx) pairs

    # Refinement results (populated during fine pass)
    keyframe_events: List[dict] = field(default_factory=list)
    refinement_depth: int = 0

    # Priority for scheduling (lower = higher priority)
    # New pages get priority 0, retained pages get priority based on novelty
    priority: float = 0.0


# =============================================================================
#  GLOBAL SCHEDULER
# =============================================================================

class WorkType(Enum):
    """Types of work the scheduler can dispatch."""
    COARSE_TRIAGE = auto()   # Highest priority: decide retain/evict for new page
    FINE_REFINEMENT = auto()  # Lower priority: detailed search on interesting page
    EVICTION = auto()         # Cleanup: free boring page after baseline encode


@dataclass(order=True)
class WorkItem:
    """A unit of work for the scheduler."""
    priority: float  # Lower = higher priority (for heapq min-heap)
    work_type: WorkType = field(compare=False)
    page_id: str = field(compare=False)
    created_at: float = field(compare=False, default_factory=time.time)


class GlobalScheduler:
    """
    Priority-based scheduler for page processing.

    Priority order:
    1. COARSE_TRIAGE of new pages (priority 0.0) - minimize eviction latency
    2. FINE_REFINEMENT of interesting pages (priority 1.0 + inverse_novelty)
    3. EVICTION cleanup (priority 2.0) - free memory

    Refinement is PREEMPTIBLE: if a new page arrives during refinement,
    the refinement yields and triage runs first.
    """

    def __init__(self):
        self.work_queue: List[WorkItem] = []  # heapq min-heap
        self.pages: Dict[str, PageMetadata] = {}
        self.shm_handles: Dict[str, shared_memory.SharedMemory] = {}  # Keep handles alive!
        self.lock = Lock()

        # Tracking for preemption
        self.current_refinement_page: Optional[str] = None
        self.preemption_requested = False

        # Flow monitor for derivative-based resource tracking
        self.flow_monitor = FlowMonitor(window_sec=5.0)

        # Stats
        self.pages_triaged = 0
        self.pages_evicted_boring = 0
        self.pages_retained_interesting = 0

    def register_page(self, page: PageMetadata, shm_handle: Optional[shared_memory.SharedMemory] = None) -> None:
        """
        Register a new page and schedule it for triage.

        IMPORTANT: On Windows, pass shm_handle to keep the SHM alive.
        Without a live handle, Windows will destroy the SHM when the creator closes it.
        """
        with self.lock:
            self.pages[page.page_id] = page

            # Store the SHM handle to keep it alive (critical for Windows!)
            if shm_handle is not None:
                self.shm_handles[page.page_id] = shm_handle

            # Track flow: page entering pending stage
            page_bytes = int(np.prod(page.shape)) * np.dtype(page.dtype).itemsize
            self.flow_monitor.record('pending', bytes_in=page_bytes)

            # Immediately schedule coarse triage (highest priority)
            work = WorkItem(
                priority=0.0,  # Triage always wins
                work_type=WorkType.COARSE_TRIAGE,
                page_id=page.page_id
            )
            heapq.heappush(self.work_queue, work)

            # If we're currently refining, request preemption
            if self.current_refinement_page is not None:
                self.preemption_requested = True

    def schedule_refinement(self, page_id: str, novelty_score: float) -> None:
        """Schedule a page for fine refinement after coarse triage deemed it interesting."""
        with self.lock:
            if page_id not in self.pages:
                return

            page = self.pages[page_id]
            page.state = PageState.INTERESTING_RETAINED
            page.coarse_novelty_score = novelty_score

            # Track flow: page moving pending → interesting_retained
            page_bytes = int(np.prod(page.shape)) * np.dtype(page.dtype).itemsize
            self.flow_monitor.record('pending', bytes_out=page_bytes)
            self.flow_monitor.record('intret', bytes_in=page_bytes)

            # Priority: 1.0 base + inverse novelty (more novel = lower priority number = process sooner)
            # But still lower priority than any triage (which is 0.x)
            priority = 1.0 + (1.0 - min(novelty_score, 1.0))

            work = WorkItem(
                priority=priority,
                work_type=WorkType.FINE_REFINEMENT,
                page_id=page_id
            )
            heapq.heappush(self.work_queue, work)
            self.pages_retained_interesting += 1

    def schedule_eviction(self, page_id: str) -> None:
        """Schedule a boring page for eviction after baseline encode completes."""
        with self.lock:
            if page_id not in self.pages:
                return

            page = self.pages[page_id]

            # Track flow: page leaving pending stage (boring eviction)
            page_bytes = int(np.prod(page.shape)) * np.dtype(page.dtype).itemsize
            self.flow_monitor.record('pending', bytes_out=page_bytes)

            page.state = PageState.BORING_EVICTING

            work = WorkItem(
                priority=2.0,  # Lowest priority
                work_type=WorkType.EVICTION,
                page_id=page_id
            )
            heapq.heappush(self.work_queue, work)
            self.pages_evicted_boring += 1

    def get_next_work(self, prefer_refinement: bool = False) -> Optional[WorkItem]:
        """Get the highest priority work item.

        Args:
            prefer_refinement: If True, skip triage work and return refinement work first.
                              This prevents refinement starvation when pages pile up.
        """
        with self.lock:
            # If preferring refinement, first scan for refinement work
            if prefer_refinement:
                for i, work in enumerate(self.work_queue):
                    if work.work_type == WorkType.FINE_REFINEMENT:
                        if work.page_id in self.pages:
                            page = self.pages[work.page_id]
                            if page.state == PageState.INTERESTING_RETAINED:
                                # Remove from queue and return
                                self.work_queue.pop(i)
                                heapq.heapify(self.work_queue)
                                page.state = PageState.REFINING
                                self.current_refinement_page = work.page_id
                                return work

            # Normal priority-based dispatch
            while self.work_queue:
                work = heapq.heappop(self.work_queue)

                # Validate page still exists and is in appropriate state
                if work.page_id not in self.pages:
                    continue

                page = self.pages[work.page_id]

                if work.work_type == WorkType.COARSE_TRIAGE:
                    if page.state == PageState.PENDING_TRIAGE:
                        page.state = PageState.TRIAGING
                        return work

                elif work.work_type == WorkType.FINE_REFINEMENT:
                    if page.state == PageState.INTERESTING_RETAINED:
                        page.state = PageState.REFINING
                        self.current_refinement_page = work.page_id
                        return work

                elif work.work_type == WorkType.EVICTION:
                    if page.state == PageState.BORING_EVICTING:
                        return work

            return None

    def should_preempt_refinement(self) -> bool:
        """Check if current refinement should yield to higher priority work."""
        with self.lock:
            if not self.preemption_requested:
                return False

            # Check if there's triage work waiting
            for work in self.work_queue:
                if work.work_type == WorkType.COARSE_TRIAGE:
                    return True

            self.preemption_requested = False
            return False

    def pause_refinement(self, page_id: str) -> None:
        """Pause refinement of a page (for preemption)."""
        with self.lock:
            if page_id in self.pages:
                page = self.pages[page_id]
                if page.state == PageState.REFINING:
                    page.state = PageState.INTERESTING_RETAINED
                    # Re-queue with slightly higher priority (it already started)
                    work = WorkItem(
                        priority=0.9,  # Just below triage priority
                        work_type=WorkType.FINE_REFINEMENT,
                        page_id=page_id
                    )
                    heapq.heappush(self.work_queue, work)

            self.current_refinement_page = None
            self.preemption_requested = False

    def complete_refinement(self, page_id: str) -> None:
        """Mark refinement as complete."""
        with self.lock:
            self.current_refinement_page = None
            if page_id in self.pages:
                page = self.pages[page_id]
                # Track flow: page leaving intret stage after refinement
                page_bytes = int(np.prod(page.shape)) * np.dtype(page.dtype).itemsize
                self.flow_monitor.record('intret', bytes_out=page_bytes)
                page.state = PageState.ENCODING_INTERESTING

    def evict_page(self, page_id: str) -> Optional[shared_memory.SharedMemory]:
        """Remove a page from tracking and return its SHM for cleanup."""
        with self.lock:
            if page_id not in self.pages:
                return None

            page = self.pages.pop(page_id)
            prev_state = page.state
            page.state = PageState.EVICTED

            # Track flow based on previous state
            page_bytes = int(np.prod(page.shape)) * np.dtype(page.dtype).itemsize
            if prev_state == PageState.INTERESTING_RETAINED:
                # Force-evicting an unrefined interesting page
                self.flow_monitor.record('intret', bytes_out=page_bytes)
            # Note: BORING_EVICTING already tracked bytes_out in schedule_eviction
            # and ENCODING_INTERESTING already tracked in complete_refinement

            # Use stored handle if available (critical for Windows!)
            if page_id in self.shm_handles:
                shm = self.shm_handles.pop(page_id)
                return shm

            # Fallback: try to open by name (may fail on Windows if handle was closed)
            try:
                shm = shared_memory.SharedMemory(name=page.shm_name)
                return shm
            except FileNotFoundError:
                return None

    def get_stats(self) -> dict:
        """Get scheduler statistics."""
        with self.lock:
            state_counts = {}
            for page in self.pages.values():
                state_name = page.state.name
                state_counts[state_name] = state_counts.get(state_name, 0) + 1

            # Get flow stats
            flow_stats = self.flow_monitor.get_stats()

            return {
                "active_pages": len(self.pages),
                "pending_work": len(self.work_queue),
                "pages_triaged": self.pages_triaged,
                "pages_evicted_boring": self.pages_evicted_boring,
                "pages_retained_interesting": self.pages_retained_interesting,
                "state_distribution": state_counts,
                "current_refinement": self.current_refinement_page,
                "flow": flow_stats,
            }

    def get_stale_intret_pages(self, max_age_sec: float = 60.0) -> List[str]:
        """
        Get list of INTERESTING_RETAINED pages that have been waiting too long.

        Used for derivative-based eviction: if intret is accumulating unsustainably,
        these are candidates for forced eviction (oldest first).
        """
        now = time.time()
        stale = []
        with self.lock:
            for page_id, page in self.pages.items():
                if page.state == PageState.INTERESTING_RETAINED:
                    age = now - page.created_at
                    if age > max_age_sec:
                        stale.append((age, page_id, page.coarse_novelty_score or 0))

        # Sort by age descending (oldest first), then by novelty ascending (least novel first)
        stale.sort(key=lambda x: (-x[0], x[2]))
        return [page_id for _, page_id, _ in stale]

    def force_evict_stale(self, max_to_evict: int = 2) -> List[str]:
        """
        Force-evict the oldest/least-interesting retained pages.

        Called when intret stage is accumulating unsustainably.
        Returns list of evicted page IDs.
        """
        stale = self.get_stale_intret_pages(max_age_sec=30.0)
        evicted = []

        for page_id in stale[:max_to_evict]:
            shm = self.evict_page(page_id)
            if shm:
                try:
                    shm.close()
                    shm.unlink()
                except:
                    pass
                evicted.append(page_id)

        return evicted


# =============================================================================
#  PAGE FACTORY
# =============================================================================

def create_page_from_frames(
    frames: List[np.ndarray],
    timestamps: List[float],
) -> Tuple[PageMetadata, shared_memory.SharedMemory]:
    """
    Create a new page from a list of frames.

    Returns:
        Tuple of (PageMetadata, SharedMemory handle for caller to manage)
    """
    page_id = f"page_{uuid.uuid4().hex[:12]}"
    shm_name = f"shm_{page_id}"

    # Create SHM block
    buffer_shape = (len(frames), *frames[0].shape)
    buffer_size = int(np.prod(buffer_shape) * np.dtype(np.uint8).itemsize)

    shm = shared_memory.SharedMemory(name=shm_name, create=True, size=buffer_size)
    shm_buffer = np.ndarray(buffer_shape, dtype=np.uint8, buffer=shm.buf)
    np.copyto(shm_buffer, np.array(frames))

    now = time.time()
    page = PageMetadata(
        page_id=page_id,
        shm_name=shm_name,
        shape=buffer_shape,
        dtype=np.uint8,
        timestamps=list(timestamps),
        state=PageState.PENDING_TRIAGE,
        created_at=now,
        last_activity=now,
    )

    return page, shm


def get_decimated_frames_for_baseline(
    shm_name: str,
    shape: Tuple[int, ...],
    dtype: np.dtype,
    timestamps: List[float],
    target_fps: int = BASELINE_TARGET_FPS,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract decimated frames from a page for baseline encoding.

    This avoids buffer duplication - we read directly from the page's SHM
    and return only the frames needed for the target framerate.

    Returns:
        Tuple of (decimated_frames, decimated_timestamps)
    """
    shm = shared_memory.SharedMemory(name=shm_name)
    buffer = np.ndarray(shape, dtype=dtype, buffer=shm.buf)

    # Calculate actual FPS from timestamps
    if len(timestamps) < 2:
        shm.close()
        return buffer, np.array(timestamps)

    actual_duration = timestamps[-1] - timestamps[0]
    actual_fps = (len(timestamps) - 1) / actual_duration if actual_duration > 0 else 60.0

    # Calculate decimation interval
    if actual_fps <= target_fps:
        # Already at or below target, keep all frames
        indices = np.arange(len(timestamps))
    else:
        # Decimate: keep every Nth frame
        decimation_factor = int(np.ceil(actual_fps / target_fps))
        indices = np.arange(0, len(timestamps), decimation_factor)

    decimated_frames = buffer[indices].copy()  # Copy because we're closing SHM
    decimated_timestamps = np.array(timestamps)[indices]

    shm.close()
    return decimated_frames, decimated_timestamps
