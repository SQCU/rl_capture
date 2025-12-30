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
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple, Any
from multiprocessing import shared_memory
from threading import Lock

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
                self.pages[page_id].state = PageState.ENCODING_INTERESTING

    def evict_page(self, page_id: str) -> Optional[shared_memory.SharedMemory]:
        """Remove a page from tracking and return its SHM for cleanup."""
        with self.lock:
            if page_id not in self.pages:
                return None

            page = self.pages.pop(page_id)
            page.state = PageState.EVICTED

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

            return {
                "active_pages": len(self.pages),
                "pending_work": len(self.work_queue),
                "pages_triaged": self.pages_triaged,
                "pages_evicted_boring": self.pages_evicted_boring,
                "pages_retained_interesting": self.pages_retained_interesting,
                "state_distribution": state_counts,
                "current_refinement": self.current_refinement_page,
            }


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
