# window_utils.py

import sys
import time

class WindowNotFoundError(Exception):
    """Custom exception for when a window cannot be found."""
    pass

# --- Platform-Specific Backends ---

class BaseWindowFinder:
    """Base class defining the window finder interface."""
    def find_window_by_title(self, title: str) -> dict | None:
        """
        Finds a window by its title and returns its geometry.
        Returns: A dict like {"top": Y, "left": X, "width": W, "height": H} or None.
        """
        raise NotImplementedError

    def is_available(self) -> bool:
        """Checks if the backend's dependencies are installed."""
        return False

# --- Windows Backend (pywin32) ---
class Win32WindowFinder(BaseWindowFinder):
    def __init__(self):
        self._win32gui = None
        try:
            import win32gui
            self._win32gui = win32gui
        except ImportError:
            print("WARNING: pywin32 not installed. Window finding on Windows is disabled.")
            print("Install with: uv pip install -e .[win-capture]")

    def is_available(self) -> bool:
        return self._win32gui is not None

    def find_window_by_title(self, title: str) -> dict | None:
        if not self.is_available():
            return None
            
        hwnd = self._win32gui.FindWindow(None, title)
        if hwnd == 0:
            return None
        
        # Get the window's client area rectangle (the part we can actually see)
        left, top, right, bottom = self._win32gui.GetClientRect(hwnd)
        
        # Convert client area coordinates to screen coordinates
        client_left, client_top = self._win32gui.ClientToScreen(hwnd, (left, top))
        client_right, client_bottom = self._win32gui.ClientToScreen(hwnd, (right, bottom))

        width = client_right - client_left
        height = client_bottom - client_top
        
        # Sometimes GetClientRect can return negative values for minimized windows
        if width <= 0 or height <= 0:
            return None

        return {"top": client_top, "left": client_left, "width": width, "height": height}

# --- Linux Backend (X11/EWMH) ---
# --- Linux Backend (X11/EWMH) ---
class X11WindowFinder(BaseWindowFinder):
    def __init__(self):
        self._ewmh = None
        try:
            from ewmh import EWMH
            self._ewmh = EWMH()
        except (ImportError, Exception): # EWMH can fail if no X server is running
            print("WARNING: python-xlib or ewmh not installed/functional. Window finding on Linux is disabled.")
            print("Install with: uv pip install -e .[linux-capture]")

    def is_available(self) -> bool:
        return self._ewmh is not None

    def find_window_by_title(self, title: str) -> dict | None:
        if not self.is_available():
            return None

        clients = self._ewmh.getClientList()
        target_window = None

        for window in clients:
            try:
                window_title = self._ewmh.getWmName(window)
                if window_title and title in window_title.decode('utf-8', 'ignore'):
                    target_window = window
                    break
            except (AttributeError, TypeError):
                continue
        
        if target_window is None:
            return None

        try:
            # Get geometry relative to the root window (the whole screen)
            geo = target_window.get_geometry()
            # This needs translation to absolute screen coordinates
            parent = target_window.query_tree().parent
            x, y = target_window.translate_coords(parent, 0, 0).x, target_window.translate_coords(parent, 0, 0).y
            
            # Adjust for window decorations (title bar, borders) if possible.
            # This is complex; for now, we use the full window geometry.
            return {"top": y, "left": x, "width": geo.width, "height": geo.height}
        except Exception: # Window might close between find and get_geometry
            return None

# --- Factory Function ---
def get_window_finder():
    """Returns the appropriate window finder for the current OS."""
    if sys.platform == "win32":
        return Win32WindowFinder()
    elif sys.platform == "linux":
        return X11WindowFinder()
    # macOS would require its own implementation using pyobjc or similar
    else:
        print(f"WARNING: Window finding is not implemented for '{sys.platform}'.")
        return BaseWindowFinder()