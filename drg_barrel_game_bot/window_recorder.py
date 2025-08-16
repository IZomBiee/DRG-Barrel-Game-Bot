import numpy as np
import pygetwindow
from threading import Thread, Lock
from time import sleep
from mss import mss
from screeninfo import get_monitors

from .utils import *
from .setting_loader import SettingLoader as SL

class WindowRecorder:
    """Threaded window recorder using mss."""
    def __init__(self):
        monitors = get_monitors()[SL()['program']['monitor_id']]
        self.region = {
            'top': 0,
            'left': 0,
            'width': monitors.width,
            'height': monitors.height,
        }
        self.target_window_title = SL()['program']['window_title']
        self.y_gap = SL()['display']['basket_y_gap']
        self.update_region()

        # Threaded capture
        self._frame = None
        self._lock = Lock()
        self._updated = False
        self._stop = False
        self._thread = Thread(target=self._capture_loop, daemon=True)
        self._thread.start()

        print(f"Initialized WindowRecorder with title '{self.target_window_title}' and region: {self.region}")

    def update_region(self) -> None:
        """Update region based on window position."""
        window = pygetwindow.getWindowsWithTitle(self.target_window_title)
        if window:
            window: pygetwindow.Window = window[0]
            x, y, w, h = window.box
            crop_each_side = (h * self.y_gap) / 2

            self.region = {
                'top': int(y + crop_each_side),
                'left': x,
                'width': w,
                'height': int(h - self.y_gap * h),
            }
        else:
            print("Can't find window!")

    def _capture_loop(self):
        """Background capture thread."""
        with mss() as sct:
            while not self._stop:
                screenshot = sct.grab(self.region)
                frame = np.array(screenshot)

                with self._lock:
                    if self._frame is None or not np.array_equal(frame, self._frame):
                        self._updated = True
                        self._frame = frame

    def get_frame(self) -> np.ndarray | None:
        """Return the latest frame and mark it as read (reset _updated)."""
        with self._lock:
            if self._frame is None:
                return None
            return self._frame.copy()

    def is_updated(self) -> bool:
        """Check if the frame has changed since last check (before calling get_frame)."""
        with self._lock:
            return self._updated

    def stop(self):
        """Stop the background capture thread."""
        self._stop = True
        self._thread.join()
