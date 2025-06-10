import numpy as np
import win32gui
import pygetwindow

from .toml_setting_loader import TOMLSettingLoader as TSL
from mss import mss
from screeninfo import get_monitors

class WindowRecorder:
    '''Help get frames from window with mss '''
    def __init__(self):  
        monitors = get_monitors()[TSL()['program']['monitor_id']]
        self.region = {
            'top': 0,
            'left': 0,
            'width': monitors.width,
            'height': monitors.height,
        }
        self.target_window_title = TSL()['program']['window_title']
        self.update_region()

    def update_region(self) -> None:
        window = pygetwindow.getWindowsWithTitle(self.target_window_title)
        if window:
            window: pygetwindow.Window = window[0]
            y, x, w, h = window.box
            self.region = {
                'top': x,
                'left': y,
                'width': w,
                'height': h,
            }
        else: print("Cant find window!")

    def get_frame(self) -> np.ndarray:
        with mss() as sct:
            screenshot = sct.grab(self.region)
        return np.array(screenshot)