import numpy as np
import pygetwindow

from .utils import *
from .setting_loader import SettingLoader as SL
from mss import mss
from screeninfo import get_monitors

class WindowRecorder:
    '''Help get frames from window with mss '''
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
        print("Initialized WindowRecorder with tilte "+
              f"{self.target_window_title} and region: {self.region}")

    def update_region(self) -> None:
        window = pygetwindow.getWindowsWithTitle(self.target_window_title) # type: ignore
        if window:
            window: pygetwindow.Window = window[0] # type: ignore
            x, y, w, h = window.box
            crop_each_side = (h * self.y_gap) / 2

            self.region = {
                'top': int(y + crop_each_side),
                'left': x,
                'width': w,
                'height': int(h - self.y_gap * h),
            }
        else: print("Cant find window!")

    def get_frame(self) -> np.ndarray:
        with mss() as sct:
            screenshot = sct.grab(self.region)
        return np.array(screenshot)