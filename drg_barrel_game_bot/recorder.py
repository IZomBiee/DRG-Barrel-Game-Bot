from mss import mss
import time
import numpy as np
from screeninfo import get_monitors

class Recorder():
    '''Help get frames with mss with fixed fps'''
    def __init__(self, region: tuple[int, int, int, int] | None = None,
                 target_fps: int = 30):  
        monitors = get_monitors()[0]
        self.region = {
            'top': region[0] if region else 0,
            'left': region[1] if region else 0,
            'width': region[2] if region else monitors.width,
            'height': region[3] if region else monitors.height,
        }
        self.target_fps = target_fps
        self.target_time = 1.0 / target_fps
        self.next_frame_time = time.perf_counter()

    def get_screenshot(self) -> np.ndarray:
        '''Getting screenshot fps uncaped'''
        with mss() as sct:
            screenshot = sct.grab(self.region)
        return np.array(screenshot)
                
    def get_last_frame(self) -> np.ndarray:
        '''Getting frames with fixed fps'''
        current_time = time.perf_counter()
        
        sleep_time = self.next_frame_time - current_time
        if sleep_time > 0:
            time.sleep(sleep_time)
        
        frame = self.get_screenshot()
        self.next_frame_time += self.target_time
        return frame