import cv2
import numpy as np
import time
import keyboard

from .toml_setting_loader import TOMLSettingLoader as TSL

class KickManager:
    '''Class for meneging when do kick and can we do kick'''
    def __init__(self) -> None:
        settings = TSL()['kick_manager']
        self.e_button_image = cv2.imread(settings['template_path'])
        self.e_button_image = cv2.cvtColor(self.e_button_image, cv2.COLOR_BGR2GRAY)
        
        self.e_button_gap = settings['template_detection_sensitivity']
        
        self.last_detected_time = 0
        self.detected_barrel_in_front = True
        self.barrel_bouncing_time = settings['barrel_bouncing_time']

    def update(self, image:np.ndarray) -> None:
        self.is_barrel_in_front(image)

    def is_barrel_in_front(self, image: np.ndarray) -> bool:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        location_image = cv2.matchTemplate(image, self.e_button_image, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_pos, max_pos = cv2.minMaxLoc(location_image)
        if max_val > self.e_button_gap:
            if not self.detected_barrel_in_front:
                self.detected_barrel_in_front = True
                self.last_detected_time = time.perf_counter()
            return True
        else:
            self.detected_barrel_in_front = False
        return False
    
    def is_barrel_debounce_time_passed(self) -> bool:
        if self.detected_barrel_in_front:
            if time.perf_counter()-self.last_detected_time>self.barrel_bouncing_time:
                return True
        return False
    

    def can_kick(self, image: np.ndarray) -> bool:
        if not self.detected_barrel_in_front:
            print("Barrel not in front!")
            return False
        if not self.is_barrel_debounce_time_passed():
            print("Barrel bouncing!")
            return False
        return True

    
    def kick(self):
        keyboard.press_and_release("e")
        self.detected_barrel_in_front = False
        self.last_kick_time = time.perf_counter()

    def draw_state(self, image: np.ndarray) -> np.ndarray:
        if not self.detected_barrel_in_front:
            text, color = "No barrel", (0,0,255)
        elif not self.is_barrel_debounce_time_passed():
            text, color = f"Bouncing: {self.barrel_bouncing_time - (time.perf_counter()-self.last_detected_time):.2f}s", (0,0,255)
        else:
            text, color = "READY TO KICK", (0,255,0)
        cv2.putText(image, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        return image