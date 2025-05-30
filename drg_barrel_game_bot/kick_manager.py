import cv2
import numpy as np
import time
import keyboard

class KickManager:
    '''Class for meneging when do kick and can we do kick'''
    def __init__(self, e_button_image_path: str, resolution:list[int], e_button_detection_gap: float,
                 barrel_boucing_time: float, minimal_kick_delay: float) -> None:
        self.e_button_image = cv2.imread(e_button_image_path)
        self.e_button_image = cv2.cvtColor(self.e_button_image, cv2.COLOR_BGR2GRAY)
        button_new_resolution = [round(self.e_button_image.shape[1]*(resolution[0]/3840)),
                                 round(self.e_button_image.shape[0]*(resolution[1]/2160))]
        self.e_button_image = cv2.resize(self.e_button_image, button_new_resolution)
        self.e_button_gap = e_button_detection_gap

        self.last_detected_barrel_in_front_time = 0
        self.detected_barrel_in_front = True
        self.barrel_boucing_time = barrel_boucing_time

        self.last_kick_time = 0
        self.kick_delay = minimal_kick_delay

    def is_barrel_in_front(self, image: np.ndarray) -> bool:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        location_image = cv2.matchTemplate(image, self.e_button_image, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_pos, max_pos = cv2.minMaxLoc(location_image)
        if max_val > self.e_button_gap:
            if not self.detected_barrel_in_front:
                self.detected_barrel_in_front = True
                self.last_detected_barrel_in_front_time = time.perf_counter()
            return True
        else:
            self.detected_barrel_in_front = False
        return False
    
    def is_barrel_debounce_time_passed(self) -> bool:
        if self.detected_barrel_in_front:
            if time.perf_counter()-self.last_detected_barrel_in_front_time>self.barrel_boucing_time:
                return True
        return False
    
    def is_kick_delay_passed(self) -> bool:
        if time.perf_counter()-self.last_kick_time > self.kick_delay:
            return True
        return False

    def can_kick(self, image: np.ndarray) -> bool:
        if self.is_barrel_in_front(image):
            if self.is_barrel_debounce_time_passed():
                if self.is_kick_delay_passed():
                    return True
                else:
                    print("Kick delay is not passed")
            else:
                print("Debounce time is not passed!")
        else:
            print("Barrel is not in front!")
        return False
    def kick(self):
        keyboard.press_and_release("e")
        self.detected_barrel_in_front = False
        self.last_kick_time = time.perf_counter()