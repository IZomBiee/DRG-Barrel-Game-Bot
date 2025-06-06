import cv2
import numpy as np

from .detectors import Detector
from .settings_loader import TOMLSettingsLoader as TSL

class BorderManager:
    def __init__(self, detector: Detector) -> None:
        self.detector = detector

        self.left_border_x:int = 100000
        self.right_border_x:int = 0
        self.border_tolirance = TSL()['basket']['border_tolirance']

    def update_borders(self, image: np.ndarray) -> None:
        x = self.detector.find_x(image)
        if x is None: return
        if self.left_border_x is not None and self.left_border_x > x:
            self.left_border_x = x
        if self.right_border_x is not None and self.right_border_x < x:
            self.right_border_x = x

    def draw_borders(self, image: np.ndarray) -> np.ndarray:
        image = cv2.line(image, (self.left_border_x, 0), (self.left_border_x, image.shape[0]), (0, 255, 0), 5)
        image = cv2.line(image, (self.right_border_x, 0), (self.right_border_x, image.shape[0]), (0, 255, 0), 5)
        return image

    def get_left_border(self) -> int:
        return self.left_border_x

    def get_right_border(self) -> int:
        return self.right_border_x
    
    def is_on_left_border(self, image: np.ndarray) -> bool:
        x = self.detector.find_x(image)
        if x is not None: 
            if x - self.border_tolirance < self.get_left_border():
                return True
        return False
    
    def is_on_right_border(self, image: np.ndarray) -> bool:
        x = self.detector.find_x(image)
        if x is not None: 
            if x + self.border_tolirance > self.get_right_border():
                return True
        return False
