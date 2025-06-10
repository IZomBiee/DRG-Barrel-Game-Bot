import numpy as np
import cv2

from ..toml_setting_loader import TOMLSettingLoader as TSL
from .detector import Detector

class DifferenceBasketDetector(Detector):
    '''Detector that use difference between 2 frames to find basket'''
    def __init__(self):
        settings = TSL()['detectors']['difference']
        self.min_contour_area = settings['min_area']

        self.blur_kernel = settings['ksize']
        self.blur_sigma_x = settings['sigma_x']
        self.blur_threshold = settings['threshold']

        self.last_frame: np.ndarray | None = None

    def _process_image(self, image: np.ndarray, rewrite_last_image: bool = True) -> np.ndarray:
        if self.last_frame is None:
            if rewrite_last_image: 
                self.last_frame = image
            return np.zeros((image.shape[0], image.shape[1], 1), dtype=np.uint8)
        
        last_gray = cv2.cvtColor(self.last_frame, cv2.COLOR_BGR2GRAY)
        current_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        diff_image = cv2.bitwise_xor(current_gray, last_gray)
        
        blurred = cv2.GaussianBlur(diff_image, self.blur_kernel, self.blur_sigma_x)
        _, thresholded_image = cv2.threshold(blurred, self.blur_threshold, 255, cv2.THRESH_BINARY)
        
        if rewrite_last_image: 
            self.last_frame = image
        
        return thresholded_image

    def find(self, image: np.ndarray) -> list[int] | None:
        processed_image = self._process_image(image)

        contours, _ = cv2.findContours(processed_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours):
            biggest_contour = sorted(contours, key=cv2.contourArea)[-1]
            if cv2.contourArea(biggest_contour) > self.min_contour_area:
                x, y, w, h = cv2.boundingRect(biggest_contour)
                position = [round(x+(w//2)), round(y+(h//2))]
                return position
        return None