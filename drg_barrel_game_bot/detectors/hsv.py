import numpy as np
import cv2

from ..toml_setting_loader import TOMLSettingLoader as TSL
from .detector import Detector

class HSVBasketDetector(Detector):
    '''Detector that use hsv thresholding and contours to find basket'''
    def __init__(self):
        settings = TSL()['detectors']['hsv']
        self.min_contour_area = settings['min_area']

        self.hsv_min = np.array(settings['hsv_min'])
        self.hsv_max = np.array(settings['hsv_max'])

        self.blur_kernel = settings['ksize']
        self.blur_sigma_x = settings['sigma_x']
        self.blur_threshold = settings['threshold']

    def _proccess_image(self, image: np.ndarray) -> np.ndarray:
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        thresholded_image = cv2.inRange(hsv_image, self.hsv_min, self.hsv_max)
        thresholded_image = cv2.GaussianBlur(thresholded_image, self.blur_kernel, self.blur_sigma_x)
        thresholded_image = cv2.threshold(thresholded_image, self.blur_threshold, 255, cv2.THRESH_BINARY)[1]
        return thresholded_image
    
    def find(self, image: np.ndarray) -> list[int] | None:
        processed_image = self._proccess_image(image)

        contours, _ = cv2.findContours(processed_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours):
            biggest_contour = sorted(contours, key=cv2.contourArea)[-1]
            if cv2.contourArea(biggest_contour) > self.min_contour_area:
                x, y, w, h = cv2.boundingRect(biggest_contour)
                position = [round(x+(w//2)), round(y+(h//2))]
                return position
        return None
