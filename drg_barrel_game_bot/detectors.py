import cv2
import numpy as np
import math

from abc import ABC, abstractmethod
from .settings_loader import TOMLSettingsLoader as TSL

class Detector(ABC):
    def find_x(self, image: np.ndarray) -> int | None: ...

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
    
    def find_x(self, image: np.ndarray) -> int | None:
        processed_image = self._proccess_image(image)

        contours, _ = cv2.findContours(processed_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours):
            biggest_contour = sorted(contours, key=cv2.contourArea)[-1]
            if cv2.contourArea(biggest_contour) > self.min_contour_area:
                x, y, w, h = cv2.boundingRect(biggest_contour)
                position_x = round(x+(w//2))
                return position_x
            
class PointBasketDetector(Detector):
    '''Detector that use not for basket position detecting, but for more
    precaise speed calculation'''
    def __init__(self):
        settings = TSL()['detectors']['point']

        self.point1 = settings['point1_position']
        self.point2 = settings['point2_position']
        self.margin = settings['position_margin']
        self.distance = self.point2[0] - self.point1[0] - settings['basket_width']
        self.color = settings['basket_color']
        self.color_margin = settings['color_margin']
        
        self.history = [1, 1]

    @staticmethod
    def get_image_point(image: np.ndarray, cord: list[int], margin: int) -> np.ndarray:
        """Extract a region around the given coordinate"""
        return image[cord[1]-margin:cord[1]+margin, cord[0]-margin:cord[0]+margin]

    @staticmethod 
    def get_image_avg_color(image: np.ndarray) -> list[int]:
        """Calculate the average color of the image region"""
        average = image.mean(axis=0).mean(axis=0)
        if any(np.isnan(average)): 
            return [0, 0, 0]
        average = list(map(int, average))
        return average

    def is_point_correct_color(self, point: list[int], image: np.ndarray) -> bool:
        """Check if the color at the given point matches the expected color"""
        avg_color = self.get_image_avg_color(
            self.get_image_point(image, point, self.margin))
        color_distance = sum((c1 - c2) ** 2 for c1, c2 in zip(avg_color, self.color)) ** 0.5
        return color_distance <= self.color_margin

    def is_point1_correct_color(self, image: np.ndarray) -> bool:
        if self.is_point_correct_color(self.point1, image):
            self.history.append(1)
            return True
        return False

    def is_point2_correct_color(self, image: np.ndarray) -> bool:
        if self.is_point_correct_color(self.point2, image):
            self.history.append(2)
            return True
        return False
    
    def draw_points(self, image: np.ndarray) -> np.ndarray:
        """Draw the detection points on the image with color coding"""
        point1_color = (0, 255, 0) if self.is_point1_correct_color(image) else (0, 0, 255)
        point2_color = (0, 255, 0) if self.is_point2_correct_color(image) else (0, 0, 255)
        
        image = cv2.circle(image, self.point1, self.margin, point1_color, 3)
        image = cv2.circle(image, self.point2, self.margin, point2_color, 3)
        return image
    
    def find_x(self, image: np.ndarray) -> int | None:
        if self.is_point1_correct_color(image):
            return self.point1[0]
        elif self.is_point2_correct_color(image) and self.history[-2] == 1:
            return self.point2[0]
        return None