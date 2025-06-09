import numpy as np
import cv2

from drg_barrel_game_bot import TOMLSettingsLoader as TSL
from .detector import Detector

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
    
    # def find_x(self, image: np.ndarray) -> int | None:
    #     if self.is_point1_correct_color(image):
    #         return self.point1[0]
    #     elif self.is_point2_correct_color(image) and self.history[-2] == 1:
    #         return self.point2[0]
    #     return None