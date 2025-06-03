import cv2
import numpy as np
import math

class PointBasketDetector():
    '''Class for detecting basket on court position'''
    def __init__(self, point1: list[int], point2: list[int], margin: int,
                 basket_color: list[int], color_margin: int, basket_width: int):
        self.point1 = point1
        self.point2 = point2
        self.margin = margin
        self.distance = round(math.sqrt((self.point2[0] - self.point1[0]+basket_width)**2+(self.point2[1] - self.point1[1])**2))
        self.color = basket_color
        self.color_margin = color_margin 

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
        return self.is_point_correct_color(self.point1, image)

    def is_point2_correct_color(self, image: np.ndarray) -> bool:
        color_distance = self.is_point_correct_color(self.point2, image)
        return color_distance

    def find_x(self, image: np.ndarray) -> int | None:
        """Find the x position of the basket (to be implemented)"""
        # Implementation goes here
        raise NotImplementedError("find_x method not implemented yet")
    
    def draw_points(self, image: np.ndarray) -> np.ndarray:
        """Draw the detection points on the image with color coding"""
        point1_color = (0, 255, 0) if self.is_point1_correct_color(image) else (0, 0, 255)
        point2_color = (0, 255, 0) if self.is_point2_correct_color(image) else (0, 0, 255)
        
        image = cv2.circle(image, self.point1, self.margin, point1_color, 3)
        image = cv2.circle(image, self.point2, self.margin, point2_color, 3)
        return image
    
    def calculate_speed(self, dt:float) -> float:
        return self.distance/dt