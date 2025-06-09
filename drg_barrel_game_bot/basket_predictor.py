import cv2
import numpy as np

from .detectors import Detector
from .settings_loader import TOMLSettingsLoader as TSL

class BasketPredictor:
    '''Class for predicting basket next position and time to necessery position'''
    def __init__(self, detector: Detector) -> None:
        self.detector = detector
        settings = TSL()['basket_predictor']

        self.time_of_positions = settings['time_of_positions']
        self.times = []
        self.positions = []
        
        self.left_border_x = 1000
        self.right_border_x = 0
        self.border_tollirance = settings['border_tolirance']

        self.avarage_velocity = [0., 0.]

    def _check_positions_count(self) -> None:
        if sum(self.times) > self.time_of_positions:
            self.times.pop(0)
            self.positions.pop(0)

    def update(self, image:np.ndarray, dt:float) -> None:
        pos = self.detector.find(image)
        if pos is None: return
        self.positions.append(pos)
        self.times.append(dt)
        self._update_avarage_velocity()
        self._check_positions_count()

    def update_borders(self) -> None:
        if len(self.positions) > 0:
            x, y = self.positions[-1]
            if self.left_border_x > x:
                self.left_border_x = x
            if self.right_border_x < x:
                self.right_border_x = x
    
    def on_left_border(self) -> bool:
        if len(self.positions) > 0:
            if self.left_border_x+self.border_tollirance>self.positions[-1][0]:
                return True
        return False
    
    def on_right_border(self) -> bool:
        if len(self.positions) > 0:
            if self.right_border_x-self.border_tollirance<self.positions[-1][0]:
                return True
        return False

    def time_to_left_border(self) -> float:
        if self.avarage_velocity[0] > 0:
            gap = self.right_border_x-self.left_border_x
            left_px = gap-(self.positions[-1][0]-self.left_border_x)
            return left_px/self.avarage_velocity[0]
        return -1
    
    def _update_avarage_velocity(self) -> None:
        if len(self.positions) >= 2:
            dt = sum(self.times)
            if dt != 0:
                dx = self.positions[-1][0] - self.positions[0][0]
                dy = self.positions[-1][1] - self.positions[0][1]
                self.avarage_velocity = [dx / dt, dy / dt]
            else:
                self.avarage_velocity = [0, 0]
        else:
            self.avarage_velocity = [0, 0]
            
    def cycle_time(self) -> float:
        if self.avarage_velocity[0] > 0:
            gap = self.right_border_x-self.left_border_x
            return gap/self.avarage_velocity[0]
        return -1

    def time_to_right_border(self) -> float:
        raise NotImplementedError()

    def _predict_next_position(self, x: int, y: int) -> tuple[int, int]:
        left_border, right_border = self.left_border_x, self.right_border_x
        if self.avarage_velocity[0] == 0 and self.avarage_velocity[1] == 0:
            return x, y
        if left_border is None or right_border is None:
            return round(x + self.avarage_velocity[0] * 0.1), round(y + self.avarage_velocity[1] * 0.1)

        span = right_border - left_border
        if span <= 0:
            return x, y

        def bounce(pos, velocity, left, right):
            span = right - left
            current_relative = pos - left
            new_relative = current_relative + velocity * 0.1
            remainder = new_relative % (2 * span)
            if remainder > span:
                final_relative = 2 * span - remainder
            else:
                final_relative = remainder
            return round(left + final_relative)

        new_x = bounce(x, self.avarage_velocity[0], left_border, right_border)
        new_y = round(y + self.avarage_velocity[1] * 0.1)

        return new_x, new_y

    def draw_basket(self, image:np.ndarray) -> np.ndarray:
        if len(self.positions) > 0:
            x, y = self.positions[-1]
            image = cv2.line(image, (x, 0), (x, image.shape[0]), (255, 0, 255), 5)
        return image
    
    def draw_trajectory(self, image: np.ndarray) -> np.ndarray:
        if len(self.positions) > 0:
            x, y = self.positions[-1]
            x, y = self._predict_next_position(x, y)
            x, y = round(x), round(y)
            image = cv2.line(image, (x, 0),
                            (x, image.shape[0]), (16, 65, 64), 5)
        return image

    def draw_borders(self, image: np.ndarray) -> np.ndarray:
        if self.left_border_x is not None:
            cv2.line(image, (self.left_border_x, 0), (self.left_border_x, image.shape[0]), (0, 255, 0), 2)
        if self.right_border_x is not None:
            cv2.line(image, (self.right_border_x, 0), (self.right_border_x, image.shape[0]), (0, 0, 255), 2)
        return image
