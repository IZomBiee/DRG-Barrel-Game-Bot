import cv2
import numpy as np

from .hsv_basket_detector import HSVBasketDetector

class BasketPredictor:
    '''Class for predicting basket next position and time to necessery position'''
    def __init__(self, detector: HSVBasketDetector,
                 time_of_positions: float, border_tolirance: int) -> None:
        self.detector = detector

        self.time_of_positions = time_of_positions
        self.x_velocities: list[float] = []
        self.times: list[float] = []
        self.avarage_velocity_x = 0 
        self.last_x_position = None

        self.border_tolirance = border_tolirance

    def _check_positions_count(self) -> None:
        if sum(self.times) > self.time_of_positions:
            self.times.pop(0)
            self.x_velocities.pop(0)

    def update(self, image:np.ndarray, dt:float) -> None:
        x = self.detector.find_x(image)
        if x is not None:
            if self.last_x_position is not None:
                self.x_velocities.append((x-self.last_x_position)/dt)
            self.last_x_position = x
            self.times.append(dt)
            self._update_avarage_velocity_x()
            self._check_positions_count()

    def _update_avarage_velocity_x(self) -> None:
        if len(self.x_velocities) > 0:
            self.avarage_velocity_x = sum(self.x_velocities)/len(self.x_velocities)

    def is_on_left_border(self) -> bool:
        if self.last_x_position is not None: 
            if self.last_x_position - self.border_tolirance < self.detector.get_left_border():
                return True
        return False
    
    def is_on_right_border(self) -> bool:
        if self.last_x_position is not None: 
            if self.last_x_position - self.border_tolirance < self.detector.get_right_border():
                return True
        return False

    def cycle_time(self) -> float:
        if self.last_x_position is not None:
            if self.avarage_velocity_x != 0:
                right_border = self.detector.get_right_border()
                left_border = self.detector.get_left_border()

                gap =  right_border - left_border
                if gap != 0:
                    return gap/self.avarage_velocity_x  
        return 0

    def time_to_right_border(self) -> float:
        if self.last_x_position is not None:
            if self.avarage_velocity_x != 0:
                right_border = self.detector.get_right_border()
                left_border = self.detector.get_left_border()
                if right_border is None or left_border is None:
                    return 0

                gap = right_border - self.last_x_position
                if gap != 0:
                    return gap/self.avarage_velocity_x
        return 0

    def _predict_next_position(self, x:int) -> int:
        left_border, right_border = self.detector.get_left_border(), self.detector.get_right_border()
        if self.avarage_velocity_x == 0: return x
        elif left_border is None or right_border is None:
            return round(x+self.avarage_velocity_x*0.1)
        
        span = right_border - left_border
        if span <= 0:
            return x
        
        current_relative = x - left_border
        new_relative = current_relative + self.avarage_velocity_x*0.1
        
        remainder = new_relative % (2 * span)
        if remainder > span:
            final_relative = 2 * span - remainder
        else:
            final_relative = remainder
        
        return round(left_border + final_relative)

    def draw_basket(self, image:np.ndarray) -> np.ndarray:
        if self.last_x_position is not None:
            image = cv2.line(image, (self.last_x_position, 0), (self.last_x_position, image.shape[0]), (255, 0, 255), 5)
        return image
    
    def draw_traectory(self, image: np.ndarray) -> np.ndarray:
        if self.last_x_position is not None:
            next_pos_x = int(self._predict_next_position(self.last_x_position))
            image = cv2.line(image, (next_pos_x, 0),
                            (next_pos_x, image.shape[0]), (16, 65, 64), 5)
        return image
