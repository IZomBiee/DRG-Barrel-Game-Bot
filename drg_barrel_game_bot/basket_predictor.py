import cv2
import numpy as np

from .hsv_basket_detector import HSVBasketDetector
from .settings_loader import TOMLSettingsLoader as TSL

class BasketPredictor:
    def __init__(self) -> None:
        self.detector = HSVBasketDetector()

        self.position_count = TSL()['basket']['position_count'] # type: ignore
        self.positions = []
        self.times = []
        self.velocity_x = None 

        self.left_border_tolirance = TSL()['basket']['left_border_tolirance'] # type: ignore
    
    def _check_positions_count(self) -> None:
        if len(self.positions) > self.position_count:
            self.positions.pop(0)

    def update(self, image:np.ndarray, dt:float) -> None:
        x = self.detector.find_x(image)
        if x is None: return
        self.positions.append(x)
        self.times.append(dt)
        self._update_velocity_x()
        self._check_positions_count()

    def _update_velocity_x(self) -> None:
        if len(self.positions) < 2:
            return None
            
        difference_sum = 0
        for i in range(1, len(self.positions)):
            difference_sum += self.positions[i] - self.positions[i-1] 

        self.velocity_x = difference_sum / (len(self.positions) - 1)

    def on_left_border(self) -> bool:
        if len(self.positions) < 1: return False
        
        if self.positions[-1] - self.left_border_tolirance < self.detector.get_left_border():
            return True
        return False
    
    def _frame_to_cycle(self) -> float | None:
        if len(self.positions) < 0 or self.velocity_x is None: return None

        right_border = self.detector.get_right_border()
        left_border = self.detector.get_left_border()
        if right_border is None or left_border is None: return None

        gap = right_border - left_border
        return gap/self.velocity_x

    def frames_to_center(self) -> float | None:
        cycle_frame_count = self._frame_to_cycle()
        if cycle_frame_count is None: return None
        return cycle_frame_count*1.5

    def _predict_next_position(self, x:int) -> int:
        left_border, right_border = self.detector.get_left_border(), self.detector.get_right_border()
        if self.velocity_x is None: return x
        elif left_border is None or right_border is None:
            return round(x+self.velocity_x)
        
            
        span = right_border - left_border
        if span <= 0:
            return x
        
        current_relative = x - left_border
        new_relative = current_relative + self.velocity_x
        
        remainder = new_relative % (2 * span)
        if remainder > span:
            final_relative = 2 * span - remainder
        else:
            final_relative = remainder
        
        return round(left_border + final_relative)

    def draw_basket(self, image:np.ndarray) -> np.ndarray:
        if len(self.positions) > 0:
            x = self.positions[-1]
            image = cv2.line(image, (x, 0), (x, image.shape[0]), (255, 0, 255), 5)
        return image
    
    def draw_traectory(self, image: np.ndarray) -> np.ndarray:
        if self.velocity_x is not None:
            next_pos_x = int(self._predict_next_position(self.positions[-1]))
            image = cv2.line(image, (next_pos_x, 0),
                            (next_pos_x, image.shape[0]), (16,  65, 64), 5)
        return image
