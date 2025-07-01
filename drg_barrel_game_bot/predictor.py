import cv2
import numpy as np
import time
from .utils import *

from .detector import Detector
from .setting_loader import SettingLoader as SL

class Predictor:
    '''Class for predicting basket next position and time to necessery position'''
    def __init__(self, detector: Detector) -> None:
        self.detector = detector
        settings = SL()['basket_predictor']

        self.times = []
        self.boxes = []
        
        self.left_border_x = 1
        self.right_border_x = 0
        self.border_tollirance = settings['border_tolirance']
        self.setup_position = settings['velocity_setup_position']

        self.avarage_velocity = [0., 0.]

        self.left_border = False
        self.right_border = False

    def _update_border_state(self):
        if self.left_border:
            if not self.on_left_border(): 
                self.left_border = False
                self.clear()
        elif self.on_left_border():
            self.left_border = True

        if self.right_border:
            if not self.on_right_border(): 
                self.right_border = False
                self.clear()
        elif self.on_right_border():
            self.right_border = True

    def _update_avarage_velocity(self) -> None:
        if len(self.boxes) >= 2:
            dt = self.times[-1]-self.times[0]
            if dt != 0:
                dx = ((self.boxes[-1][0]+self.boxes[-1][2])/2) - (self.boxes[0][0]+self.boxes[0][2])/2 
                dy = self.boxes[-1][1] - self.boxes[0][1]
                self.avarage_velocity = [dx / dt, dy / dt]
            else:
                self.avarage_velocity = [0, 0]
        else:
            self.avarage_velocity = [0, 0]

    def get_last_box(self) -> list[float] | None:
        if len(self.boxes) > 0:
            return self.boxes[-1]
        return None

    def get_last_center_position(self) -> list[float] | None:
        box = self.get_last_box()
        if box is not None:
            x0, y0, x1, y1 = box
            return [(x0+x1)/2, (y0+y1)/2]

    def is_on_setup_position(self) -> bool:
        last_pos = self.get_last_center_position()
        if last_pos is None or self.get_setup_position()>last_pos[0]: 
            return True
        return False

    def update_borders(self, image:np.ndarray) -> None:
        box = self.detector.find(image)
        if box is not None:
            x0, y0, x1, y1 = box

            x = (x0+x1)/2
            if self.left_border_x > x:
                self.left_border_x = x
            if self.right_border_x < x:
                self.right_border_x = x

    def update(self, image:np.ndarray, time:float) -> None:
        box = self.detector.find(image)
        if box is None:
            return

        self.boxes.append(box)
        self.times.append(time)
        self._update_avarage_velocity()
        self._update_border_state()

    def clear(self) -> None:
        self.times = []
        self.boxes = []
        self.avarage_velocity = [0, 0]

    def get_setup_position(self) -> float:
        return self.left_border_x + (self.right_border_x - self.left_border_x) * self.setup_position

    def on_left_border(self) -> bool:
        pos = self.get_last_center_position()
        if pos is not None:
            if self.left_border_x+self.border_tollirance>pos[0]:
                return True
        return False
    
    def on_right_border(self) -> bool:
        pos = self.get_last_center_position()
        if pos is not None:
            if self.right_border_x-self.border_tollirance<pos[0]:
                return True
        return False

    def time_to_right_border(self) -> float:
        if self.avarage_velocity[0] > 0:
            gap = self.right_border_x-self.left_border_x
            left_px = gap-(self.boxes[-1][0]-self.left_border_x)
            return left_px/self.avarage_velocity[0]
        return -1
            
    def cycle_time(self) -> float:
        if self.avarage_velocity[0] > 0:
            gap = self.right_border_x-self.left_border_x
            return round((gap/self.avarage_velocity[0])*2, 3)
        return -1

    def draw(self, image: np.ndarray) -> np.ndarray:
        last_position = self.get_last_center_position()
        last_box = self.get_last_box()
        if last_position is None or last_box is None:
            return image
        
        image = self.detector.draw(image)
        height, width = image.shape[:2]
        
        normalized_x, normalized_y = last_position
        normalized_velocity_x, normalized_velocity_y = self.avarage_velocity
        Draw.vector_normalized(image, normalized_x, normalized_y, normalized_velocity_x, normalized_velocity_y, (0, 255, 0), 1)


        Draw.vertical_line(image, width*self.right_border_x, (0, 0, 255))
        Draw.vertical_line(image, width*self.left_border_x, (255, 0, 0))
        Draw.vertical_line(image, width*self.get_setup_position(), (0, 255, 255))
        
        self.detector.draw(image)        

        Draw.texts(image, width*last_position[0], height*last_box[3]*1.1, [
            f'Right: {round(self.time_to_right_border(), 2)}',
            f'Cycle: {round(self.cycle_time(), 2)}'
        ], (255, 255, 255))

        return image
    
    
    def draw_trail(self, image:np.ndarray, lenght:float=1.0,
                   from_center:bool=True) -> np.ndarray:
        previous_position = None
        height, width = image.shape[:2]
        thickness = max(1, width//600)

        for index, box in enumerate(self.boxes[::-1]):
            if self.times[index] - self.times[0] > lenght:
                break
            
            center_y = int((box[1]+box[3])/2*height)
            if from_center:
                center_x = int((box[0]+box[2])/2*width)
            else:
                center_x = int(box[0]*width)
            if previous_position is not None:
                image = cv2.line(image, (previous_position[0], previous_position[1]),
                         (center_x, center_y), (255, 0, 0), thickness)
            image = cv2.circle(image, (center_x, center_y), thickness, (0, 120, 255), thickness)
            previous_position = [center_x, center_y]

        return image