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
        self.direction_detection_time = settings['direction_detection_time']

        self.avarage_velocity = [0., 0.]

        self.previous_moving_direction = 'Right'
        print("Initialized Predictor")

    def _update_moving_direction(self):
        if self.is_on_left_border():
            changed_direction = self.previous_moving_direction == 'Left' and \
            self.is_moving_right()
            if changed_direction:
                self.previous_moving_direction = 'Right'
                self.on_direction_change()
        elif self.is_on_right_border():
            changed_direction = self.previous_moving_direction == 'Right' and \
            not self.is_moving_right()
            if changed_direction:
                self.previous_moving_direction = 'Left'
                self.on_direction_change()

    def _update_avarage_velocity(self) -> None:
        if len(self.boxes) >= 2:
            dt = self.times[-1]-self.times[0]
            if dt != 0:
                dx = self.boxes[-1][0] - self.boxes[0][0]
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

    def get_last_center_position(self) -> tuple[float, float] | None:
        box = self.get_last_box()
        if box is not None:
            x0, y0, x1, y1 = box
            return ((x0+x1)/2, (y0+y1)/2)

    def is_moving_right(self) -> bool:
        if len(self.boxes) >= 2:
            start_time = self.times[-1]  # latest time
            index = len(self.times) - 2  # start from second last

            # go backwards in time while still inside the detection window
            while index >= 0 and start_time - self.times[index] < self.direction_detection_time:
                index -= 1

            # clamp index (make sure we don’t fall below 0)
            if index < 0:
                index = 0

            # compare x positions (assuming boxes are (x, y, w, h) or similar)
            delta_x = self.boxes[-1][0] - self.boxes[index][0]

            # if delta_x is positive, object moved right
            if delta_x > 0:
                return True
        return False

    def is_on_setup_position(self) -> bool:
        last_pos = self.get_last_center_position()
        if last_pos is None or self.get_setup_position()>last_pos[0]: 
            return True
        return False

    def update_borders(self) -> None:
        pos = self.get_last_center_position() 
        if pos is not None:
            x, y = pos

            if self.left_border_x > x:
                self.left_border_x = x
            if self.right_border_x < x:
                self.right_border_x = x

    def update(self, image:np.ndarray, time:float) -> None:
        detections = self.detector.detect(image)
        if len(detections) < 1:
            return
        
        detection = max(detections, key=lambda data: data['conf'])
        box = detection['normalized_box']

        self.boxes.append(box)
        self.times.append(time)
        self._update_moving_direction()
        self._update_avarage_velocity()

    def on_direction_change(self) -> None:
        if len(self.times) > 0:
            self.times = [self.times[-1]]
            self.boxes = [self.boxes[-1]]
        self.avarage_velocity = [0, 0]

    def get_setup_position(self) -> float:
        return self.left_border_x + (self.right_border_x - self.left_border_x) * self.setup_position

    def is_on_left_border(self) -> bool:
        pos = self.get_last_center_position()
        if pos is not None:
            if self.left_border_x+self.border_tollirance>pos[0]:
                return True
        return False
    
    def is_on_right_border(self) -> bool:
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
            f'Right: {self.time_to_right_border():.2f}',
            f'Cycle: {self.cycle_time():.2f}'
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