import cv2
import numpy as np
import time

from .detector import Detector
from .setting_loader import SettingLoader as SL

class Predictor:
    '''Class for predicting basket next position and time to necessery position'''
    def __init__(self, detector: Detector) -> None:
        self.detector = detector
        settings = SL()['basket_predictor']

        self.times = []
        self.positions = []
        
        self.left_border_x = 1
        self.right_border_x = 0
        self.border_tollirance = settings['border_tolirance']
        self.setup_position = settings['velocity_setup_position']

        self.avarage_velocity = [0, 0]

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
        if len(self.positions) >= 2:
            dt = self.times[-1]-self.times[0]
            if dt != 0:
                dx = self.positions[-1][0] - self.positions[0][0]
                dy = self.positions[-1][1] - self.positions[0][1]
                self.avarage_velocity = [dx / dt, dy / dt]
            else:
                self.avarage_velocity = [0, 0]
        else:
            self.avarage_velocity = [0, 0]

    def is_on_setup_position(self) -> bool:
        if len(self.positions) > 0:
            gap = self.right_border_x - self.left_border_x
            realative_position = self.positions[-1][0]-self.left_border_x
            if gap*self.setup_position>realative_position: 
                return True
            else:
                return False
        return True

    def update_borders(self, image:np.ndarray) -> None:
        pos = self.detector.find(image)
        if pos is not None:
            x, y = pos
            if self.left_border_x > x:
                self.left_border_x = x
            if self.right_border_x < x:
                self.right_border_x = x

    def update(self, image:np.ndarray, time:float) -> None:
        pos = self.detector.find(image)
        if pos is None:
            if len(self.times) > 0:
                self.times[-1] += time
            return
        
        self.positions.append(pos)
        self.times.append(time)
        self._update_avarage_velocity()
        self._update_border_state()

    def clear(self) -> None:
        self.times = []
        self.positions = []
        self.avarage_velocity = [0, 0]

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

    def time_to_right_border(self) -> float:
        if self.avarage_velocity[0] > 0:
            gap = self.right_border_x-self.left_border_x
            left_px = gap-(self.positions[-1][0]-self.left_border_x)
            return left_px/self.avarage_velocity[0]
        return -1
            
    def cycle_time(self) -> float:
        if self.avarage_velocity[0] > 0:
            gap = self.right_border_x-self.left_border_x
            return round((gap/self.avarage_velocity[0])*2, 3)
        return -1

    def draw(self, image: np.ndarray) -> np.ndarray:
        image = self.detector.draw(image)

        if len(self.positions) == 0:
            return image

        h, w = image.shape[:2]
        norm_x, norm_y = self.positions[-1]
        abs_x = int(norm_x * w)
        abs_y = int(norm_y * h)

        if self.left_border_x is not None:
            cv2.line(image,
                    (int(w * self.left_border_x), 0),
                    (int(w * self.left_border_x), h),
                    (0, 255, 0), 2)

        if self.right_border_x is not None:
            cv2.line(image,
                    (int(w * self.right_border_x), 0),
                    (int(w * self.right_border_x), h),
                    (0, 0, 255), 2)

        cv2.circle(image, (abs_x, abs_y), 6, (255, 0, 255), -1)

        vx, vy = self.avarage_velocity
        vec_scale = 500
        end_x = int(abs_x + vx * vec_scale)
        end_y = int(abs_y + vy * vec_scale)

        cv2.arrowedLine(image, (abs_x, abs_y), (end_x, end_y), (0, 255, 255), 2, tipLength=0.2)

        pred_x = int((norm_x + vx) * w)
        pred_y = int((norm_y + vy) * h)
        cv2.circle(image, (pred_x, pred_y), 5, (0, 165, 255), -1)

        spacing = 25
        y_offset = abs_y - spacing * 3 - 10
        if y_offset < 10:
            y_offset = abs_y + 30

        t_right = self.time_to_right_border()
        cycle = self.cycle_time()

        info_lines = [
            f'Velocity: ({vx:.3f}, {vy:.3f})',
            f'Next pos: ({norm_x + vx:.2f}, {norm_y + vy:.2f})'
        ]

        if t_right >= 0:
            info_lines.append(f'Time to Right: {t_right:.2f}s')
        if cycle >= 0:
            info_lines.append(f'Cycle Time: {cycle:.2f}s')

        for i, line in enumerate(info_lines):
            cv2.putText(image, line, (abs_x, y_offset + i * spacing),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        return image

