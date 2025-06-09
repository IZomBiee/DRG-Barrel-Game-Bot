import numpy as np
import cv2

from drg_barrel_game_bot import TOMLSettingsLoader as TSL
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

        self.time_of_positions = settings['time_of_positions']
        self.positions = []
        self.delta_times = []

        self.avarage_velocity = [0., 0.]

    def _proccess_image(self, image: np.ndarray) -> np.ndarray:
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        thresholded_image = cv2.inRange(hsv_image, self.hsv_min, self.hsv_max)
        thresholded_image = cv2.GaussianBlur(thresholded_image, self.blur_kernel, self.blur_sigma_x)
        thresholded_image = cv2.threshold(thresholded_image, self.blur_threshold, 255, cv2.THRESH_BINARY)[1]
        return thresholded_image
    
    def _add_position(self, position:list[int], dt: float) -> None:
        if sum(self.delta_times) > self.time_of_positions:
            self.positions.pop(0)
            self.delta_times.pop(0)
        self.positions.append(position)
        self.delta_times.append(dt)

    def _update_avarage_velocity(self) -> None:
        if len(self.positions) > 1:
            delta_positions_x = []
            delta_positions_y = []
            for i in range(len(self.positions)-1):
                delta_positions_x.append((self.positions[i+1][0]-self.positions[i][0])/self.delta_times[i])
                delta_positions_x.append((self.positions[i+1][1]-self.positions[i][1])/self.delta_times[i])
            delta_positions_x_sum = sum(delta_positions_x)
            if delta_positions_x_sum != 0:
                self.avarage_velocity[0] = delta_positions_x_sum/len(delta_positions_x)

            delta_positions_y_sum = sum(delta_positions_y)
            if delta_positions_y_sum != 0:
                self.avarage_velocity[1] = sum(delta_positions_y)/len(delta_positions_y)

    def get_last_position(self) -> list[int] | None:
        if len(self.positions) > 0:
            return self.positions[-1]

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
    
    def update(self, image: np.ndarray, dt: float) -> None:
        position = self.find(image)
        if position is not None:
            self._add_position(position, dt)
            self._update_avarage_velocity()
        
    
    def draw_basket_position(self, image:np.ndarray) -> np.ndarray:
        pos = self.get_last_position()
        if pos is not None:
            image = cv2.circle(image, pos, 15, (0, 255, 255), 3)
        return image

    def draw_trajectory(self, image: np.ndarray) -> np.ndarray:
        pos = self.get_last_position()
        if pos is not None:
            next_pos_x = round(self.avarage_velocity[0]+pos[0])
            next_pos_y = round(self.avarage_velocity[1]+pos[1])
            next_pos = [next_pos_x, next_pos_y]
            image = cv2.circle(image, next_pos, 15, (255, 0, 255), 3)
        return image
