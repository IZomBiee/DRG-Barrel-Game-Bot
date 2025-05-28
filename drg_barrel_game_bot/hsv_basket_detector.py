import cv2
import numpy as np

from .settings_loader import TOMLSettingsLoader as TSL

class HSVBasketDetector():
    def __init__(self):
        self.left_border_x = None
        self.right_border_x = None

        self.basket_axis_y_gap: list[float] = TSL()['basket']['axis_line_gap'] # type: ignore

        self.min_contour_area = TSL()['basket']['min_area'] # type: ignore

        self.hsv_min = np.array(TSL()['basket']['hsv_min']) # type: ignore
        self.hsv_max = np.array(TSL()['basket']['hsv_max']) # type: ignore

    def _update_borders(self, x:int) -> None:
        if self.left_border_x is None or self.left_border_x > x:
            self.left_border_x = x
        if self.right_border_x is None or self.right_border_x < x:
            self.right_border_x = x

    def _proccess_image(self, image: np.ndarray) -> np.ndarray:
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        thresholded_image = cv2.inRange(hsv_image, self.hsv_min, self.hsv_max)
        thresholded_image = cv2.GaussianBlur(thresholded_image, [9, 9], 15)
        thresholded_image = cv2.threshold(thresholded_image, 1, 255, cv2.THRESH_BINARY)[1]
        return thresholded_image
    
    def find_x(self, image: np.ndarray) -> int | None:
        processed_image = self._proccess_image(image)

        contours, _ = cv2.findContours(processed_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours):
            biggest_contour = sorted(contours, key=cv2.contourArea)[-1]
            if cv2.contourArea(biggest_contour) < self.min_contour_area:
                return None
            x, y, w, h = cv2.boundingRect(biggest_contour)
            position_x = round(x+(w//2))
            self._update_borders(position_x)
            return position_x
        return None
    
    def draw_borders(self, image: np.ndarray) -> np.ndarray:
        if self.left_border_x is not None:
            image = cv2.line(image, (self.left_border_x, 0), (self.left_border_x, image.shape[0]), (0, 255, 0), 5)
        if self.right_border_x is not None:
            image = cv2.line(image, (self.right_border_x, 0), (self.right_border_x, image.shape[0]), (0, 255, 0), 5)
        return image

    def get_left_border(self) -> int | None:
        return self.left_border_x

    def get_right_border(self) -> int | None:
        return self.right_border_x



    
    # def get_seconds_to_position(self, speed_per_s, current_position, left_border, right_border, target_position, tolirance):
    #     left_for_target_position = 1
    #     while True:
    #         predicted_position = self._predict_bouincing((speed_per_s*left_for_target_position), current_position, left_border, right_border)
    #         if predicted_position < target_position-tolirance:
    #             left_for_target_position -= 0.1
    #         elif predicted_position > target_position+tolirance:
    #             left_for_target_position += 0.1
    #         else:
    #             return abs(left_for_target_position)