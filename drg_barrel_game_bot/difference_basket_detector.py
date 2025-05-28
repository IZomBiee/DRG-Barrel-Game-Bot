import cv2
import numpy as np

class DifferenceBasketDetector():
    def __init__(self):
        self.left_border_x = None
        self.right_border_x = None

        self.basket_axis_y_gap = (0.45, 0.55)
        self.last_grayscale_image = None

    def _update_borders(self, x:int) -> None:
        if self.left_border_x is None or self.left_border_x > x:
            self.left_border_x = x
        if self.right_border_x is None or self.right_border_x < x:
            self.right_border_x = x

    def preview(self, image: np.ndarray) -> np.ndarray:
        grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if self.last_grayscale_image is None:
            return np.zeros((image.shape[0], image.shape[1], 1), np.uint8)
        image_copy = self.last_grayscale_image.copy()
        grayscale_image = self._proccess_image(image)
        self.last_grayscale_image = image_copy.copy()
        return grayscale_image

    def _proccess_image(self, image: np.ndarray) -> np.ndarray:
        image = image[round(image.shape[0]*self.basket_axis_y_gap[0]):round(image.shape[0]*self.basket_axis_y_gap[1])]
        grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        

        if self.last_grayscale_image is not None:
            difference_image = cv2.bitwise_xor(grayscale_image, self.last_grayscale_image)
            blurred_difference_image = cv2.GaussianBlur(difference_image, [5, 5], 5)
            _, thresholded_difference_image = cv2.threshold(blurred_difference_image, 100, 255, cv2.THRESH_BINARY)
            self.last_grayscale_image = grayscale_image
            return thresholded_difference_image
        self.last_grayscale_image = grayscale_image
        return np.zeros((image.shape[0], image.shape[1], 1), np.uint8)
    
    def find_x(self, image: np.ndarray) -> int | None:
        image = self._proccess_image(image)
        contours, _ = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours):
            biggest_contour = sorted(contours, key=lambda contour: cv2.minEnclosingCircle(contour)[1])[-1]
            (x, y), r = cv2.minEnclosingCircle(biggest_contour)
            position_x = round(x)
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