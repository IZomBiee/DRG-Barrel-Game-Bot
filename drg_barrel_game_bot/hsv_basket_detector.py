import cv2
import numpy as np

class HSVBasketDetector():
    '''Class for detecting basket using hsv thesholding and basket border detection'''
    def __init__(self, minimal_contour_area:float, hsv_min:list[int], hsv_max:list[int],
                 blur_kernel:list[int], blur_sigma_x:int, blur_threshold:int):
        self.left_border_x:int = 100000
        self.right_border_x:int = 0

        self.min_contour_area = minimal_contour_area

        self.hsv_min = np.array(hsv_min)
        self.hsv_max = np.array(hsv_max)

        self.blur_kernel = blur_kernel
        self.blur_sigma_x = blur_sigma_x
        self.blur_threshold = blur_threshold

    def update_borders(self, x:int) -> None:
        if self.left_border_x is None or self.left_border_x > x:
            self.left_border_x = x
        if self.right_border_x is None or self.right_border_x < x:
            self.right_border_x = x

    def _proccess_image(self, image: np.ndarray) -> np.ndarray:
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        thresholded_image = cv2.inRange(hsv_image, self.hsv_min, self.hsv_max)
        thresholded_image = cv2.GaussianBlur(thresholded_image, self.blur_kernel, self.blur_sigma_x)
        thresholded_image = cv2.threshold(thresholded_image, self.blur_threshold, 255, cv2.THRESH_BINARY)[1]
        return thresholded_image
    
    def find_x(self, image: np.ndarray) -> int | None:
        processed_image = self._proccess_image(image)

        contours, _ = cv2.findContours(processed_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours):
            biggest_contour = sorted(contours, key=cv2.contourArea)[-1]
            if cv2.contourArea(biggest_contour) > self.min_contour_area:
                x, y, w, h = cv2.boundingRect(biggest_contour)
                position_x = round(x+(w//2))
                return position_x
    
    def draw_borders(self, image: np.ndarray) -> np.ndarray:
        image = cv2.line(image, (self.left_border_x, 0), (self.left_border_x, image.shape[0]), (0, 255, 0), 5)
        image = cv2.line(image, (self.right_border_x, 0), (self.right_border_x, image.shape[0]), (0, 255, 0), 5)
        return image

    def get_left_border(self) -> int:
        return self.left_border_x

    def get_right_border(self) -> int:
        return self.right_border_x