import cv2
import numpy as np
import math

class Draw:
    @staticmethod
    def vertical_line(image: np.ndarray, x: int | float, color: tuple[int, int, int]) -> np.ndarray:
        x = int(x)
        height, width = image.shape[:2]
        thickness = max(1, width // 800)
        cv2.line(image, (x, 0), (x, height), color, thickness)
        return image

    @staticmethod
    def text(image: np.ndarray, x: int | float, y: int | float,
             text: str, color: tuple[int, int, int]) -> np.ndarray:
        x = int(x)
        y = int(y)
        height, width = image.shape[:2]
        font_scale = max(1, width // 1000)
        thickness = max(1, width // 800)
        text_size_x, text_size_y = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
        x -= text_size_x // 2
        y -= text_size_y // 2
        return cv2.putText(image, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)
    
    @staticmethod
    def texts(image: np.ndarray, x: int | float, y: int | float,
                   texts: list[str], color: tuple[int, int, int], direction:int=1) -> np.ndarray:
        x = int(x)
        y = int(y)
        height, width = image.shape[:2]
        font_scale = max(1, min(width, height) // 1000)
        thickness = max(1, min(width, height) // 800)

        for i, text in enumerate(texts):
            text_size_x, text_size_y = cv2.getTextSize(texts[0], cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
            y_gap = i*text_size_y*1.5*direction
            Draw.text(image, x, y+y_gap, text, color)

        return image

    @staticmethod
    def vector_absolute(image: np.ndarray, x: int, y: int, vector_x: float, vector_y: float,
                        color: tuple[int, int, int], size_multiplier: float = 1.0) -> np.ndarray:
        height, width = image.shape[:2]
        thickness = max(1, width // 500)

        end_x = int(x + vector_x*size_multiplier)
        end_y = int(y + vector_y*size_multiplier)
        image = cv2.arrowedLine(image, (x, y), (end_x, end_y), color, thickness, tipLength=0.1)

        vector_amplitude = int(math.hypot(vector_x, vector_y))
        center_x = (end_x+x) // 2
        center_y = (end_y+y) // 2
        return Draw.text(image, center_x, center_y, f'{vector_amplitude}', (255, 255, 255))

    @staticmethod
    def vector_normalized(image: np.ndarray, normalized_x: float, normalized_y: float,
                          normalized_vx: float, normalized_vy: float,
                          color: tuple[int, int, int], size_multiplier: float = 1.0) -> np.ndarray:
        height, width = image.shape[:2]
        absolute_x = int(normalized_x * width)
        absolute_y = int(normalized_y * height)
        absolute_vector_x = normalized_vx * width
        absolute_vector_y = normalized_vy * height
        return Draw.vector_absolute(image, absolute_x, absolute_y,
                                    absolute_vector_x, absolute_vector_y, color, size_multiplier)
