import numpy as np
import cv2

from .setting_loader import SettingLoader as SL
from ultralytics import YOLO

class Detector:
    '''Detector that uses YOLOv8 model to find basket'''
    def __init__(self):
        settings = SL()['detectors']['ai']
        model_path = settings['model_path']
        self.score_threshold = settings['score_threshold']
        self.iou_threshold = settings['iou_threshold']
        self.target_size = settings['model_size']
        self.model = YOLO(model_path)

        self.last_box = None

    def find(self, image: np.ndarray) -> list[float] | None:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)

        results = self.model.predict(
            source=image,
            imgsz=self.target_size,
            conf=self.score_threshold,
            iou=self.iou_threshold,
            save=False,
            save_txt=False,
            save_conf=False,
            verbose=False,
            device=''
        )

        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    self.last_box = box
                    x_min, y_min, x_max, y_max = box.xyxyn[0].tolist()
                    center_x = (x_min + x_max) / 2
                    center_y = (y_min + y_max) / 2
                    return [center_x, center_y]
        return None

    def draw(self, image: np.ndarray) -> np.ndarray:
        if self.last_box is None:
            return image

        x_min, y_min, x_max, y_max = self.last_box.xyxyn[0].tolist()
        center_x = (x_min + x_max) / 2
        center_y = (y_min + y_max) / 2

        h, w = image.shape[:2]
        abs_center_x = int(center_x * w)
        abs_center_y = int(center_y * h)

        conf = float(self.last_box.conf)

        abs_x_min = int(x_min * w)
        abs_y_min = int(y_min * h)
        abs_x_max = int(x_max * w)
        abs_y_max = int(y_max * h)
        cv2.rectangle(image, (abs_x_min, abs_y_min), (abs_x_max, abs_y_max), (0, 255, 0), 2)

        text_y_start = abs_y_max + 20
        spacing = 30

        cv2.putText(image, f'Norm Pos: ({center_x:.2f}, {center_y:.2f})', 
                    (abs_x_min, text_y_start), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.putText(image, f'Abs Pos: ({abs_center_x}, {abs_center_y})', 
                    (abs_x_min, text_y_start + spacing), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.putText(image, f'Confidence: {conf:.2f}', 
                    (abs_x_min, text_y_start + 2 * spacing), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        return image
